import math
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

class BaselineModel():
    def __init__(self, args):

        # Placeholder for states and control inputs
        self.x = tf.Variable(np.zeros((args.batch_size*args.seq_length, args.n_x, args.n_y, 4), dtype=np.float32), trainable=False, name="state_values")
        self.params = tf.Variable(np.zeros((args.batch_size, args.param_dim), dtype=np.float32), trainable=False, name="parameter_values")

        # Parameters to be set externally
        self.learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")
        self.kl_weight = tf.Variable(0.0, trainable=False, name="kl_weight")
        self.param_weight = tf.Variable(0.0, trainable=False, name="param_weight")
        self.grad_weight = tf.Variable(0.0, trainable=False, name="grad_weight")
        self.generative = tf.Variable(False, trainable=False, name="generate_flag")

        # Load and assign mask for points inside cylinder
        pt_mask = np.load('double_cyl_mask.npy')
        self.cyl_mask =tf.constant(np.tile(pt_mask, (args.seq_length, 1, 1, 1)), dtype=np.float32)

        # Get arrays to be used for calculating lift/drag
        self.mask_reshape = np.tile(np.load('mask_reshape.npy'), (args.seq_length, 1))
        self.pressure_idxs_top = np.load('pt_idxs_top.npy')
        self.pressure_idxs_bottom = np.load('pt_idxs_bottom.npy')
        self.thetas = np.linspace(0, 2*np.pi, 200)
        self.dtheta = self.thetas[1] - self.thetas[0]
        self.sines = np.sin(self.thetas)
        self.cosines = np.cos(self.thetas)

        # Define step sizes (NOTE: it's confusing how I've defined x and y so far)
        self.h_x = 8.0/(args.n_x-1)
        self.h_y = 16.0/(args.n_y-1)

        # Normalization parameters to be stored
        self.shift = tf.Variable(np.zeros(4), trainable=False, name="state_shift", dtype=tf.float32)
        self.scale = tf.Variable(np.zeros(4), trainable=False, name="state_scale", dtype=tf.float32)
        self.shift_params = tf.Variable(np.zeros(args.param_dim), trainable=False, name="params_shift", dtype=tf.float32)
        self.scale_params = tf.Variable(np.zeros(args.param_dim), trainable=False, name="params_scale", dtype=tf.float32)

        # Get list of available GPUs
        devices = [d.name for d in device_lib.list_local_devices() if ':GPU' in d.name]
        
        # Create the computational graph
        with tf.device(devices[0]):
            self._create_encoder(args)
            self._infer_encodings(args)
            self._create_dynamics_network_params(args)
            self._simulate_dynamics(args)
            self._propagate_solution(args)
        with tf.device(devices[-1]):
            self._create_decoder(args)
            self._create_reconstructor_params(args)
            self._create_optimizer(args)

    # Define conv operation depending on number of dimensions in input
    def _conv_operation(self, in_tensor, num_filters, kernel_size, args, name, stride=1):
        return tf.layers.conv2d(in_tensor, 
                                    num_filters, 
                                    kernel_size=kernel_size,
                                    strides=stride, 
                                    padding='same', 
                                    name=name, 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))

    # Define conv operation depending on number of dimensions in input
    def _conv_operation_transpose(self, in_tensor, num_filters, kernel_size, args, name, stride=1):
        return tf.layers.conv2d_transpose(in_tensor, 
                            num_filters, 
                            kernel_size=kernel_size,
                            strides=stride, 
                            padding='same', 
                            name=name, 
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))


    # Code for initalizing resnet layer in encoder
    # Based on https://arxiv.org/pdf/1603.05027.pdf 
    # order is BN -> activation -> weights -> BN -> activation -> weights
    def _create_bottleneck_layer(self, in_tensor, args, name, num_filters):
        bn_output1 = tf.layers.batch_normalization(in_tensor, training=True)
        act_output1 = tf.nn.relu(bn_output1)
        conv_output1 = self._conv_operation(act_output1, num_filters/2, 1, args, name+'_conv1')

        bn_output2 = tf.layers.batch_normalization(conv_output1, training=True)
        act_output2 = tf.nn.relu(bn_output2)
        conv_output2 = self._conv_operation(act_output2, num_filters/2, 3, args, name+'_conv2')

        bn_output3 = tf.layers.batch_normalization(conv_output2, training=True)
        act_output3 = tf.nn.relu(bn_output3)
        bottleneck_output = self._conv_operation(act_output3, num_filters, 1, args, name+'_conv3')

        return bottleneck_output

    # Create encoder (maps state -> distribution over latent states)
    def _create_encoder(self, args):
        # Series of downconvolutions and bottleneck layers
        downconv_input = self.x
        
        # Determine how many possible downsamples to perform in each direction
        self.down_x = int(math.log2(args.n_x))-2
        self.down_y = int(math.log2(args.n_y))-2

        for i in range(len(args.num_filters)):
            stride = (1 + 1*(i < self.down_x), 1 + 1*(i < self.down_y))
            downconv_output = self._conv_operation(downconv_input, args.num_filters[i], 3, args, 'extractor_downconv'+str(i), stride=stride)
            bottleneck_output = self._create_bottleneck_layer(downconv_output, args, 'extractor_bn'+str(i), args.num_filters[i])
            downconv_input = downconv_output + bottleneck_output
        self.extractor_conv_output = tf.nn.relu(downconv_input)

        # Fully connected layer to get features
        self.reshape_output = tf.reshape(self.extractor_conv_output, [args.batch_size*args.seq_length, -1])
        features = tf.layers.dense(self.reshape_output, 
                                    2*args.latent_dim,
                                    name='to_features', 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.z_dists = tf.reshape(features, [args.batch_size, args.seq_length, 2*args.latent_dim])

        # Nonlinear transformation of parameter values to higher-dimensional representation
        self.param_w = tf.ones([args.param_dim, args.latent_dim])
        hidden = tf.layers.dense(self.params, 
                                    units=args.transform_size, 
                                    activation=tf.nn.relu, 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.params_transform = tf.layers.dense(hidden, 
                                        units=args.latent_dim,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))

    # Function to generate samples given distribution parameters
    def _gen_sample(self, args, dist_params):
        g_mean, g_logstd = tf.split(dist_params, [args.latent_dim, args.latent_dim], axis=1)

        # Make standard deviation estimates better conditioned, otherwise could be problem early in training
        g_std = tf.minimum(tf.exp(g_logstd) + 1e-3, 10.0) 
        samples = tf.random_normal([args.batch_size, args.latent_dim])
        g = samples*g_std + g_mean
        return g

    # Step through time and sample from posterior distributions
    def _infer_encodings(self, args):
        # Sample value for initial observation from distribution
        z_t = self._gen_sample(args, self.z_dists[:, 0])

        # Start list of g-distributions and sampled values
        z_vals = [z_t]

        for t in range(1, args.seq_length):
            z_t = self._gen_sample(args, self.z_dists[:, t])
            z_vals.append(z_t)

        # Finally, stack inferred observations
        self.z_vals = tf.reshape(tf.stack(z_vals, axis=1), [args.batch_size, args.seq_length, args.latent_dim])

    # Create parameters to comprise dynamics network
    def _create_dynamics_network_params(self, args):
        # Create RNN for predicting evolution of latent state
        self.W_g_out_dyn = tf.get_variable("w_g_out_dyn", [args.rnn_size, args.transform_size], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.b_g_out_dyn = tf.get_variable("b_g_out_dyn", [args.transform_size])
        self.W_to_g_enc_dyn = tf.get_variable("w_to_g_enc_dyn", [args.transform_size, args.latent_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.b_to_g_enc_dyn = tf.get_variable("b_to_g_enc_dyn", [args.latent_dim])

        # Initialize single-layer GRU network to create observation encodings
        self.cell_dyn = tf.nn.rnn_cell.GRUCell(args.rnn_size, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dyn_gru')
        self.rnn_state_dyn = self.cell_dyn.zero_state(args.batch_size, tf.float32)

        self.dyn_w = []
        self.dyn_b = []

        # Loop through elements of inference network and define parameters
        for i in range(len(args.prior_size)):
            if i == 0:
                prev_size = 2*args.latent_dim
            else:
                prev_size = args.prior_size[i-1]
            self.dyn_w.append(tf.get_variable("dyn_w"+str(i), [prev_size, args.prior_size[i]], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
            self.dyn_b.append(tf.get_variable("dyn_b"+str(i), [args.prior_size[i]]))

        # Last set of weights to map to output
        self.dyn_w.append(tf.get_variable("dyn_w_end", [args.prior_size[-1], args.latent_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.dyn_b.append(tf.get_variable("dyn_b_end", [args.latent_dim]))

    # Function to get prior distribution over g
    def _get_dyn_prediction(self, args, g_t):
        # Get encoding of previous g-values
        self.rnn_output_dyn, self.rnn_state_dyn = self.cell_dyn(g_t, self.rnn_state_dyn)
        hidden = tf.nn.relu(tf.nn.xw_plus_b(self.rnn_output_dyn, self.W_g_out_dyn, self.b_g_out_dyn))
        dyn_enc = tf.nn.xw_plus_b(hidden, self.W_to_g_enc_dyn, self.b_to_g_enc_dyn)

        dyn_input = tf.concat([dyn_enc, self.params_transform], axis=1)
        for i in range(len(args.prior_size)):
            dyn_input = tf.nn.relu(tf.nn.xw_plus_b(dyn_input, self.dyn_w[i], self.dyn_b[i]))
        dyn_pred = tf.nn.xw_plus_b(dyn_input, self.dyn_w[-1], self.dyn_b[-1])
        return dyn_pred

    # Construct network and generate paramaters for conditional prior distributions
    def _simulate_dynamics(self, args):        
        # Reset initial state
        dyn_preds = []
        zval_t =  self.z_vals[:, 0]
        zvals_pred = [zval_t]
        for t in range(args.seq_length-1):
            zval_t = self._get_dyn_prediction(args, zval_t)
            zvals_pred.append(zval_t)
        self.zvals_pred = tf.reshape(tf.stack(zvals_pred, axis=1), [args.batch_size, args.seq_length, args.latent_dim])

        # Construct prior params for z1 as a function of flow params
        hidden = tf.layers.dense(self.params_transform, 
                                    units=args.transform_size, 
                                    activation=tf.nn.relu, 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        z1_prior = tf.layers.dense(hidden, 
                                        units=2*args.latent_dim,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.z1_prior = tf.expand_dims(z1_prior, axis=1)

        # Combine and reshape to get full set of prior distribution parameter values
        self.z_prior = tf.tile(self.z1_prior, [1, args.seq_length, 1])
        self.z_prior = tf.reshape(self.z_prior, [args.batch_size*args.seq_length, 2*args.latent_dim])

    # Sample from prior and recursively simulate
    def _sample_dyn_zvals(self, args):
        # Sample initial z-value
        z_t = self._gen_sample(args, self.z1_prior[:, 0])

        # Start list of g-samples
        z_sample_vals = [z_t]

        # Loop through time and calculate new values
        for t in range(1, args.n_gen_seq*args.seq_length):
            z_t = self._get_dyn_prediction(args, z_t)

            # Add sampled value to list
            z_sample_vals.append(z_t)

        return tf.reshape(tf.stack(z_sample_vals, axis=1), [args.batch_size, args.n_gen_seq*args.seq_length, args.latent_dim])

    # Perform least squares to get A-matrix and propagate forward
    def _propagate_solution(self, args):
        # Use inferred g_vals or values drawn from prior depending on whether model is being used in generative fashion (CLEAN UP)
        self.gen_g = self._sample_dyn_zvals(args)
        self.z_decode = tf.cond(tf.cast(args.finetune, dtype=tf.bool), lambda: self.zvals_pred, lambda: self.z_vals)
        self.decode_vals_reshape = tf.cond(self.generative, lambda: self.gen_g[:, :args.seq_length], lambda: self.z_decode)

        # Reshape predicted z-values
        self.decode_vals = tf.reshape(self.decode_vals_reshape, [args.batch_size*args.seq_length, args.latent_dim])

    # Run z-values through decoder
    def _create_decoder(self, args):
        # Reverse fully connected layer and reshape
        self.rev_fc_output = tf.layers.dense(self.decode_vals, 
                                            self.reshape_output.get_shape().as_list()[-1], 
                                            activation=tf.nn.relu,
                                            name='from_code', 
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        conv_shape = self.extractor_conv_output.get_shape().as_list()
        upconv_output = tf.reshape(self.rev_fc_output, conv_shape)

        # Series of bottleneck layers and upconvolutions
        # Specify number of filter after each upconv (last upconv needs to have the same number of channels as input)
        num_filters_upconv = [5] + args.num_filters
        for i in range(len(args.num_filters)-1,-1,-1):
            stride = (1 + 1*(i < self.down_x), 1 + 1*(i < self.down_y))
            bottleneck_output = self._create_bottleneck_layer(upconv_output, args, 'bn_decode'+str(i), num_filters_upconv[i+1])
            upconv_output += bottleneck_output
            upconv_output = self._conv_operation_transpose(upconv_output, num_filters_upconv[i], 3, args, 'upconv'+str(i), stride=stride)

        # Ouput of upconvolutions is reconstructed solution
        output_shape = upconv_output.get_shape().as_list()
        self.x_pred_norm = self.rec_sol = tf.slice(upconv_output, [0, (output_shape[1] - args.n_x)//2, (output_shape[2] - args.n_y)//2, 0],\
                                                                     [args.batch_size*args.seq_length, args.n_x, args.n_y, 4])

    # Create parameters to comprise prior network
    def _create_reconstructor_params(self, args):
        self.rec_w = []
        self.rec_b = []

        self.rec_w.append(tf.get_variable("rec_w1", [args.latent_dim, args.transform_size], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.rec_b.append(tf.get_variable("rec_b1", [args.transform_size]))

        # Last set of weights to map to output
        self.rec_w.append(tf.get_variable("rec_w_end", [args.transform_size, args.param_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.rec_b.append(tf.get_variable("rec_b_end", [args.param_dim]))

    # Use finite difference to estimate spatial gradients of scalar arrays
    def _estimate_grad(self, in_array):
        grad_x = (in_array[:, :, 2:, :] - in_array[:, :, :-2, :])/2/self.h_x
        dxbound_1 = tf.expand_dims((in_array[:, :, 1, :] - in_array[:, :, 0, :])/self.h_x, axis=2)
        dxbound_2 = tf.expand_dims((in_array[:, :, -1, :] - in_array[:, :, -2, :])/self.h_x, axis=2)
        grad_x = tf.concat([dxbound_1, grad_x, dxbound_2], axis=2)
        
        grad_y = (in_array[:, :, :, 2:] - in_array[:, :, :, :-2])/2/self.h_y
        dybound_1 = tf.expand_dims((in_array[:, :, :, 1] - in_array[:, :, :, 0])/self.h_y, axis=3)
        dybound_2 = tf.expand_dims((in_array[:, :, :, -1] - in_array[:, :, :, -2])/self.h_y, axis=3)
        grad_y = tf.concat([dybound_1, grad_y, dybound_2], axis=3)
        
        return tf.concat([grad_x, grad_y], axis=4)

    # Function to calculate lift and drag on top and bottom cylinder (NOTE: Currently assumes batch size of 1)
    def _calc_lift_drag(self, args, pressure_field):
        # Find pressure field and remove unwanted points
        pressure_field_reshape = tf.reshape(pressure_field, [args.seq_length, -1])
        pressure_masked = tf.boolean_mask(pressure_field_reshape, self.mask_reshape)
        pressure_masked = tf.reshape(pressure_masked, [args.seq_length, 58270])
        
        pressure_top = tf.stack([tf.gather(pressure_masked[t], self.pressure_idxs_top) for t in range(args.seq_length)])
        pressure_bottom = tf.stack([tf.gather(pressure_masked[t], self.pressure_idxs_bottom) for t in range(args.seq_length)])

        # Find pressure values on top and bottom in x- and y-directions
        hforce_top = -pressure_top*self.cosines
        drag_top = self.dtheta*(tf.reduce_sum(hforce_top, axis=1)) - self.dtheta/2.0*(hforce_top[:, 0] + hforce_top[:, -1])
        vforce_top = -pressure_top*self.sines
        lift_top = self.dtheta*(tf.reduce_sum(vforce_top, axis=1)) - self.dtheta/2.0*(vforce_top[:, 0] + vforce_top[:, -1])
        hforce_bottom = -pressure_bottom*self.cosines
        drag_bottom = self.dtheta*(tf.reduce_sum(hforce_bottom, axis=1)) - self.dtheta/2.0*(hforce_bottom[:, 0] + hforce_bottom[:, -1])
        vforce_bottom = -pressure_bottom*self.sines
        lift_bottom = self.dtheta*(tf.reduce_sum(vforce_bottom, axis=1)) - self.dtheta/2.0*(vforce_bottom[:, 0] + vforce_bottom[:, -1])
        
        return drag_top, lift_top, drag_bottom, lift_bottom

    # BerHu loss definition
    def _berhu_loss(self, abs_error, c=0.1):
        berHu_loss = tf.where(abs_error <= c, c*abs_error, tf.square(abs_error))
        return tf.reduce_sum(berHu_loss)

    # Create optimizer to minimize loss
    def _create_optimizer(self, args):
        # First extract mean and std for prior dists, dist over g, and dist over x
        z_prior_mean, z_prior_logstd = tf.split(self.z_prior, [args.latent_dim, args.latent_dim], axis=1)
        z_prior_std = tf.exp(z_prior_logstd) + 1e-3
        dists = tf.reshape(self.z_dists, [args.batch_size*args.seq_length, 2*args.latent_dim])
        z_mean, z_logstd = tf.split(dists, [args.latent_dim, args.latent_dim], axis=1)
        z_std = tf.exp(z_logstd) + 1e-3

        # First component of loss: NLL of observed states
        reshape_dim = [args.batch_size, args.seq_length, args.n_x, args.n_y, 4]
        x_norm_reshape = tf.reshape(self.x, reshape_dim)
        x_reshape = x_norm_reshape*self.scale + self.shift
        self.x_pred_norm_reshape = tf.reshape(self.x_pred_norm, reshape_dim)
        self.x_pred_reshape = self.x_pred_norm_reshape*self.scale + self.shift

        # Replace values in cylinder
        self.x_pred_reshape = self.x_pred_reshape*self.cyl_mask + x_reshape*(1.0 - self.cyl_mask)

        # Prediction loss
        self.pred_loss = self._berhu_loss(tf.abs(x_reshape - self.x_pred_reshape))/args.batch_size

        # Add in loss for time derivative to encourage smoothness
        x_diff = x_reshape[:, 1:] - x_reshape[:, :-1]
        x_pred_diff = self.x_pred_reshape[:, 1:] - self.x_pred_reshape[:, :-1]
        self.pred_loss += self._berhu_loss(tf.abs(x_diff - x_pred_diff))/args.batch_size

        # Add in loss for spatial deriv
        x_grad = self._estimate_grad(x_reshape)
        x_pred_grad = self._estimate_grad(self.x_pred_reshape)
        self.grad_loss = tf.reduce_sum(tf.square(x_grad - x_pred_grad))/args.batch_size

        drag_top, lift_top, drag_bottom, lift_bottom = self._calc_lift_drag(args, x_reshape[:, :, :, :, 3])
        drag_top_pred, lift_top_pred, drag_bottom_pred, lift_bottom_pred = self._calc_lift_drag(args, self.x_pred_reshape[:, :, :, :, 3])
        lift_drag_array = tf.concat([drag_top, lift_top, drag_bottom, lift_bottom], axis=0)
        lift_drag_pred_array = tf.concat([drag_top_pred, lift_top_pred, drag_bottom_pred, lift_bottom_pred ], axis=0)
        # self.pred_loss += 1000.0*self._berhu_loss(tf.abs(lift_drag_array - lift_drag_pred_array))
        lift_drag_diff = lift_drag_array[1:] - lift_drag_array[:-1]
        lift_drag_pred_diff = lift_drag_pred_array[1:] - lift_drag_pred_array[:-1]
        # self.pred_loss += 1000.0*self._berhu_loss(tf.abs(lift_drag_diff - lift_drag_pred_diff))

        # Second component of loss: KLD between approximate posterior and prior
        z_prior_dist = tf.distributions.Normal(loc=z_prior_mean, scale=z_prior_std)
        z_dist = tf.distributions.Normal(loc=z_mean, scale=z_std)
        self.kl_loss = tf.reduce_sum(tf.distributions.kl_divergence(z_dist, z_prior_dist))/args.batch_size

        # Attempt to reconstruct params based on prior means at each time step
        self.param_loss = 0.0
        prior_reshape = tf.reshape(self.z_prior, [args.batch_size, args.seq_length, 2*args.latent_dim])
        for t in range(args.seq_length):
            rec_input = self._gen_sample(args, prior_reshape[:, t])
            hidden = tf.nn.relu(tf.nn.xw_plus_b(rec_input, self.rec_w[0], self.rec_b[0]))
            self.param_pred = tf.nn.xw_plus_b(hidden, self.rec_w[1], self.rec_b[1])
            self.param_loss += self._berhu_loss(tf.abs(self.params - self.param_pred))/args.seq_length

        # Sum with regularization losses to form total cost
        self.cost = self.pred_loss + self.kl_weight*self.kl_loss + self.param_weight*self.param_loss + tf.reduce_sum(tf.losses.get_regularization_losses())

        # Perform parameter update on encoder/decoder
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tvars = [v for v in tf.trainable_variables() if 'dyn' not in v.name]
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars, colocate_gradients_with_ops=True), args.grad_clip)
        self.train = optimizer.apply_gradients(zip(self.grads, tvars))

        # Define loss for dynamics network
        if args.finetune:
            self.pred_loss = self._berhu_loss(tf.abs(self.z_vals - self.zvals_pred))/args.batch_size
        self.cost_dyn = self.pred_loss

        # Perform parameter update on dynamics network
        tvars_prior = [v for v in tf.trainable_variables() if 'dyn' in v.name]
        self.grads_prior, _ = tf.clip_by_global_norm(tf.gradients(self.cost_dyn, tvars_prior, colocate_gradients_with_ops=True), args.grad_clip)
        self.train_prior = optimizer.apply_gradients(zip(self.grads_prior, tvars_prior))