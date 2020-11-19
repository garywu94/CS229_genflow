import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from gen_model_3d import GenModel3d
from generative_model import GenerativeModel
from baseline_model import BaselineModel
import numpy as np
from data_loader import DataLoader
import tensorflow as tf
import time
import random
from utils import generate_solutions, generate_solutions_3d, view_solutions, view_solutions_3d, generate_time_avg, visualize_dataset, find_time_variation

def main():
    ######################################
    #          General Params            #
    ######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',       type=str,   default='checkpoints', help='directory to store checkpointed models')
    parser.add_argument('--val_frac',       type=float, default=0.1,        help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name',      type= str,  default='',         help='name of checkpoint file to load (blank means none)')
    parser.add_argument('--save_name',      type= str,  default='koopman_model', help='name of checkpoint files for saving')

    parser.add_argument('--seq_length',     type=int,   default= 32,        help='sequence length for training') #32
    parser.add_argument('--batch_size',     type=int,   default= 1,         help='minibatch size')
    parser.add_argument('--latent_dim',     type=int,   default= 32,        help='dimensionality of code')
    parser.add_argument('--param_dim',      type=int,   default= 1,         help='dimensionality of flow parameters') # 1
    parser.add_argument('--start_epoch',    type=int,   default= 0,         help='epoch of training in which to start')
    parser.add_argument('--start_kl',       type=int,   default= 12,        help='epoch of training in which to start enforcing KL penalty')
    parser.add_argument('--anneal_time',    type=int,   default= 3,         help='number of epochs over which to anneal KLD')

    parser.add_argument('--num_epochs',     type=int,   default= 100,        help='number of epochs')
    parser.add_argument('--learning_rate',  type=float, default= 0.0005,    help='learning rate')
    parser.add_argument('--decay_rate',     type=float, default= 0.5,       help='decay rate for learning rate')
    parser.add_argument('--l2_regularizer', type=float, default= 15.0,      help='regularization for least squares')
    parser.add_argument('--grad_clip',      type=float, default= 5.0,       help='clip gradients at this value')
    parser.add_argument('--kl_weight',      type=float, default= 1.0,       help='weight applied to kl-divergence loss')
    parser.add_argument('--grad_weight',    type=float, default= 100.0,       help='weight applied to velocity gradient loss')
    parser.add_argument('--param_weight',   type=float, default= 100.0,     help='weight applied to param prediction loss')
    parser.add_argument('--prior_weight',   type=float, default= 1.0,       help='weight applied to error in samples from prior')
    parser.add_argument('--disjoint_training', type=bool,  default=False,   help='whether feature extractor is trained separately from the rest of network')
    parser.add_argument('--train_baseline', type=bool,  default=False,      help='whether to train baseline model')
    parser.add_argument('--finetune',       type=bool,  default=False,      help='whether to finetune based on preds from prior')
 
    ######################################
    #          Data-loading              #
    ######################################
    parser.add_argument('--data_dir',           type=str,   default='./3d_data/', help='directory containing cylinder data')
    parser.add_argument('--n_sequences',        type=int,   default= 1200,      help='number of files to load for training')
    parser.add_argument('--min_num',            type=int,   default= 1,         help='lowest number time snapshot to load for training')
    parser.add_argument('--max_num',            type=int,   default= 940,      help='highest number time snapshot to load for training')
    parser.add_argument('--start_file',         type=int,   default= 100,       help='first file number to load for training')
    parser.add_argument('--stagger',            type=int,   default= 1,         help='number of time steps between training examples')
    parser.add_argument('--data_3d',            type=bool,  default=False,      help='whether to use 3d fluid data')
    parser.add_argument('--n_coeff',            type=int,   default= 21,        help='number of fourier coefficients to model')
    parser.add_argument('--n_x',                type=int,   default= 128,       help='number of points in x-direction')
    parser.add_argument('--n_y',                type=int,   default= 64,        help='number of points in y-direction')
    parser.add_argument('--n_z',                type=int,   default= 32,        help='number of points in z-direction')

    ######################################
    #          Network Params            #
    ######################################
    parser.add_argument('--num_filters', nargs='+',   type=int, default=[32],     help='number of filters after each down/uppconv')
    parser.add_argument('--inference_size', nargs='+', type=int, default=[32],    help='hidden layer sizes in feature inference network')
    parser.add_argument('--prior_size',     nargs='+', type=int, default=[32],    help='hidden layer sizes in prior network')
    parser.add_argument('--rnn_size',       type=int,   default= 64,        help='size of RNN layers')
    parser.add_argument('--transform_size', type=int,   default= 64,        help='size of transform layers')
    parser.add_argument('--reg_weight',     type=float, default= 1e-4,      help='weight applied to regularization losses')
    parser.add_argument('--linear_model',   type=bool,  default=False,      help='whether to enforce that learned dynamics are linear')
    parser.add_argument('--seed',           type=int,   default= 1,         help='random seed for sampling operations')

    ######################################
    #         Generative Params          #
    ######################################
    parser.add_argument('--generate_solns', type=bool,  default=False,      help='whether to use model in generative fashion')
    parser.add_argument('--n_gen_seq',      type=int,   default=1,          help='number of sequences to generate')
    parser.add_argument('--generate_time_avg', type=bool,  default=False,      help='whether to use model in generative and find time average')
    parser.add_argument('--reynolds_num',   type=float, default=350,         help='reynolds number at which to generate solutions')
    parser.add_argument('--omega',          type=float, default=0.0,         help='angular velocity on cylinder')

    args = parser.parse_args()

    # Set random seed
    random.seed(1)

    # Construct model
    if args.data_3d:
        net = GenModel3d(args)
    elif args.train_baseline:
        net = BaselineModel(args)
    else:
        net = GenerativeModel(args)

    # Begin training or generate solutions
    if args.generate_solns:
        if args.data_3d:
            generate_solutions_3d(args, net, re=args.reynolds_num)
        else:
            generate_solutions(args, net, re=args.reynolds_num, om=args.omega)
    elif args.generate_time_avg:
        if args.data_3d:
            generate_time_avg(args, net, re=args.reynolds_num)
        else:
            find_time_variation(args, net)
    else:
        train(args, net)

# Train network
def train(args, net):
    # Begin tf session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))

        # Generate data
        shift = sess.run(net.shift)
        scale = sess.run(net.scale)
        shift_params = sess.run(net.shift_params)
        scale_params = sess.run(net.scale_params)

        # Generate training data
        data_loader = DataLoader(args, shift, scale, shift_params, scale_params)

        #Function to evaluate loss on validation set
        def val_loss(kl_weight, grad_weight):
            data_loader.reset_batchptr_val()
            loss = 0.0
            for b in range(data_loader.n_batches_val):
                # Get inputs
                batch_dict = data_loader.next_batch_val()
                x = batch_dict['states']
                params = batch_dict['params']

                # Construct inputs for network
                feed_in = {}
                if args.data_3d:
                    feed_in[net.x] = np.reshape(x, (args.batch_size*args.seq_length, args.n_x, args.n_y, args.n_z, 5))
                else:
                    #feed_in[net.x] = np.reshape(x, (args.batch_size*args.seq_length, args.n_x, args.n_y, 4))
                    feed_in[net.x] = np.reshape(x, (args.batch_size * args.seq_length, args.n_x, args.n_y, 6))
                    feed_in[net.param_weight] = args.param_weight
                feed_in[net.params] = params
                feed_in[net.kl_weight] = kl_weight
                feed_in[net.grad_weight] = grad_weight

                # Find loss
                if args.finetune and args.train_baseline:
                    feed_out = net.cost_dyn
                else:
                    feed_out = net.cost
                cost = sess.run(feed_out, feed_in)
                loss += cost

            return loss/data_loader.n_batches_val


        # Store normalization parameters
        sess.run(tf.assign(net.shift, data_loader.shift_x))
        sess.run(tf.assign(net.scale, data_loader.scale_x))
        sess.run(tf.assign(net.shift_params, data_loader.shift_params))
        sess.run(tf.assign(net.scale_params, data_loader.scale_params))

        # Initialize variable to track validation score over time
        old_score = 1e20
        best_score = 1e20

        # Set initial learning rate and weight on kl divergence
        print('setting learning rate to ', args.learning_rate)
        sess.run(tf.assign(net.learning_rate, args.learning_rate))
        lr = args.learning_rate

        # Define temperature for annealing kl_weight
        T = args.anneal_time*data_loader.n_batches_train

        # Define counting variables
        count = max(0, (args.start_epoch - args.start_kl)*data_loader.n_batches_train)
        count_decay = 0
        decay_epochs = []

        # Set initial learning rate
        lr = args.learning_rate

        # Evaluate loss on validation set
        score = val_loss(args.kl_weight, args.grad_weight)
        print('Validation Loss: {0:f}'.format(score))

        # Loop over epochs
        for e in range(args.start_epoch, args.num_epochs):
            if args.data_3d:
                view_solutions_3d(args, net, sess, re=args.reynolds_num)
            elif args.train_baseline:
                if args.finetune:
                    view_solutions(args, net, sess)
                else:
                    pass
            else:
                view_solutions(args, net, sess)

            # Initialize loss
            loss = 0.0
            kl_loss = 0.0
            pred_loss = 0.0
            param_loss = 0.0
            grad_loss = 0.0
            loss_count = 0
            b = 0
            data_loader.reset_batchptr_train()

            # Loop over batches
            while b < data_loader.n_batches_train:
                start = time.time()

                # Update kl_weight and grad_weight
                kl_weight = args.kl_weight
                grad_weight = args.grad_weight
                param_weight = min(args.param_weight, args.param_weight*((e-2)*data_loader.n_batches_train + b)/float(T))
                param_weight = max(param_weight, 0.0)
                if e < args.start_kl:
                    kl_weight = min(args.kl_weight, 1e-3)
                else:
                    count += 1
                    kl_weight = min(args.kl_weight, 1e-3 + args.kl_weight*count/float(T))

                # Get inputs
                batch_dict = data_loader.next_batch_train()
                x = batch_dict['states']
                params = batch_dict['params']

                feed_in = {}
                if args.data_3d:
                    feed_in[net.x] = np.reshape(x, (args.batch_size*args.seq_length, args.n_x, args.n_y, args.n_z, 5))
                else:
                    feed_in[net.x] = np.reshape(x, (args.batch_size*args.seq_length, args.n_x, args.n_y, 4))
                feed_in[net.param_weight] = param_weight
                feed_in[net.params] = params
                feed_in[net.kl_weight] = kl_weight
                feed_in[net.grad_weight] = grad_weight
                    
                # Find loss and perform training operation
                feed_out = [net.cost, net.kl_loss, net.pred_loss, net.param_loss, net.train]
                if args.finetune: feed_out[-1] = net.train_prior
                feed_out += [net.grad_loss]
                out = sess.run(feed_out, feed_in, options=tf.RunOptions(report_tensor_allocations_upon_oom=True))

                # Update and display cumulative losses
                loss += out[0]
                kl_loss += out[1]
                pred_loss += out[2]
                param_loss += out[3]
                grad_loss += out[-1]
                loss_count += 1

                end = time.time()
                b += 1

                # Print loss
                if (e * data_loader.n_batches_train + b) % 10 == 0 and b > 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b, args.num_epochs * data_loader.n_batches_train,
                              e, loss/loss_count, end - start))
                    print("{}/{} (epoch {}), pred_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b, args.num_epochs * data_loader.n_batches_train,
                              e, pred_loss/loss_count, end - start))
                    print("{}/{} (epoch {}), kl_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b, args.num_epochs * data_loader.n_batches_train,
                              e, kl_loss/loss_count, end - start))
                    print("{}/{} (epoch {}), param_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b, args.num_epochs * data_loader.n_batches_train,
                              e, param_loss/loss_count, end - start))
                    print("{}/{} (epoch {}), grad_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b, args.num_epochs * data_loader.n_batches_train,
                              e, grad_loss/loss_count, end - start))

                    print('')
                    loss = 0.0
                    kl_loss = 0.0
                    pred_loss = 0.0
                    param_loss = 0.0
                    grad_loss = 0.0
                    loss_count = 0

            # Evaluate loss on validation set
            score = val_loss(args.kl_weight*(e >= args.start_kl), args.grad_weight)
            print('Validation Loss: {0:f}'.format(score))
            b = 0

            # Set learning rate
            if (old_score - score) < -0.01 and e != args.start_kl:
                # Don't let learning rate decay too much before KLD penalty is applied
                if (args.learning_rate * (args.decay_rate ** count_decay) <= 7.5e-5) and e < (args.start_kl + args.anneal_time):
                    lr = 7.5e-5
                else:
                    count_decay += 1
                    decay_epochs.append(e)
                    if len(decay_epochs) >= 3 and np.sum(np.diff(decay_epochs)[-2:]) == 2: break
                    lr = args.learning_rate * (args.decay_rate ** count_decay)
                if lr < 1e-5: break
                print('setting learning rate to ', lr)
                sess.run(tf.assign(net.learning_rate, lr))
            print('learning rate is set to ', lr)

            # Save model every epoch
            checkpoint_path = os.path.join(args.save_dir, args.save_name + '.ckpt')
            saver.save(sess, checkpoint_path, global_step = e)
            print("model saved to {}".format(checkpoint_path))

            # Restart best score calculations when KLD penalty is applied
            if e == args.start_kl: best_score = score

            if score <= best_score:
                checkpoint_path = os.path.join('best_' + args.save_dir, args.save_name + '.ckpt')
                saver.save(sess, checkpoint_path, global_step = e)
                print("model saved to {}".format(checkpoint_path))
                best_score = score
        
            old_score = score

if __name__ == '__main__':
    main()
