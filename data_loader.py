import h5py
import math
import numpy as np
import random
import progressbar

# Class to load and preprocess data
class DataLoader():
    def __init__(self, args, shift, scale, shift_params=None, scale_params=None):
        self.shift_x = shift
        self.scale_x = scale
        self.shift_params = shift_params
        self.scale_params = scale_params

        print('validation fraction: ', args.val_frac)

        print('loading data...')
        if args.data_3d:
            self._load_3d_data(args)
        else:
            self._load_jet_data(args)
            #self._load_double_cyl_data(args)
        self._create_inputs_targets(args)

        print('creating splits...')
        self._create_split(args)

        print('shifting/scaling data...')
        self._shift_scale(args)

    # Function to load subset of data at specified Reynolds number
    def _load_data_subset(self, args, n_sequences, re):
        # Define numbers of files that need to be loaded for training    
        n_files = n_sequences*args.seq_length
        max_gap = (args.max_num - args.stagger*args.seq_length)/n_sequences
        start_idxs = np.linspace(args.min_num, max_gap*n_sequences, n_sequences)
        file_nums = np.array([])
        for i in range(n_sequences):
            file_nums = np.concatenate([file_nums, np.linspace(start_idxs[i], start_idxs[i] + args.seq_length*args.stagger, args.seq_length)])

        # Define progress bar
        print('loading data for Reynolds number', re, '...')
        bar = progressbar.ProgressBar(maxval=n_files).start()

        # Load data
        x = np.zeros((n_files, 128, 256, 4), dtype=np.float32)
        for i in range(n_files):
            f = h5py.File(args.data_dir + 're' + str(re) + '/' + 'sol_data_'+str(int(file_nums[i])).zfill(4)+'.h5', 'r')
            x[i] = np.array(f['sol_data'])
            bar.update(i)
        bar.finish()

        return x

    def _load_data(self, args):
        # Define flow types to load from and number of sequences from each
        res = [50, 60, 70, 90, 110, 150]
        n_seq = args.n_sequences//len(res)

        # Initialize data array
        x = np.zeros((n_seq*len(res)*args.seq_length, 128, 256, 4), dtype=np.float32)
        params = np.zeros((n_seq*len(res), args.param_dim))

        # Loop through each flow condition and load data
        n_files = n_seq*args.seq_length
        for (i, re) in enumerate(res):
            x[i*n_files:(i+1)*n_files] = self._load_data_subset(args, n_seq, re)
            params[i*n_seq:(i+1)*n_seq] = re

        # Divide into sequences
        self.x = x.reshape(-1, args.seq_length, 128, 256, 4)
        self.x = self.x[:int(np.floor(len(self.x)/args.batch_size)*args.batch_size)]
        self.x = self.x.reshape(-1, args.batch_size, args.seq_length, 128, 256, 4)

        self.params = params[:int(np.floor(len(params)/args.batch_size)*args.batch_size)]
        self.params = self.params.reshape(-1, args.batch_size, args.param_dim)

    # Function to load subset of data at specified Reynolds number
    def _load_double_cyl_subset(self, args, n_sequences, params):
        # Define numbers of files that need to be loaded for training    
        n_files = n_sequences*args.seq_length
        max_gap = (args.max_num - args.stagger*args.seq_length - args.min_num)/n_sequences
        start_idxs = np.linspace(args.min_num, max_gap*n_sequences + args.min_num, n_sequences)
        file_nums = np.array([])
        for i in range(n_sequences):
            file_nums = np.concatenate([file_nums, np.linspace(start_idxs[i], start_idxs[i] + args.seq_length*args.stagger, args.seq_length)])

        # Define progress bar
        om = params[0]
        re = params[1]
        print('loading data for Reynolds number', re, 'omega number', om, '...')
        bar = progressbar.ProgressBar(maxval=n_files).start()

        # Load data
        x = np.zeros((n_files, args.n_x, args.n_y, 4), dtype=np.float32)
        for i in range(n_files):
            f = h5py.File(args.data_dir + 're' + str(int(re))  + '/om' + str(int(4*om)+1) + '/sol_data/' + 'sol_data_'+str(int(file_nums[i])).zfill(4)+'.h5', 'r')
            sol_data = np.array(f['sol_data'])

            # Find pressure values
            rho = sol_data[:, :, 0]
            u = sol_data[:, :, 1]/rho
            v = sol_data[:, :, 2]/rho
            e = sol_data[:, :, 3]
            P = 0.4*rho*(e - 0.5*(u**2 + v**2))
            
            # Replace x- and y-momentum with x- and y- velocity, energy with pressure
            sol_data[:, :, 1] = u
            sol_data[:, :, 2] = v
            sol_data[:, :, 3] = P

            # Concatenate and store data
            x[i] = sol_data
            bar.update(i)
        bar.finish()

        return x
    def _load_jet_data(self,args):
        # Define flow types to load from and number of sequences from each
        M_j  = [1.69]

        params = [np.array([ma]) for ma in M_j]
        n_seq = args.n_sequences // len(params)

        # Initialize data array
        x = np.zeros((n_seq * len(params) * args.seq_length, args.n_x, args.n_y, 6), dtype=np.single)
        parameters = np.zeros((n_seq * len(params), args.param_dim))

        # Loop through each flow condition and load data
        n_files = n_seq * args.seq_length
        f = h5py.File('nearfield_1_p1.h5', 'r')
        x = np.array(f['sol_data'])
        for (i, param) in enumerate(params):
            print('loading jet flow data for Mj = ', param )
            #x[i * n_files:(i + 1) * n_files] =
            parameters[i * n_seq:(i + 1) * n_seq] = param



        # Divide into sequences
        self.x = x.reshape(-1, args.seq_length, args.n_x, args.n_y, 6)
        self.x = self.x[:int(np.floor(len(self.x) / args.batch_size) * args.batch_size)]
        self.x = self.x.reshape(-1, args.batch_size, args.seq_length, args.n_x, args.n_y, 6)

        self.params = parameters[:int(np.floor(len(parameters) / args.batch_size) * args.batch_size)]
        self.params = self.params.reshape(-1, args.batch_size, args.param_dim)

    def _load_double_cyl_data(self, args):
        # Define flow types to load from and number of sequences from each
        oms = [0.25*om for om in range(11)]
        res = [75, 85, 100, 125, 150, 200]
        params = [np.array([om, re]) for om in oms for re in res]
        n_seq = args.n_sequences//len(params)

        # Initialize data array
        x = np.zeros((n_seq*len(params)*args.seq_length, args.n_x, args.n_y, 4), dtype=np.float32)
        parameters = np.zeros((n_seq*len(params), args.param_dim))

        # Loop through each flow condition and load data
        n_files = n_seq*args.seq_length
        for (i, param) in enumerate(params):
            x[i*n_files:(i+1)*n_files] = self._load_double_cyl_subset(args, n_seq, param)
            parameters[i*n_seq:(i+1)*n_seq] = param

        # Divide into sequences
        self.x = x.reshape(-1, args.seq_length, args.n_x, args.n_y, 4)
        self.x = self.x[:int(np.floor(len(self.x)/args.batch_size)*args.batch_size)]
        self.x = self.x.reshape(-1, args.batch_size, args.seq_length, args.n_x, args.n_y, 4)

        self.params = parameters[:int(np.floor(len(parameters)/args.batch_size)*args.batch_size)]
        self.params = self.params.reshape(-1, args.batch_size, args.param_dim)

    # Function to load subset of data at specified Reynolds number
    def _load_3d_data_subset(self, args, n_sequences, re):
        # Define numbers of files that need to be loaded for training    
        n_files = n_sequences*args.seq_length
        max_gap = (args.max_num - args.stagger*args.seq_length - args.min_num)/n_sequences
        start_idxs = np.linspace(args.min_num, max_gap*n_sequences + args.min_num, n_sequences)
        file_nums = np.array([])
        for i in range(n_sequences):
            file_nums = np.concatenate([file_nums, np.linspace(start_idxs[i], start_idxs[i] + args.seq_length*args.stagger, args.seq_length)])

        # Define progress bar
        print('loading data for Reynolds number', re, '...')
        bar = progressbar.ProgressBar(maxval=n_files).start()

        # Load data
        x = np.zeros((n_files, args.n_x, args.n_y, args.n_z, 5), dtype=np.float32)
        for i in range(n_files):
            x[i] = np.load(args.data_dir + 're' + str(re) + '_data/data-' + str(int(file_nums[i])) + '.npy')
            bar.update(i)
        bar.finish()

        return x

    def _load_3d_data(self, args):
        # Define flow types to load from and number of sequences from each
        res = [175, 200, 250, 300]
        n_seq = args.n_sequences//(len(res) + 1)

        # Define numbers of files that need to be loaded for training    
        n_files = args.n_sequences*args.seq_length

        # Load data
        x = np.zeros((n_files, args.n_x, args.n_y, args.n_z, 5), dtype=np.float32)
        params = np.zeros((args.n_sequences, args.param_dim))
        total_files = 0
        total_seq = 0
        for (i, re) in enumerate(res):
            seq_mult = 1 + 1*(re >= 300)
            n_files_subset = n_seq*args.seq_length*seq_mult
            x[total_files:total_files+n_files_subset] = self._load_3d_data_subset(args, n_seq*seq_mult, re)
            params[total_seq:total_seq + n_seq*seq_mult] = re
            total_files += n_files_subset
            total_seq += n_seq*seq_mult

        # Divide into sequences
        self.x = x.reshape(-1, args.seq_length, args.n_x, args.n_y, args.n_z, 5)
        self.x = self.x[:int(np.floor(len(self.x)/args.batch_size)*args.batch_size)]
        self.x = self.x.reshape(-1, args.batch_size, args.seq_length, args.n_x, args.n_y, args.n_z, 5)

        self.params = params[:int(np.floor(len(params)/args.batch_size)*args.batch_size)]
        self.params = self.params.reshape(-1, args.batch_size, args.param_dim)


    def _create_inputs_targets(self, args):
        # Create batch_dict and permuatation
        self.batch_dict = {}

        # Print tensor shapes
        print('inputs: ', self.x.shape)
        print('params: ', self.params.shape)
        if args.data_3d:
            self.batch_dict["inputs"] = np.zeros((args.batch_size, args.seq_length, args.n_x, args.n_y, args.n_z, 5))
        else:
            self.batch_dict["inputs"] = np.zeros((args.batch_size, args.seq_length, args.n_x, args.n_y, 4))
        self.batch_dict["params"] = np.zeros((args.batch_size, args.param_dim))

        # Shuffle data
        print('shuffling...')
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.params = self.params[p]

    # Separate data into train/validation sets
    def _create_split(self, args):
        # compute number of batches
        self.n_batches = len(self.x)
        self.n_batches_val = int(math.floor(args.val_frac * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val

        print('num training batches: ', self.n_batches_train)
        print('num validation batches: ', self.n_batches_val)

        self.reset_batchptr_train()
        self.reset_batchptr_val()

    # Shift and scale data to be zero-mean, unit variance
    def _shift_scale(self, args):
        # Find means and std if not initialized to anything
        if np.sum(self.scale_x) == 0.0:
            if args.data_3d:
                self.shift_x = np.mean(self.x[:self.n_batches_train], axis=(0, 1, 2, 3, 4, 5))
                self.scale_x = np.std(self.x[:self.n_batches_train], axis=(0, 1, 2, 3, 4, 5))
            else:
                self.shift_x = np.mean(self.x[:self.n_batches_train], axis=(0, 1, 2, 3, 4, 5, 6))
                self.scale_x = np.std(self.x[:self.n_batches_train], axis=(0, 1, 2, 3, 4, 5, 6))
            self.shift_params = np.mean(self.params[:self.n_batches_train], axis=(0, 1))
            self.scale_params = np.std(self.params[:self.n_batches_train], axis=(0, 1))

    # Sample a new batch of data
    def next_batch_train(self):
        # Extract next batch
        batch_index = self.batch_permuation_train[self.batchptr_train]
        self.batch_dict["states"] = (self.x[batch_index] - self.shift_x)/self.scale_x
        self.batch_dict["params"] = (self.params[batch_index] - self.shift_params)/self.scale_params

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    # Return to first batch in train set
    def reset_batchptr_train(self):
        self.batch_permuation_train = np.random.permutation(self.n_batches_train)
        self.batchptr_train = 0

    # Return next batch of data in validation set
    def next_batch_val(self):
        # Extract next validation batch
        batch_index = self.batchptr_val + self.n_batches_train-1
        self.batch_dict["states"] = (self.x[batch_index] - self.shift_x)/self.scale_x
        self.batch_dict["params"] = (self.params[batch_index] - self.shift_params)/self.scale_params

        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    # Return to first batch in validation set
    def reset_batchptr_val(self):
        self.batchptr_val = 0

