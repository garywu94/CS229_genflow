import h5py
import tensorflow as tf
import numpy as np
import os, sys
import progressbar
import matplotlib.pyplot as plt
import time

from scipy import integrate

# Function to employ trained network in generative fashion
def generate_solutions(args, net, re=75, om=1.0):
    print('generating solutions...')
    # Begin tf session
    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join('best_' + args.save_dir, args.ckpt_name))

        params = np.zeros((args.batch_size, args.param_dim))
        #params[:] = (np.array([om, re]) - sess.run(net.shift_params))/sess.run(net.scale_params)
        params[:] = (np.array([1.69]) - sess.run(net.shift_params)) / sess.run(net.scale_params)

        # Define array to hold all solutions
        #solns = np.zeros((args.n_gen_seq*args.seq_length, args.n_x, args.n_y, 4))
        solns = np.zeros((args.n_gen_seq * args.seq_length, args.n_x, args.n_y, 6))

        for i in range(args.n_gen_seq):
            # Construct inputs to network and get out generated solutions
            feed_in = {net.generative: True, net.params: params}
            if i > 0:
                feed_in[net.decode_vals_reshape] = gen_g[:, i*args.seq_length:(i+1)*args.seq_length]
                #feed_out = net.x_pred_reshape
                feed_out = [net.x_pred_reshape, net.decode_vals]
                gen_soln = sess.run(feed_out, feed_in)
            else:
                #feed_out = [net.x_pred_reshape, net.gen_g]
                feed_out = [net.x_pred_reshape, net.gen_g, net.decode_vals]
                gen_soln, gen_g = sess.run(feed_out, feed_in)

            # Assign generated solutions to array
            solns[i*args.seq_length:(i+1)*args.seq_length] = gen_soln[0]

        # Define freestream values for to be used for cylinders
        #rho = 1.0
        #P = 1.0
        #u = 0.1*np.sqrt(1.4)
        #v = 0.0
        #freestream = np.array([rho, u, v, P])

        # Load and assign mask for points inside cylinders
        #pt_mask = np.load('double_cyl_mask.npy')
        #cyl_mask = np.tile(pt_mask, (args.n_gen_seq*args.seq_length, 1, 1, 1))
        #solns = (1 - cyl_mask)*freestream + cyl_mask*solns

        # Find range of values
        t0 = (args.n_gen_seq-1)*args.seq_length
        lims0 = [np.amin(solns[t0, :, :, 0]), np.amax(solns[t0, :, :, 0])]
        lims1 = [np.amin(solns[t0, :, :, 1]), np.amax(solns[t0, :, :, 1])]
        lims2 = [np.amin(solns[t0, :, :, 2]), np.amax(solns[t0, :, :, 2])]
        lims3 = [np.amin(solns[t0, :, :, 3]), np.amax(solns[t0, :, :, 3])]
        lims4 = [np.amin(solns[t0, :, :, 3]), np.amax(solns[t0, :, :, 4])]
        lims5 = [np.amin(solns[t0, :, :, 3]), np.amax(solns[t0, :, :, 5])]

        # Replace with nans so cylinder can be plotted in gray
        #solns[cyl_mask == 0.0] = np.nan
        cmap = plt.cm.coolwarm
        cmap.set_bad('gray', 0.2)

    # Save images to file
    print('saving images...')
    for t in range(5):
        time = (args.n_gen_seq-1)*args.seq_length + t
        plt.close()
        plt.clf
        fig = plt.imshow(solns[time, :, :, 0], cmap=cmap, vmin=lims0[0], vmax=lims0[1])
        fig.axes.set_axis_off()
        plt.tight_layout()
        plt.savefig('gen_images/density_' + str(t) + '.png')  

        plt.close()
        plt.clf
        fig = plt.imshow(solns[time, :, :, 1], cmap=cmap, vmin=lims1[0], vmax=lims1[1])
        fig.axes.set_axis_off()
        plt.tight_layout()
        plt.savefig('gen_images/xvel_' + str(t) + '.png') 

        plt.close()
        plt.clf
        fig = plt.imshow(solns[time, :, :, 2], cmap=cmap, vmin=lims2[0], vmax=lims2[1])
        fig.axes.set_axis_off()
        plt.tight_layout()
        plt.savefig('gen_images/yvel_' + str(t) + '.png')  

        plt.close()
        plt.clf
        fig = plt.imshow(solns[time, :, :, 5], cmap=cmap, vmin=lims5[0], vmax=lims5[1])
        fig.axes.set_axis_off()
        plt.tight_layout()
        plt.savefig('gen_images/press_' + str(t) + '.png')
    print('done.')  

# Function to calculate lift and drag on top and bottom cylinder
def calc_lift_drag(soln, idxs_top, idxs_bottom, thetas, mask):
    # Find pressure field and remove unwanted points
    pressure_field = soln[:, :, 3]
    pressure_field_reshape = pressure_field.reshape(-1)
    pressure_masked = pressure_field_reshape[mask]

    # Find pressure distributions on upper and lower cylinders    
    pressure_top = pressure_masked[idxs_top]
    pressure_bottom = pressure_masked[idxs_bottom]

    # Define sine and cosine values associated with each theta
    sines = np.sin(thetas)
    cosines = np.cos(thetas)

    # Find pressure values on top on bottom in x- and y-directions
    drag_top = integrate.trapz(-pressure_top*cosines, thetas)
    lift_top = integrate.trapz(-pressure_top*sines, thetas)
    drag_bottom = integrate.trapz(-pressure_bottom*cosines, thetas)
    lift_bottom = integrate.trapz(-pressure_bottom*sines, thetas)
    
    return drag_top, lift_top, drag_bottom, lift_bottom

# Find time variation in solutions at various flow conditions
def find_time_variation(args, net):
    print('generating solutions...')
    # Begin tf session
    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join('best_' + args.save_dir, args.ckpt_name))

        # Load arrays from file
        mask_reshape = np.load('mask_reshape.npy')
        idxs_top = np.load('pt_idxs_top.npy')
        idxs_bottom = np.load('pt_idxs_bottom.npy')
        thetas = np.linspace(0, 2*np.pi, 200)

        # Define set of reynolds numbers over which to simulate
        res = [75.0, 85.0, 100.0, 125.0, 150.0, 175.0, 200.0]
        num_re = len(res)

        # Define list of omegas
        num_omegas = 11
        omegas = np.linspace(0.0, 2.5, num_omegas)
        xmom = np.zeros((num_re, num_omegas, args.n_gen_seq*args.seq_length, 4))

        # Lift and drag arrays
        lift_top_array = np.zeros((num_re, num_omegas, args.n_gen_seq*args.seq_length))
        lift_bottom_array = np.zeros((num_re, num_omegas, args.n_gen_seq*args.seq_length))
        drag_top_array = np.zeros((num_re, num_omegas, args.n_gen_seq*args.seq_length))
        drag_bottom_array = np.zeros((num_re, num_omegas, args.n_gen_seq*args.seq_length))

        print('extracting lift/drag data...')
        # Loop over each reynolds number
        for (i, re) in enumerate(res):

            # Loop over omegas
            for (j, om) in enumerate(omegas):
                # Find initial time
                t0 = time.time()

                # Set omega value
                params = np.zeros((args.batch_size, args.param_dim))
                params[:] = (np.array([om, re]) - sess.run(net.shift_params))/sess.run(net.scale_params)

                # Initialize array to hold solutions
                soln_array = np.zeros((args.n_gen_seq*args.seq_length, args.n_x, args.n_y, 4))

                # Construct inputs to network and get out generated solutions
                for k in range(args.n_gen_seq):
                    feed_in = {net.params: params, net.generative: True}
                    feed_in[net.generative] = True
                    if k > 0:
                        feed_in[net.decode_vals_reshape] = gen_g[:, k*args.seq_length:(k+1)*args.seq_length]
                        feed_out = net.x_pred_reshape
                        gen_soln = sess.run(feed_out, feed_in)
                    else:
                        # # Load data
                        # x = np.zeros((args.seq_length, args.n_x, args.n_y, 4), dtype=np.float32)
                        # for t in range(args.seq_length):
                        #     f = h5py.File(args.data_dir + 're' + str(int(re))  + '/om' + str(int(4*om)+1) + '/sol_data/' + 'sol_data_'+str(200+t).zfill(4)+'.h5', 'r')
                        #     sol_data = np.array(f['sol_data'])

                        #     # Find pressure values
                        #     rho = sol_data[:, :, 0]
                        #     u = sol_data[:, :, 1]/rho
                        #     v = sol_data[:, :, 2]/rho
                        #     e = sol_data[:, :, 3]
                        #     P = 0.4*rho*(e - 0.5*(u**2 + v**2))
                            
                        #     # Replace x- and y-momentum with x- and y- velocity, energy with pressure
                        #     sol_data[:, :, 1] = u
                        #     sol_data[:, :, 2] = v
                        #     sol_data[:, :, 3] = P

                        #     # Concatenate and store data
                        #     x[t] = sol_data
                        # feed_in[net.x] = (x - sess.run(net.shift))/sess.run(net.scale)
                        feed_out = [net.x_pred_reshape, net.gen_g]
                        gen_soln, gen_g = sess.run(feed_out, feed_in)

                    soln_array[k*args.seq_length:(k+1)*args.seq_length] = gen_soln[0]

                # Find simulation time
                t1 = time.time()
                total_time = t1 - t0
                print('simulation time at Reynolds num', re, 'and omega', om, 'is', total_time)

                # Find pressure values in wake
                xmom[i, j] = soln_array[:, 44, 150]

                # Find lift and drag values
                for t in range(args.n_gen_seq*args.seq_length):
                    drag_top, lift_top, drag_bottom, lift_bottom = calc_lift_drag(soln_array[t], idxs_top, idxs_bottom, thetas, mask_reshape)
                    lift_top_array[i, j, t] = lift_top
                    lift_bottom_array[i, j, t] = lift_bottom
                    drag_top_array[i, j, t] = drag_top
                    drag_bottom_array[i, j, t] = drag_bottom

        # Define set of reynolds numbers over which to simulate
        num_re = 126
        res = np.linspace(75.0, 200.0, num_re)

        # Define list of omegas
        num_omegas = 101
        omegas = np.linspace(1.1, 2.1, num_omegas)
        
        # Array to hold time differences
        diffs_array = np.zeros((num_re, num_omegas, 2))

        print('extracting time variation...')
        # Loop over each reynolds number
        for (i, re) in enumerate(res):

            # Loop over omegas
            for (j, om) in enumerate(omegas):
                # Find initial time
                t0 = time.time()

                # Set omega value
                params = np.zeros((args.batch_size, args.param_dim))
                params[:] = (np.array([om, re]) - sess.run(net.shift_params))/sess.run(net.scale_params)

                # Initialize array to hold solutions
                soln_array = np.zeros((args.n_gen_seq*args.seq_length, args.n_x, args.n_y, 4))

                # Construct inputs to network and get out generated solutions
                for k in range(args.n_gen_seq):
                    feed_in = {net.params: params, net.generative: True}
                    feed_in[net.generative] = True
                    if k > 0:
                        feed_in[net.decode_vals_reshape] = gen_g[:, k*args.seq_length:(k+1)*args.seq_length]
                        feed_out = net.x_pred_reshape
                        gen_soln = sess.run(feed_out, feed_in)
                    else:
                        feed_out = [net.x_pred_reshape, net.gen_g]
                        gen_soln, gen_g = sess.run(feed_out, feed_in)
                    soln_array[k*args.seq_length:(k+1)*args.seq_length] = gen_soln[0]

                # Find simulation time
                t1 = time.time()
                total_time = t1 - t0
                print('simulation time at Reynolds num', re, 'and omega', om, 'is', total_time)

                # Find time variation in lift values
                lift_array = np.zeros(args.n_gen_seq*args.seq_length)
                for t in range(args.n_gen_seq*args.seq_length):
                    _, lift_top, _, _ = calc_lift_drag(soln_array[t], idxs_top, idxs_bottom, thetas, mask_reshape)
                    lift_array[t] = lift_top
                diffs_array[i, j, 0] = np.std(lift_array[2*args.seq_length:])
                diffs_array[i, j, 1] = np.linalg.norm(np.diff(soln_array[2*args.seq_length:], axis=0))

        # Save results to file
        f = h5py.File('time_diffs_no_param.h5', 'w')
        f['time_diffs'] = diffs_array
        f['omegas'] = omegas
        f['xmom'] = xmom
        f['lift_top'] = lift_top_array
        f['lift_bottom'] = lift_bottom_array
        f['drag_top'] = drag_top_array
        f['drag_bottom'] = drag_bottom_array
        f.close()


# Generate solutions during training
def view_solutions(args, net, sess, om=1.0, re=75):
    params = np.zeros((args.batch_size, args.param_dim))
    params[:] = (np.array([om, re]) - sess.run(net.shift_params))/sess.run(net.scale_params)

    # Construct inputs to network and get out generated solutions
    feed_in = {net.generative: True, net.params: params}
    feed_out = net.x_pred_reshape
    solns = sess.run(feed_out, feed_in)
    solns = solns[0]

    # Define freestream values for to be used for cylinders
    #rho = 1.0
    #P = 1.0
    #u = 0.1*np.sqrt(1.4)
    #v = 0.0
    #freestream = np.array([rho, u, v, P])

    # Load and assign mask for points inside cylinders
    #pt_mask = np.load('double_cyl_mask.npy')
    #cyl_mask = np.tile(pt_mask, (args.seq_length, 1, 1, 1))
    #solns = (1 - cyl_mask)*freestream + cyl_mask*solns

    # Save images to file
    for t in range(args.seq_length):
        plt.close()
        plt.clf
        fig = plt.imshow(solns[t, :, :, 1], cmap='coolwarm')
        fig.axes.set_axis_off()
        plt.tight_layout()
        plt.savefig('train_images/u_' + str(t) + '.png')

# Function to generate sequence of images from dataset at a given Reynolds number
def visualize_dataset(args, re=125, om=0.0):
    # Define data directory
    data_dir = args.data_dir + 're' + str(int(re))  + '/om' + str(int(4*om)+1) + '/sol_data/'

    # Choose random start index
    start_idx = np.random.randint(args.min_num, args.max_num - args.seq_length*args.stagger - 1)

    # Load data and save images
    x = np.zeros((args.n_x, args.n_y, 4), dtype=np.float32)
    for i in range(15):
        print(i)
        f = h5py.File(data_dir + 'sol_data_'+str(start_idx+i*args.stagger).zfill(4)+'.h5', 'r')
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

        if i == 0:
            # Define freestream values for to be used for cylinders
            rho = 1.0
            P = 1.0
            u = 0.1*np.sqrt(1.4)
            v = 0.0

            lims0 = [np.amin(sol_data[:, :, 0]), np.amax(sol_data[:, :, 0])]
            lims1 = [np.amin(sol_data[:, :, 1]), np.amax(sol_data[:, :, 1])]
            lims2 = [np.amin(sol_data[:, :, 2]), np.amax(sol_data[:, :, 2])]
            lims3 = [np.amin(sol_data[:, :, 3]), np.amax(sol_data[:, :, 3])]

        # Find mask for points in cylinder
        pt_mask = np.load('double_cyl_mask.npy')
        pt_mask = pt_mask[0]
        sol_data[pt_mask == 0.0] = np.nan

        plt.close()
        plt.clf
        cmap = plt.cm.coolwarm
        cmap.set_bad('gray', 0.2)
        fig = plt.imshow(sol_data[:, :, 0], cmap=cmap, vmin=lims0[0], vmax=lims0[1])
        fig.axes.set_axis_off()
        plt.tight_layout()
        plt.savefig('gen_images/density_' + str(i) + '.png') 

        plt.close()
        plt.clf
        fig = plt.imshow(sol_data[:, :, 1], cmap=cmap, vmin=lims1[0], vmax=lims1[1])
        fig.axes.set_axis_off()
        plt.tight_layout()
        plt.savefig('gen_images/xvel_' + str(i) + '.png') 

        plt.close()
        plt.clf
        fig = plt.imshow(sol_data[:, :, 2], cmap=cmap, vmin=lims2[0], vmax=lims2[1])
        fig.axes.set_axis_off()
        plt.tight_layout()
        plt.savefig('gen_images/yvel_' + str(i) + '.png') 

        plt.close()
        plt.clf
        fig = plt.imshow(sol_data[:, :, 3], cmap=cmap, vmin=lims3[0], vmax=lims3[1])
        fig.axes.set_axis_off()
        plt.tight_layout()
        plt.savefig('gen_images/press_' + str(i) + '.png')   

# Function to visualize reconstructed solution
def view_solutions_3d(args, net, sess, re=300):
    # Load data
    start_idx = np.random.randint(args.min_num, args.max_num - args.seq_length*args.stagger - 1)
    x = np.zeros((args.batch_size, args.seq_length, args.n_x, args.n_y, args.n_z, 5), dtype=np.float32)
    for i in range(args.seq_length):
        x[:, i] = np.load(args.data_dir + 're' + str(int(re)) + '_data/data-' + str(start_idx+args.stagger*i) + '.npy')
    x = x.reshape(args.batch_size*args.seq_length, args.n_x, args.n_y, args.n_z, 5)
    solns = x

    # # Normalize data
    # x = (x - sess.run(net.shift))/sess.run(net.scale)
    # params = np.zeros((args.batch_size, args.param_dim))
    # params[:] = (re - sess.run(net.shift_params))/sess.run(net.scale_params)

    # # Construct inputs to network and get out generated solutions
    # feed_in = {net.x: x, net.params: params}
    # feed_out = net.x_pred_reshape
    # solns = sess.run(feed_out, feed_in)


    # Save generated solutions to file
    print('saving solutions...')
    headers = "density,xvel,yvel,zvel,pressure"
    for t in range(10,16):
        data = solns[t]
        data = data.transpose(2, 1, 0, 3).reshape(-1, 5)
        np.savetxt("3d_data/gen_data-" + str(t) + ".csv", data, fmt='%1.5f', delimiter=',', header=headers, comments='')
    print('done.') 

# Function to employ trained network in generative fashion
def generate_solutions_3d(args, net, re=350):
    print('generating solutions...')
    # Begin tf session
    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join('best_' + args.save_dir, args.ckpt_name))

        params = np.zeros((args.batch_size, args.param_dim))
        params[:] = (re - sess.run(net.shift_params))/sess.run(net.scale_params)

        # # Initialize array to hold solutions
        # solns = np.zeros((args.n_gen_seq*args.seq_length, args.n_x, args.n_y, args.n_z, 5))
        for _ in range(2):

            # Find initial time
            t0 = time.time()

            # Construct inputs to network and get out generated solutions
            for k in range(args.n_gen_seq):
                feed_in = {net.generative: True, net.params: params}
                if k > 0:
                    feed_in[net.decode_vals_reshape] = gen_g[:, k*args.seq_length:(k+1)*args.seq_length]
                    #feed_out = [net.x_pred_reshape, net.param_pred]
                    feed_out = [net.x_pred_reshape, net.param_pred, net.decode_vals]
                    #dim: net.decode_vals = [args.batch_size*args.seq_length, args.latent_dim]
                    gen_soln, pred = sess.run(feed_out, feed_in)
                    print(pred*sess.run(net.scale_params) + sess.run(net.shift_params))
                else:
                    #feed_out = [net.x_pred_reshape, net.gen_g]
                    feed_out = [net.x_pred_reshape, net.gen_g, net.decode_vals]
                    gen_soln, gen_g = sess.run(feed_out, feed_in)
                solns = gen_soln[0]

            # Find final time
            t1 = time.time()
            print('total time:', t1-t0)


    # Save generated solutions to file
    print('saving solutions...')
    headers = "density,xvel,yvel,zvel,pressure"
    for t in range(args.seq_length):
        data = solns[t]
        data = data.transpose(2, 1, 0, 3).reshape(-1, 5)
        np.savetxt("3d_data/gen_data-" + str(t) + ".csv", data, fmt='%1.5f', delimiter=',', header=headers, comments='')
    print('done.')    


# Function to employ trained network in generative fashion
def generate_time_avg(args, net, re=350):
    print('generating solutions...')
    # Begin tf session
    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join('best_' + args.save_dir, args.ckpt_name))

        # Specify params
        params = np.zeros((args.batch_size, args.param_dim))
        params[:] = (re - sess.run(net.shift_params))/sess.run(net.scale_params)

        # Define number of passes to perform
        n_passes = 10

        # Initialize array to hold time average of flow quantities
        time_avg = np.zeros((args.n_x, args.n_y, 5))

        # Define progress bar
        bar = progressbar.ProgressBar(maxval=n_passes).start()

        # Loop through passes and update averages
        for i in range(n_passes):
            # Construct inputs to network and get out generated solutions
            feed_in = {net.generative: True, net.params: params}
            feed_out = net.x_pred_reshape
            solns = sess.run(feed_out, feed_in)
            solns = solns[0]

            # Loop through time and find average across z-direction
            for t in range(args.seq_length):
                time_avg += 1/31.0*(0.5*(solns[t, :, :, 0] + solns[t, :, :, -1]) + np.sum(solns[t, :, :, 1:-1], axis=2))
            bar.update(i)
        bar.finish()

        # Finally divide the time_avg by the total number of time steps
        time_avg /= (args.seq_length*n_passes)

    # Save time averages to file
    print('saving solutions...')
    np.save('3d_data/avg_flow_re' + str(re) + '_gen.npy', time_avg)
    print('done.')               








