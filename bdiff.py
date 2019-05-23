import tensorflow as tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from rietools import *
import scipy


# create the synthetic tree dataset here iteratively
def generate_tree_dataset(n_train,n_test,scale,treedepth,b,c,d,offset,strict_norms,center,stricter_norms=False):
    N = n_train + n_test
    
    nr_nodes = b**(treedepth+1)

    #X = scale*np.random.randn(N,nr_nodes,c,d)
    X = np.zeros([N,nr_nodes,c,d])
    print("Generating trees...")
    for i in range(N):
        for j in range(b**treedepth-1):
            # true tree
            ancestor = X[i,j,0,:]
            for k in range(b):
                if strict_norms:
                    X[i,2*j+k+1,0,:] = np.random.randn(d)*offset + ancestor 
                    while(np.linalg.norm(X[i,2*j+k+1,0,:]) <= np.linalg.norm(ancestor)):
                        X[i,2*j+k+1,0,:] = np.random.randn(d)*offset + ancestor 
                elif stricter_norms:
                    X[i,2*j+k+1,0,:] = np.random.randn(d)*offset + ancestor 
                    while(np.linalg.norm(X[i,2*j+k+1,0,:]) <= np.linalg.norm(ancestor) and np.abs(np.dot(X[i,2*j+k+1,0,:],ancestor))>=3): 
                        X[i,2*j+k+1,0,:] = np.random.randn(d)*offset + ancestor 
                    
                    #scale = np.linalg.norm(X[i,2*j+k+1,0,:])/(1.5*np.linalg.norm(ancestor))
                    #X[i,2*j+k+1,0,:] /= scale 
                else:
                    X[i,2*j+k+1,0,:] = np.random.randn(d)*offset + ancestor 

                # additional noise of var 1/c
                for l in range(1,c):
                    X[i,2*j+k+1,l,:] = np.sqrt(1/c)*np.random.randn(d) + X[i,2*j+k+1,0,:]

    print(X[0,0,0,:])
    print("Done")

    # flatten the data            
    X_train = X[:n_train,:]
    X_test = X[-n_test:,:]
    X_train = np.reshape(X_train,[-1,d])
    X_test = np.reshape(X_test,[-1,d])

    # zero mean, unit variance
    if center:
        mu = np.mean(np.reshape(X,[-1,d]),0)
        stdev = np.std(np.reshape(X,[-1,d]),0)
        X_train = (X_train-mu)/stdev
        X_test = (X_test-mu)/stdev

    #X_train[0,:] = X_train[0,:]*0
    #X_test[0,:] = X_test[0,:]*0

    X_train_flat = X_train
    X_train = np.reshape(X_train,[n_train,-1,c,d])
    X_test = np.reshape(X_test,[n_test,-1,c,d])
    
    return X_train,X_train_flat,X_test

def generate_binary_tree(n_train,n_test,scale,treedepth,b,c,offset,strict_norms,center):
    N = n_train + n_test
    nr_nodes = b**(treedepth+1)
    d = treedepth

    #X = np.random.randn(N,nr_nodes,c,d)
    X = np.zeros([N,nr_nodes,c,d])
    X[end] = 1
    for i in range(N):
        for j in range(b**treedepth-1):
            # true tree
            ancestor = X[i,j,0,:]
            for k in range(b):
                # perturb one bit not shared with ancestor or neighbour
                bit_to_be_flipped = np.rand.randint(1,d-np.sum(ancestor)+1)
                X[i,2*j+k+1,0,:] = ancestor
                X[i,2*j+k+1,0,bit_to_be_flipped] = 1

                # additional noise: perturb extra bits with uniform prob
                for l in range(1,c):
                    X[i,2*j+k+1,l,:] = np.random.randint(2,d) + X[i,2*j+k+1,0,:]
    
    # flatten the data            
    X_train = X[:n_train,:]
    X_test = X[-n_test:,:]
    X_train_flat = np.reshape(X_train,[-1,d])
    X_test_flat = np.reshape(X_test,[-1,d])

    return X_train,X_train_flat,X_test

def draw_latent_tree_tsne(opts,X,treedepth,b,c,d,subset_len,plot_means=False):
    input_dim = d
    X_tree = np.reshape(X,[-1,b**(treedepth+1),c,d])
    X_subset = X_tree[0:subset_len,:,:,:]
    plt.figure(figsize=(10, 10)) 
    
    # plot tree
    z_signal = np.reshape(X_subset[:,:,0,:],[-1,input_dim])
    # mean over vertex noise points
    z_signal_means = np.mean(X_subset,2)
    z_signal = np.reshape(z_signal,[X_subset.shape[0],b**(treedepth+1),-1])

    z_noise = X_subset[:,:,1:,:]    
    color = ['r','g','y','m','b','c','k']

    for i in range(subset_len):
        for j in range(b**treedepth-1):
            clr = np.int(np.floor(np.log2(j+1)))
            parent = z_signal[i,j,:]

            plt.scatter(parent[0], parent[1], c=color[clr],marker='+',s=0.5)

            left = z_signal[i,2*j+1,:]
            right = z_signal[i,2*j+2,:]

            plt.scatter(left[0], left[1], c=color[clr+1],marker='+')
            plt.scatter(right[0], right[1], c=color[clr+1],marker='+')

            plt.scatter(z_noise[i,j,:,0],z_noise[i,j,:,1],color=color[clr],marker='.')
            plt.scatter(z_noise[i,2*j+1,:,0],z_noise[i,2*j+1,:,1],color=color[clr+1],marker='.')
            plt.scatter(z_noise[i,2*j+2,:,0],z_noise[i,2*j+2,:,1],color=color[clr+1],marker='.')

            #draw lines of tree
            x = [parent[0],left[0]]
            y = [parent[1],left[1]]
            plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))
            x = [parent[0],right[0]]
            y = [parent[1],right[1]]
            plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))
            
            #draw lines of mean tree
            if plot_means:
                parent_mean = z_signal_means[i,j,:]
                left_mean = z_signal_means[i,2*j+1,:]
                right_mean = z_signal_means[i,2*j+1,:]
                plt.scatter(parent_mean[0], parent_mean[1], c=color[clr],marker='*')
                plt.scatter(left_mean[0], left_mean[1], c=color[clr+1],marker='*')
                plt.scatter(right_mean[0], right_mean[1], c=color[clr+1],marker='*')

                x = [parent_mean[0],left_mean[0]]
                y = [parent_mean[1],left_mean[1]]
                plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))
                x = [parent_mean[0],right_mean[0]]
                y = [parent_mean[1],right_mean[1]]
                plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))

    #plt.scatter(z_noise[:,0],z_noise[:,1],color='k',marker='+')
    #plt.colorbar()
    ax = plt.gca()
    circle = plt.Circle((0, 0), 1.0, color='k', fill=False)
    ax.add_artist(circle)
    plt.axis('equal')
    plt.grid()
    plt.show()

def draw_latent_tree(opts,X_test,model,treedepth,b,c,d,subset_len,plot_means=False):
    """
    
        Transforms a test tree in input space using the trained model and draws
        a tree in latent space for visualization purposes
        
        X_test: test data (R^{nxd}: flattened tree)
        model: trained encoder model
        
    
    """
    input_dim = d
    X_test_subset = X_test[0:subset_len,:,:,:]
    plt.figure(figsize=(10, 10)) 

    
    nr_prior_samples = 300
    if opts['use_expmap']:
        z_prior_sample = np_exp_map(opts['prior_var']*np.random.randn(nr_prior_samples,2))
    else:
        z_prior_sample = np_sample_gauss_pd([nr_prior_samples,2],radius=1.0,sigma=opts['prior_var'])

    #plt.scatter(z_prior_sample[:, 0], z_prior_sample[:, 1], c='g')

    # plot tree
    signal = np.reshape(X_test[:,:,0,:],[-1,input_dim])
    z_signal = model.transformer_mu(signal) # posterior means (signal)
    # mean over vertex noise points
    z_signal_means = np.mean(X_test,2)
    #z_signal_means = np.mean(z_signal_means,1)

    # use subset or means
    # z_signal = z_signal[0:subset_len,:,:] 
    z_signal = np.reshape(z_signal,[X_test.shape[0],b**(treedepth+1),-1])
    #plt.scatter(z_signal[:, 0], z_signal[:, 1], c='r')

    noise = np.reshape(X_test_subset[:,:,1:,:],[-1,input_dim])
    z_noise = model.transformer_mu(noise)
    z_noise = np.reshape(z_noise,[subset_len,b**(treedepth+1),c-1,-1])   

    color = ['r','g','y','m','b','c','k']

    for i in range(subset_len):
        for j in range(b**treedepth-1):
            clr = np.int(np.floor(np.log2(j+1)))
            parent = z_signal[i,j,:]

            plt.scatter(parent[0], parent[1], c=color[clr],marker='+',s=0.5)

            left = z_signal[i,2*j+1,:]
            right = z_signal[i,2*j+2,:]

            plt.scatter(left[0], left[1], c=color[clr+1],marker='+')
            plt.scatter(right[0], right[1], c=color[clr+1],marker='+')

            plt.scatter(z_noise[i,j,:,0],z_noise[i,j,:,1],color=color[clr],marker='.')
            plt.scatter(z_noise[i,2*j+1,:,0],z_noise[i,2*j+1,:,1],color=color[clr+1],marker='.')
            plt.scatter(z_noise[i,2*j+2,:,0],z_noise[i,2*j+2,:,1],color=color[clr+1],marker='.')

            #draw lines of tree
            x = [parent[0],left[0]]
            y = [parent[1],left[1]]
            #plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))
            x = [parent[0],right[0]]
            y = [parent[1],right[1]]
            #plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))
                        
            #draw lines of mean tree
            if plot_means:
                parent_mean = z_signal_means[i,j,:]
                left_mean = z_signal_means[i,2*j+1,:]
                right_mean = z_signal_means[i,2*j+1,:]
                plt.scatter(parent_mean[0], parent_mean[1], c=color[clr],marker='*')
                plt.scatter(left_mean[0], left_mean[1], c=color[clr+1],marker='*')
                plt.scatter(right_mean[0], right_mean[1], c=color[clr+1],marker='*')

                x = [parent_mean[0],left_mean[0]]
                y = [parent_mean[1],left_mean[1]]
                plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))
                x = [parent_mean[0],right_mean[0]]
                y = [parent_mean[1],right_mean[1]]
                plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))

    #plt.scatter(z_noise[:,0],z_noise[:,1],color='k',marker='+')
    #plt.colorbar()
    ax = plt.gca()
    circle = plt.Circle((0, 0), 1.0, color='k', fill=False)
    ax.add_artist(circle)
    plt.axis('equal')
    plt.grid()
    plt.show()
    
def draw_means(opts,X_test,model,treedepth,b,c,d,subset_len):

    input_dim = d
    X_test_subset = X_test[0:subset_len,:,:,:]
    plt.figure(figsize=(10, 10)) 
    ax = plt.gca()
    circle = plt.Circle((0, 0), 1.0, color='k', fill=False)
    ax.add_artist(circle)
    nr_prior_samples = 300
    if opts['use_expmap']:
        z_prior_sample = np_exp_map(opts['prior_var']*np.random.randn(nr_prior_samples,2))
    else:
        z_prior_sample = np_sample_gauss_pd([nr_prior_samples,2],radius=1.0,sigma=opts['prior_var'])

    #plt.scatter(z_prior_sample[:, 0], z_prior_sample[:, 1], c='g')

    # plot tree
    signal = np.reshape(X_test[:,:,0,:],[-1,input_dim])
    z_signal = model.transformer_mu(signal)
    # mean over vertex noise points
    z_signal_means = np.mean(X_test,2)
    #z_signal_means = np.mean(z_signal_means,1)

    # use subset or means
    # z_signal = z_signal[0:subset_len,:,:] 
    z_signal = np.reshape(z_signal,[X_test.shape[0],b**(treedepth+1),-1])
    #plt.scatter(z_signal[:, 0], z_signal[:, 1], c='r')

    noise = np.reshape(X_test_subset[:,:,1:,:],[-1,input_dim])
    z_noise = model.transformer_mu(noise)
    z_noise = np.reshape(z_noise,[subset_len,b**(treedepth+1),c-1,-1])   

    color = ['r','g','y','m','b','c','k']

    for i in range(subset_len):
        for j in range(b**treedepth-1):
            clr = np.int(np.floor(np.log2(j+1)))
            parent = z_signal[i,j,:]

            plt.scatter(parent[0], parent[1], c=color[clr],marker='+',s=0.5)

            left = z_signal[i,2*j+1,:]
            right = z_signal[i,2*j+2,:]

            plt.scatter(left[0], left[1], c=color[clr+1],marker='+')
            plt.scatter(right[0], right[1], c=color[clr+1],marker='+')

            plt.scatter(z_noise[i,j,:,0],z_noise[i,j,:,1],color=color[clr],marker='.')
            plt.scatter(z_noise[i,2*j+1,:,0],z_noise[i,2*j+1,:,1],color=color[clr+1],marker='.')
            plt.scatter(z_noise[i,2*j+2,:,0],z_noise[i,2*j+2,:,1],color=color[clr+1],marker='.')

            #draw lines of tree
            x = [parent[0],left[0]]
            y = [parent[1],left[1]]
            plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))
            x = [parent[0],right[0]]
            y = [parent[1],right[1]]
            plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))
                        
            #draw lines of mean tree

            parent_mean = np.mean(z_signal[:,j,:],0)
            left_mean = np.mean(z_signal[:,2*j+1,:],0)
            right_mean = np.mean(z_signal[:,2*j+1,:],0)
            #plt.scatter(parent_mean[0], parent_mean[1], c=color[clr],marker='*')
            #plt.scatter(left_mean[0], left_mean[1], c=color[clr+1],marker='*')
            #plt.scatter(right_mean[0], right_mean[1], c=color[clr+1],marker='*')

                #x = [parent_mean[0],left_mean[0]]
                #y = [parent_mean[1],left_mean[1]]
                #plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))
                #x = [parent_mean[0],right_mean[0]]
                #y = [parent_mean[1],right_mean[1]]
                #plt.plot(x,y, c=color[clr], linewidth=5/(clr+1))

    #plt.scatter(z_noise[:,0],z_noise[:,1],color='k',marker='+')
    #plt.colorbar()
    ax = plt.gca()
    circle = plt.Circle((0, 0), 1.0, color='k', fill=False)
    ax.add_artist(circle)
    plt.axis('equal')
    plt.grid()
    plt.show()

def compute_avg_distortion_tree(data_u, data_v, b=2,treedepth=6,c=10):        
    dist = []
    for i in range(subset_len):
        for j in range(b**treedepth-1):
            parent_u = data_u[i,j,:]
            left_u = data_u[i,2*j+1,:]
            right_u = data_u[i,2*j+2,:]
            parent_v = data_v[i,j,:]
            left_v = data_v[i,2*j+1,:]
            right_v = data_v[i,2*j+2,:]
            d_v_left = dist_pd(parent_u,left_u)
            d_u_left = tf.norm(parent_u-left_u)
            dist.append(tf.abs(d_v_left - d_u_left)/d_u_left)
            dist.append(tf.abs(d_v_right - d_u_right)/d_u_right)
            # add noise?
            
    dist = tf.stack(dist)
    return tf.reduce_mean(dist)
            
def compute_avg_distortion_all(data_u, data_v):        
    # all unique pairs
    d_u = [tf.norm(data_u[p1] - data_u[p2]) for p1 in range(len(data_u)) for p2 in range(p1+1,len(data_u))]
    d_v = [dist_pd(data_v[p1], data_v[p2]) for p1 in range(len(data_v)) for p2 in range(p1+1,len(data_v))]
    return tf.abs(d_v-d_u)/d_u

def np_compute_avg_distortion_tree(data_u, data_v, b=2,treedepth=6,c=10):        
    dist = []
    for i in range(len(data_u)):
        for j in range(b**treedepth-1):
            parent_u = np.expand_dims(data_u[i,j,0,:],0)
            left_u = np.expand_dims(data_u[i,2*j+1,0,:],0)
            right_u = np.expand_dims(data_u[i,2*j+2,0,:],0)
            parent_v = np.expand_dims(data_v[i,j,0,:],0)
            left_v = np.expand_dims(data_v[i,2*j+1,0,:],0)
            right_v = np.expand_dims(data_v[i,2*j+2,0,:],0)
            d_v_left = np_dist_pd(parent_v,left_v)
            d_u_left = np.linalg.norm(parent_u-left_u)
            d_v_right = np_dist_pd(parent_v,right_v)
            d_u_right = np.linalg.norm(parent_u-right_u)
            dist.append(np.abs(d_v_left - d_u_left)/d_u_left)
            dist.append(np.abs(d_v_right - d_u_right)/d_u_right)
            # add noise?
    dist = np.array(dist)
    return np.mean(dist)

def np_compute_avg_distortion_tree_eucl(data_u, data_v, b=2,treedepth=6,c=10):        
    dist = []
    for i in range(len(data_u)):
        for j in range(b**treedepth-1):
            parent_u = np.expand_dims(data_u[i,j,0,:],0)
            left_u = np.expand_dims(data_u[i,2*j+1,0,:],0)
            right_u = np.expand_dims(data_u[i,2*j+2,0,:],0)
            parent_v = np.expand_dims(data_v[i,j,0,:],0)
            left_v = np.expand_dims(data_v[i,2*j+1,0,:],0)
            right_v = np.expand_dims(data_v[i,2*j+2,0,:],0)
            d_v_left = np.linalg.norm(parent_v-left_v)
            d_u_left = np.linalg.norm(parent_u-left_u)
            d_v_right = np.linalg.norm(parent_v-right_v)
            d_u_right = np.linalg.norm(parent_u-right_u)
            dist.append(np.abs(d_v_left - d_u_left)/d_u_left)
            dist.append(np.abs(d_v_right - d_u_right)/d_u_right)
            # add noise?
    dist = np.array(dist)
    return np.mean(dist)

def np_compute_avg_distortion_all(data_u, data_v):        
    # all unique pairs
    d_u = [np.linalg.norm(data_u[p1] - data_u[p2]) for p1 in range(len(data_u)) for p2 in range(p1+1,len(data_u))]
    d_v = [np_dist_pd(np.expand_dims(data_v[p1],0), np.expand_dims(data_v[p2],0)) for p1 in range(len(data_v)) for p2 in range(p1+1,len(data_v))]
    d_u = np.array(d_u)
    d_v = np.array(d_v)
    return np.mean(np.abs(d_v-d_u)/d_u)

def np_compute_avg_distortion_all_eucl(data_u, data_v):        
    # all unique pairs
    d_u = scipy.spatial.distance.cdist(data_u,data_u)
    d_v = scipy.spatial.distance.cdist(data_v,data_v)
    
    d_u = d_u + np.eye(d_u.shape[0])
    d_v = d_v + np.eye(d_v.shape[0])    
    
    print(d_u)

    print(d_u.shape)
    
    #print("closed to zero sum", np.sum(np.array(d_u<1e-5,dtype=np.int)))
    return np.mean(np.abs(d_v-d_u)/d_u)


