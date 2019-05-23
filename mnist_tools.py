import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from rietools import *
import scipy

def recon(model):
    #reconstruction: left recon, right ground truth
    #z = np.random.normal(size=[model.batch_size,model.dim_z])
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    batch = mnist.test.next_batch(100)
    batch = batch[0]
    if opts['binarize']:
        batch = np.random.binomial(1,batch)
    x_reconstructed, z = model.reconstruction(batch)
    #x_generated = model.generate(z)
    n = 10
    h = 28
    w = 28
    I_reconstructed = np.empty((h*n,2*w*n))

    for i in range(n):
        for j in range(n):
            x = np.concatenate((x_reconstructed[i*n+j,:].reshape(h,w),batch[i*n+j,:].reshape(h,w)),axis=1)
            I_reconstructed[i*h:(i+1)*h,j*2*w:(j+1)*2*w] = x
    plt.figure(figsize = (10,20))
    plt.imshow(I_reconstructed,cmap='gray')
    #plt.savefig('mnist_recon.png',dpi=300)

def gen_circle(model):
    #Test the trained model: transformation
    n = 50
    #x = np.concatenate([np.logspace(-1.0, 0, n/2)*-1, np.logspace(-1.0, 0, n/2)])
    #y = np.concatenate([np.logspace(-1.0, 0, n/2)*-1, np.logspace(-1.0, 0, n/2)])
    x = np.linspace(-1.0,1.0,n)
    y = np.linspace(-1.0,1.0,n)

    #y = np.logspace(-1.0, 1.0, n)

    #r = np.logspace(0, 1, n)
    #y = np.linspace(0,2*pi,n)
    h = 28
    w = 28

    I_latent = np.empty((h*n, w*n))
    for i, yi in enumerate(x):
        for j, xi in enumerate(y):
            z = np.array([[xi, yi]]*model_2d.batch_size)
            exp_z = np_exp_map(z)
            x_hat = model.generate(z)
            I_latent[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = x_hat[20].reshape(28, 28)
        
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(I_latent, cmap="gray_r")
    plt.savefig("fig/mnist_disk_5ep.png")
    plt.show()
    
def gen_grid(model):
    ### Test the trained model: generation
    # Sample noise vectors from N(0, 1)
    #z = np.random.normal(size=[model.batch_size, model.dim_z])
    #prior_sample = exp_map(sample_gauss_h(tf.shape(mu)))
    #prior_sample = sample_gauss_h(tf.shape(mu))
    #z = np_exp_map(np.random.randn(model.batch_size, model.dim_z).astype(np.float32),c=1.0)
    #z = np_project_hyp_vec(np_sample_gauss_hp(shape=[model.batch_siz, model.dim_z],oversample=20,radius=3.0,sigma=1.0),1.0)
    #z = np_sample_gauss_pd(shape=[model.batch_size, model.dim_z],oversample=20,radius=0.85,sigma=1.0)
    z = np_project_hyp_vec(np_sample_gauss_pd(shape=[model.batch_size, model.dim_z],oversample=20,radius=1.0,sigma=1.0),1.0)
    #z = np.reshape(np_sample_uniform_h_r(n_samples=model.batch_size*model.dim_z,radius=1.0,sigma=1.0),[2*model.batch_size,model.dim_z])

    print(z.shape)

    x_generated = model.generate(z)
    h = 28
    w = 28
    # 10x10 grid
    n = np.sqrt(model.batch_size).astype(np.int32)
    I_generated = np.empty((h*n, w*n))
    for i in range(n):
        for j in range(n):
            I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = x_generated[i*n+j, :].reshape(28, 28)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(I_generated, cmap='gray_r')
    #plt.savefig('mnist_5_h_100ep.png')
    
def gen_geodesic_interpolation(model):
    # gen line in euclidean space
    