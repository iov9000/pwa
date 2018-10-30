import tensorflow as tf 
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import np_sample_gauss_pd, np_project_hyp_vec
from hvae import HVAE, VAE

from tensorflow.examples.tutorials.mnist import input_data
#from datasets.sprite 
from tensor2tensor import problems
from tensor2tensor.data_generators import celeba, cifar
data_dir = os.path.expanduser("/tmp/t2t/data")
tmp_dir = os.path.expanduser("/tmp/t2t/tmp")
train_dir = os.path.expanduser("/tmp/t2t/train")
checkpoint_dir = os.path.expanduser("/tmp/t2t/checkpoints")
tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)
tf.gfile.MakeDirs(train_dir)
tf.gfile.MakeDirs(checkpoint_dir)
gs_data_dir = "gs://tensor2tensor-data"
gs_ckpt_dir = "gs://tensor2tensor-checkpoints/"
Modes = tf.estimator.ModeKeys

def trainer(sess, opts, model,epoch,nr_samples):
    #model
    batch_size = opts['batch_size']
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    fmnist = input_data.read_data_sets('Fashion_MNIST_data',one_hot=True)
    #training loop
    for ep in range(epoch):
        for iter in range(nr_samples // batch_size):
            #obtain a mini-batch
            #tuple:images,labels
            if opts['dataset'] == 'mnist':
                #if opts['dataset'] == 'fmnist':
                #    mnist = fmnist

                batch = mnist.train.next_batch(batch_size)[0]
                if opts['binarize']:
                    batch = np.random.binomial(1,batch)
                loss, mloss, rloss, kloss = model.run_single_step(batch,training=False)

            elif opts['dataset'] == 'celeba':
                loss, mloss, rloss, kloss = model.run_single_step(np.zeros(shape=(batch_size,784),dtype=float),training=False)

            #train: execute forward and backward pass
        print('[Epoch {}] Loss: {}, {}, {}'.format(ep,rloss,mloss,kloss))
    print('Done!')
    return model

def reconstruction(model,batch):
    #reconstruction: left recon, right ground truth
    #z = np.random.normal(size=[model.batch_size,model.dim_z])
    #batch = mnist.test.next_batch(100)
    #batch = batch[0]
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

def sample_mnist(model):
    ### Test the trained model: generation
    # Sample noise vectors from N(0, 1)
    #z = np.random.normal(size=[model.batch_size, model.dim_z])
    z = np_project_hyp_vec(np_sample_gauss_pd(shape=[model.batch_size, model.dim_z],oversample=20,radius=1.0,sigma=1.0),1.0)

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
    plt.savefig('mnist_5_h.png')

def sample_celeba(model,hyp=False):
    # Test the trained model: generation
    # Sample noise vectors from N(0, 1)
    if hyp:
        z = np_project_hyp_vec(np_sample_gauss_pd(shape=[model.batch_size, model.dim_z],oversample=20,radius=1.0,sigma=1.0),1.0)
    else:
        z = np.random.normal(size=[model.batch_size, model.n_z])

    x_generated = model.generate(z)

    # 10x10 grid
    n = np.sqrt(model.batch_size).astype(np.int32)
    h = 64
    w = 64
    I_generated = np.empty((h*n, w*n,3))
    for i in range(n):
        for j in range(n):
            I_generated[i*h:(i+1)*h, j*w:(j+1)*w,:] = x_generated[i*n+j]

    plt.figure(figsize=(8, 8))
    plt.imshow(I_generated)
    plt.savefig('celeba_recon.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S2S model training')
    parser.add_argument('--hparams', type=str,
                     help='Comma separated list of "name=value" pairs.')
    parser.add_argument('--mode', type=str, default='train',
                      help='train/eval switch')
    args = parser.parse_args()
    tf.reset_default_graph()
    np.random.seed(13)
    tf.set_random_seed(13)
    opts = {}
    opts['dataset'] = 'mnist' # 'fmnist', 'celeba'
    opts['lr_adam'] = 1e-3
    opts['lr_rsgd'] = 1e-3
    opts['batch_size'] = 100
    opts['beta_mmd'] = 2.e2
    opts['beta_kld'] = 10.0
    opts['lambda_mmd'] = 0.0
    opts['varregnorm'] = 2 # L1 or L2
    opts['dim_z'] = 5
    opts['hyp_c'] = 1.0
    opts['mmd_q'] = 1
    opts['use_expmap'] = False
    opts['nw_size'] = 'small'
    opts['binarize'] = True
    opts['init_std'] = 0.01
    opts['init_bias'] = 0.0
    opts['conv_filters_dim'] = 5
    opts['num_filters'] = 1024
    opts['num_layers'] = 4

    print(opts)
    sess = tf.train.MonitoredSession()
    #nr_samples = mnist.train.num_examples
    model = HVAE(opts)
    model = trainer(sess, model, opts, epoch=100, nr_samples=60000) 
