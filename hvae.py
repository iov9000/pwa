import tensorflow as tf 
import numpy as np 
from utils import *
from ops import *

class HVAE(object):
    def __init__(self,opts):
        self._opts = opts
        self.lr_adam = opts['lr_adam']
        self.lr_rsgd = opts['lr_rsgd']
        self.batch_size = opts['batch_size']
        self.beta_mmd = opts['beta_mmd']
        self.beta_kld = opts['beta_kld']
        self.lambda_mmd = opts['lambda_mmd']
        self.varregnorm = opts['varregnorm']
        self.dim_z = opts['dim_z']
        self.c = opts['hyp_c']
        self.q = opts['mmd_q']
        self.use_expmap= opts['use_expmap']
        
        #build a network
        self.build()
        #launch a session

        
    #build a network
    def build(self):
        #input
        self.x = tf.placeholder(name='x',dtype=tf.float32,shape=[None,784])  #shape=(batch_size,input_dim)
        self.training = tf.placeholder(name='training',dtype=tf.bool,shape=None)  #shape=(batch_size,input_dim)

        #encoder
        if 'mnist' in self._opts['dataset']:
            if self._opts['nw_size'] == 'small':
                e = enc_mnist_small(self.x)
            else:
                e = enc_mnist(self.x)
        else:
            self.celeba_batch = get_celeba_batch(self.batch_size)
            #cn_batch = contrast_norm(self.celeba_batch['inputs'])
            e = enc_conv(opts, self.celeba_batch['inputs'])
            print(e)
            e = tf.reshape(e,[self.batch_size,4*4*opts['num_filters']])
        #e = enc_conv(self.x)
        
        # project Euclidean (mu,sigma) to Poincaré disk
        #self.mu = exp_map(tf.layers.dense(f3,self.dim_z,name='enc_mu',activation=None),c=self.c)
        #self.log_sigma = exp_map(tf.layers.dense(f3,self.dim_z,name='enc_sigma',activation=tf.nn.softplus),c=self.c)
        
        # make these layers fully hyperbolic    
        self.mu = hyp_fflayer(e,self.dim_z,name='mu',activation=None)
        self.log_sigma = hyp_fflayer(e,self.dim_z,name='sigma',activation=tf.nn.softplus)
        
        #self.mu = self.mu/tf.norm(self.mu,2,axis=1,keepdims=True)
        #self.log_sigma = self.log_sigma/tf.norm(self.log_sigma,2,axis=1,keepdims=True)

        #N(z_mu,z_sigma)
        #eps = tf.random_normal(shape=tf.shape(self.log_sigma_z_sq),mean=0,stddev=1,dtype=tf.float32)
        #self.z = tf.add(self.z_mu,tf.sqrt(1e-8 + tf.exp(self.log_sigma_z_sq)) * eps,name='sample')
        self.z, prior_sample = reparam_h(self.mu, self.log_sigma, radius=0.85, c=self.c, use_expmap=self.use_expmap)
        self.mu_norm = tf.norm(self.mu,2)
        self.sigma_norm = tf.norm(self.log_sigma,2)
        self.z_norm = tf.norm(self.z,2)
        self.z_prior_norm = tf.norm(prior_sample,2)
                        
        if 'mnist' in self._opts['dataset']:
            if self._opts['nw_size'] == 'small':
                self.x_hat_s, self.x_hat = dec_mnist_small(log_map(self.z))
            else:
                self.x_hat_, self.x_hat = dec_mnist(log_map(self.z))
            
            epsilon = 1e-10
            self.bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.x_hat_s)
            self.recon_loss = tf.reduce_mean(tf.reduce_sum(self.bce, axis=-1))
            #self.recon_loss = -tf.reduce_sum(self.x*tf.log(self.x_hat+epsilon)+\
            #                            (1-self.x)*tf.log(1-self.x_hat+epsilon),axis=1)
            #self.recon_loss = tf.reduce_mean(self.recon_loss)
        else:
            self.x_hat, _  = dec_conv(opts,self.z,output_shape=[64,64,3])
            #cn_targets = contrast_norm(self.celeba_batch['targets'])
            self.recon_loss = 0.2*tf.reduce_mean(tf.nn.l2_loss(self.x_hat - self.celeba_batch['targets']))
        
        #latent loss
        self.kld_loss = kld_loss(self.mu,self.log_sigma,d=self.dim_z)        
        #self.kld_loss = -0.5*tf.reduce_sum(1+self.log_sigma-tf.square(selfmu)- tf.exp(self.log_sigma),axis=1)

        # MMD loss
        #prior_sample = tf.random_normal([self.batch_size, self.dim_z])
        #prior_sample = sample_gauss_h(tf.shape(self.mu))
        mmd_params = {'nr_mmd_samples':1, 
                  'batch_size':self.batch_size,
                  'mmd_kernel':'RBF',
                  'embedding_dim':self.dim_z,
                  'c':self.c
                 }
        prior_sample = project_hyp_vec(sample_gauss_pd(tf.shape(self.mu),radius=0.88),c=self.c)
        self.mmd_loss = mmd_penalty(mmd_params,prior_sample,self.z,q=self.q)
        self.varreg = tf.reduce_mean(tf.norm(self.log_sigma,self.varregnorm,axis=1))
    
        #total loss
        self.loss = self.recon_loss + self.beta_mmd*self.mmd_loss + self.lambda_mmd*self.varreg
        #self.loss = self.recon_loss - self.beta_kld*self.kld_loss

        #optimiser
        self.optim_adam = tf.train.AdamOptimizer(learning_rate=self.lr_adam,beta1=0.5)
        self.optim_rsgd = tf.train.GradientDescentOptimizer(learning_rate=self.lr_rsgd)
        self.full_grad_list = self.optim_rsgd.compute_gradients(self.loss)
        self.grad_rsgd,self.grad_adam = self.scale_grad(self.full_grad_list)

        gradients, variables = zip(*self.grad_adam)
        gradients = [tf.check_numerics(gr,'nan found: '+gr.name) for gr in gradients]
        gradients = [tf.clip_by_norm(gr, 1.0-1e-5) for gr in gradients]
        grad_adam_clipped = list(zip(gradients,variables)) 
        # Do two sets of gradients - one with just (mu,sigma) and one without for adam
        apply_rsgd_op, self.rsgd_steps = self.apply_rsgd(self.lr_rsgd)

        self.train_op_1 = self.optim_adam.apply_gradients(grad_adam_clipped)
        # full exp_map
        self.train_op_2 = tf.group([apply_rsgd_op,
                                    self.optim_adam.apply_gradients(grad_adam_clipped)]) 
        # via retraction approximation
        self.train_op_3 = tf.group([self.optim_rsgd.apply_gradients(self.grad_rsgd),
                                  self.optim_adam.apply_gradients(grad_adam_clipped),
                                  self.reproj_weights()])
        
        #self.train_op = self.optim_adam.minimize(self.loss)
        self.reproj_op = self.reproj_weights()
        
    def get_latent_var(self):
        var_list = []
        for v in tf.trainable_variables():
            if 'mu/bias' in v.name or 'sigma/bias' in v.name or 'hypbias' in v.name:
                var_list.append(v)
                
        return var_list
            
    @tf.custom_gradient
    def scale_grad_layer(x):
        def grad(g):
            return g*(1-var_norm)**2/4
        return tf.identity(x), grad
                      
    def scale_grad(self,gradient_list,c=1.0):
        grad_rsgd = []
        grad_adam = []
        for g,v in gradient_list:
            if 'mu/bias' in v.name or 'sigma/bias' in v.name or 'hypbias' in v.name:
                print(g,v)
                grad_rsgd.append((g*riemannian_gradient_c(v,c),v))
            else:
                grad_adam.append((g,v))
        return grad_rsgd, grad_adam
    
    def apply_rsgd(self,lr):
        hyp_vars = self.get_latent_var()
        hyp_grads = tf.gradients(self.loss, hyp_vars)
        clipped_hyp_grads = [tf.clip_by_norm(grad, 1.-1e-5) for grad in hyp_grads]  ###### Clip gradients

        def rsgd(v,g,lr):
            #return exp_map_x(v, -mobius_scalar_mul(lr,g,c=1.0), c=1.0)
            return tf.squeeze(exp_map_x(tf.expand_dims(v,0),-lr*tf.expand_dims(g,0), c=1.0),0)
            
        update_ops = []
        steps = []
        for i in range(len(hyp_vars)):
            riemannian_rescaling_factor = riemannian_gradient_c(hyp_vars[i], c=1.0)
            g_r = riemannian_rescaling_factor * clipped_hyp_grads[i]
            updated_var = rsgd(hyp_vars[i], g_r, lr)
            steps.append(updated_var)
            update_ops.append(tf.assign(hyp_vars[i], updated_var))
        #apply_op = tf.group([tf.assign(v,exp_map_x(v, -lr * g, c=1.0)) for (g,v) in zip(*grad_rsgd)])
        apply_op = tf.group(update_ops)
        return apply_op, steps
    
    def reproj_weights(self):
        #parameter_list = tf.trainable_variables()
        parameter_list = self.get_latent_var()
        #global_norm = tf.Print(tf.global_norm(parameter_list),[tf.global_norm(parameter_list)],first_n=5)
        global_norm = tf.global_norm(parameter_list)
        eps = 1e-5
        reproj_assign_op = tf.group([tf.assign(param,tf.squeeze(param/tf_norm(tf.expand_dims(param,0)))-eps) for param in parameter_list])
        reproj_op = tf.cond(global_norm>=1,true_fn=lambda:reproj_assign_op,false_fn=tf.no_op)

        return reproj_op

    #execute single forward and backward pass    
    #report loss
    def run_single_step(self,sess,x,training):
        _, loss, rloss, mloss, kloss, z_norm, zp_norm, mu, sigma, grads = sess.run([self.train_op_2,
                                    self.loss,
                                    self.recon_loss,
                                    self.mmd_loss,
                                    self.kld_loss,
                                    self.z_norm,
                                    self.z_prior_norm,
                                    self.mu,
                                    self.log_sigma,                                                                     
                                    self.full_grad_list], feed_dict={self.x: x, self.training: training})
        #rep = self.sess.run(self.reproj_op)
        #print(np.linalg.norm(mu),np.linalg.norm(sigma))

        return loss, mloss, rloss, kloss
        
    #reconstruction
    #x->x_hat
    def reconstruction(self,sess,x):
        training = False
        return sess.run([self.x_hat,self.z], feed_dict={self.x: x, self.training: training})
        
    #generation
    #z->x_hat
    def generate(self,sess,z):
        training = False
        return sess.run(self.x_hat,feed_dict={self.z: z, self.training: training})

    #transformation of features
    #x->z
    def transformer(self,sess,x):
        training = False
        return sess.run(self.z,feed_dict={self.x: x, self.training: training})

class VAE(object):
    def __init__(self,opts):
        self.learning_rate = opts['lr_adam']
        self.batch_size = opts['batch_size']
        self.n_z = opts['dim_z']
        self._opts = opts
                 
        #build a network
        self.build()
        
    #build a network
    def build(self):
        #input
        self.x = tf.placeholder(name='x',dtype=tf.float32,shape=[None,input_dim])  #shape=(batch_size,input_dim)

        #encoder
        #slim.fc(input,output_dim,scope,act_fn)
        if 'mnist' in self._opts['dataset']:
            if self._opts['nw_size'] == 'small':
                e = enc_mnist_small(self.x)
            else:
                e = enc_mnist(self.x)
        
        self.z_mu = fc(e,self.n_z,name='enc_fc4_mu',activation=None)
        #log(sigma^2)
        self.log_sigma_z_sq = fc(e,self.n_z,name='enc_fc4_sigma',activation=tf.nn.softplus)

        #N(z_mu,z_sigma)
        eps = tf.random_normal(shape=tf.shape(self.log_sigma_z_sq),mean=0,stddev=1,dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.log_sigma_z_sq)) * eps

        #decoder
        
        if 'mnist' in self._opts['dataset']:
            if self._opts['nw_size'] == 'small':
                self.x_hat_s, self.x_hat = dec_mnist_small(self.z)
            else:
                self.x_hat_, self.x_hat = dec_mnist(self.z)
        
        #loss
        #reconstruction loss
        # x<->x_hat
        #H(x,x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-10
        self.recon_loss = tf.reduce_mean(-tf.reduce_sum(self.x*tf.log(self.x_hat+epsilon)+(1-self.x)*tf.log(1-self.x_hat+epsilon),axis=1))
        
        #latent loss
        #KL divergence: measure the difference between two distributions
        #the latent distribution and N(0,1)
        params = {'nr_mmd_samples':1, 
                  'batch_size':self.batch_size,
                  'mmd_kernel':'RBF',
                  'embedding_dim':self.n_z,
                  'zdim':self.n_z,
                  'pz':'normal',
                  'pz_scale':1.0,
                  'c':1.0
        }
        self.latent_loss = tf.reduce_mean(-0.5*tf.reduce_sum(1+self.log_sigma_z_sq-tf.square(self.z_mu)- tf.exp(self.log_sigma_z_sq),axis=1))
        #self.mmd_loss = mmd_penalty(params,60000,eps,self.z)
        self.mmd_loss = mmdpp_penalty(params,self.z_mu,self.log_sigma_z_sq,eps)

        #total loss
        self.loss = self.recon_loss+self.latent_loss
        
        #optimiser
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
    #execute single forward and backward pass    
    #report loss
    def run_single_step(self,sess,x):
        _, rloss,kloss = sess.run([self.train_op,self.recon_loss,self.latent_loss], feed_dict={self.x: x})
        return rloss,kloss 
        
    #reconstruction
    #x->x_hat
    def reconstruction(self,sess,x):
        return sess.run(self.x_hat , feed_dict={self.x: x})
        
    #generation
    #z->x_hat
    def generate(self,sess,z):
        return sess.run(self.x_hat,feed_dict={self.z: z})

    #transformation of features
    #x->z
    def transformer(self,sess,x):
        return sess.run(self.z,feed_dict={self.x: x})