import os
import tensorflow as tf
import numpy as np
from rietools import *
from tools import *
from layers import *

class PWA(object):
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
        self.act_mu = opts['enc_act_mu']
        self.act_sigma = opts['enc_act_sigma']
        
        #build a network
        self.build()
        #launch a session
        #self.sess.run(tf.global_variables_initializer())  
        checkpoint_dir = os.path.expanduser("./"+'/ckpt')
        saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=checkpoint_dir,
            save_steps=3000,
            saver=tf.train.Saver())
            #hooks = [saver_hook]
        hooks = []
        self.sess = tf.train.SingularMonitoredSession(hooks=hooks,checkpoint_dir=checkpoint_dir)
        
    #build a network
    def build(self):
        #input
        self.x = tf.placeholder(name='x',dtype=tf.float32,shape=[None,self._opts['input_dim']])  #shape=(batch_size,input_dim)
        #self.bs_ph = tf.placeholder(name='bs',dtype=tf.int32,shape=())
        #encoder
        if self._opts['nw_type'] == 'simple':
            e = enc_simple(self.x,self._opts)
        else:
            e = enc_mlp(self.x,self._opts)

        # project Euclidean (mu,sigma) to Poincaré disk
        #self.mu = exp_map(tf.layers.dense(f3,self.dim_z,name='enc_mu',activation=None),c=self.c)
        #self.log_sigma = exp_map(tf.layers.dense(f3,self.dim_z,name='enc_sigma',activation=tf.nn.softplus),c=self.c)
        
        # make these layers fully hyperbolic
        self.mu = hyp_fflayer(e,self.dim_z,name='mu',activation=self.act_mu)

        #self.mu = exp_map(tf.layers.dense(e,self.dim_z,name='mu',activation=None),c=self.c)
        if self._opts['scalar_dispersion']:
            if self._opts['hyp_var']:
                self.log_sigma = hyp_fflayer(e,1,name='sigma',activation=self.act_sigma)
            else:
                self.log_sigma = tf.layers.dense(e,1,name='sigma',activation=self.act_sigma)
        else:
            if self._opts['hyp_var']:
                self.log_sigma = hyp_fflayer(e,self.dim_z,name='sigma',activation=self.act_sigma)
            else:    
                self.log_sigma = tf.layers.dense(e,self.dim_z,name='sigma',activation=self.act_sigma)

        #self.mu = self.mu/tf.norm(self.mu,2,axis=1,keepdims=True)
        #self.log_sigma = self.log_sigma/tf.norm(self.log_sigma,2,axis=1,keepdims=True)

        #N(z_mu,z_sigma)
        #eps = tf.random_normal(shape=tf.shape(self.log_sigma_z_sq),mean=0,stddev=1,dtype=tf.float32)
        #self.z = tf.add(self.z_mu,tf.sqrt(1e-8 + tf.exp(self.log_sigma_z_sq)) * eps,name='sample')
        if self._opts['reparam_eucl']:
            self.z, eps_sample = reparam_h_2(self.mu, self.log_sigma, prior_var=self._opts['prior_var'])
        else:
            self.z, eps_sample = reparam_h(self.mu, self.log_sigma, radius=1.0, 
                                         c=self.c, use_expmap=self.use_expmap, 
                                         scalar_dispersion=self._opts['scalar_dispersion'],
                                         prior_var=self._opts['prior_var'])
        self.mu_norm = tf.norm(self.mu,2)
        self.sigma_norm = tf.norm(self.log_sigma,2)
        self.z_norm = tf.norm(self.z,2)
        self.logmap_z_norm = tf.norm(log_map(self.z),2)
        
        if self._opts['nw_type'] == 'simple':
            #self.x_hat, self.x_hat_s = dec_simple(tf_log_map_x(self.mu,self.z,self.c),self._opts)
            if self._opts['dec_mlr']:
                self.x_hat, self.x_hat_s = dec_mlr(self.z,self._opts)
            else:
                self.x_hat, self.x_hat_s = dec_simple(log_map(self.z,self.c),self._opts)
            
        else:
            #self.x_hat, self.x_hat_s = dec_mlp(tf_log_map_x(self.mu,self.z,self.c),self._opts)
            self.x_hat, self.x_hat_s = dec_mlp(log_map(self.z,self.c),self._opts)

        #self.x_hat, self.x_hat_s = dec_simple(self.z,self._opts)        
        epsilon = 1e-10
        self.bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.x_hat)
        if self._opts['loss'] == 'bce':
            self.recon_loss = tf.reduce_mean(tf.reduce_sum(self.bce, axis=-1))
            #x = tf.sigmoid(self.x)
            #x = self.x
            #self.recon_loss = tf.reduce_mean(-tf.reduce_sum(x*tf.log(self.x_hat_s+epsilon)+(1-x)*tf.log(1-self.x_hat_s+epsilon),axis=1))
            #self.recon_loss = tf.reduce_mean(-tf.reduce_sum(self.x*tf.log(self.x_hat+epsilon)+(1-self.x)*tf.log(1-self.x_hat+epsilon),axis=1))
        else:
            # non deterministic decoder version of L2-loss (N(G_x(Z),0.3*I_n) -> WAE paper)
            #self.recon_loss = self._opts['recon_weight']*tf.reduce_mean(tf.nn.l2_loss((self.x_hat + 0.3*tf.random_normal(tf.shape(self.x_hat))) - self.x))
            self.recon_loss = self._opts['recon_weight']*tf.reduce_mean(tf.nn.l2_loss((self.x_hat - self.x)))

            #self.recon_loss = tf.losses.mean_squared_error(labels=self.x,predictions=self.x_hat)

        #self.kld_loss = kld_loss(self.mu,self.log_sigma,d=self.dim_z)        
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
        if self._opts['use_expmap']:
            prior_sample = exp_map(self._opts['prior_var']*tf.random_normal(tf.shape(self.mu)),self.c)
        else:
            #prior_sample = project_hyp_vec(sample_gauss_pd(tf.shape(self.mu),radius=0.88,sigma=self._opts['prior_var']),c=self.c)
            prior_sample = project_hyp_vec(sample_gauss_hn_(tf.shape(self.mu),radius=1.0,sigma=self._opts['prior_var']),c=self.c)

        self.z_prior_norm = tf.norm(prior_sample,2)
        self.mmd_loss = mmd_penalty(mmd_params,prior_sample,self.z,sigma=self._opts['sigma_mmd'],q=self.q)
        self.varreg = tf.reduce_mean(tf.norm(self.log_sigma,self.varregnorm,axis=1))
        
        # one sample MC approximation of KLD
        logp_prior = tf.log(gauss_prob_pd(prior_sample,self.mu,0))#tf.exp(0.5*self.log_sigma)))
        logp_posterior = tf.log(gauss_prob_pd(self.z,self.mu,0))#tf.exp(0.5*self.log_sigma)))
        self.kld_loss = tf.reduce_mean(logp_prior - logp_posterior,0) 
    
        #total loss
        self.loss = self.recon_loss + self.beta_mmd*self.mmd_loss + self.lambda_mmd*self.varreg
        print(mc_elbo(self.x_hat, self.z, self.mu, self.log_sigma))
        #self.loss = tf.reduce_mean(mc_elbo(self.x_hat, self.z, self.mu, self.log_sigma),0)
        #self.loss = self.recon_loss - self.beta_kld*self.kld_loss

        #optimiser
        self.optim_adam = tf.train.AdamOptimizer(learning_rate=self.lr_adam)#,beta1=0.5)
        self.optim_rsgd = tf.train.GradientDescentOptimizer(learning_rate=self.lr_rsgd)
        self.full_grad_list = self.optim_rsgd.compute_gradients(self.loss)
        self.grad_rsgd,self.grad_adam = self.scale_grad(self.full_grad_list)

        gradients, variables = zip(*self.grad_adam)
        gradients = [tf.check_numerics(gr,'nan found: '+gr.name) for gr in gradients]
        gradients = [tf.clip_by_norm(gr, 1.0-1e-5) for gr in gradients]
        grad_adam_clipped = list(zip(gradients,variables)) 
        # Do two sets of gradients - one with just (mu,sigma) and one without for adam
        apply_rsgd_op, self.rsgd_steps = self.apply_rsgd(self.lr_rsgd)


        self.train_op_adam_clipped = self.optim_adam.apply_gradients(grad_adam_clipped)
        self.train_op_adam = self.optim_adam.apply_gradients(self.grad_adam)

        # full exp_map
        self.train_op_adam_rsgd = tf.group([apply_rsgd_op,
                                    self.optim_adam.apply_gradients(grad_adam_clipped)]) 
        # via retraction approximation
        self.train_op_adam_rsgd_retract = tf.group([self.optim_rsgd.apply_gradients(self.grad_rsgd),
                                  self.optim_adam.apply_gradients(grad_adam_clipped),
                                  self.reproj_weights()])
        

        self.check_op = tf.add_check_numerics_ops()
        self.summarize_op = self.summarize()
        t = tf.trainable_variables()
        norms = []
        for var in t:
            print(var)
            norms.append(tf.nn.l2_loss(var))
            
        self.norms = norms
        #self.train_op = self.optim_adam.minimize(self.loss)
        self.reproj_op = self.reproj_weights()
        
    def get_latent_var(self):
        var_list = []
        for v in tf.trainable_variables():
            #if 'mu/bias' in v.name or 'sigma/bias' in v.name or 'hypbias' in v.name:
            if 'mu/bias' in v.name or 'P_mlr' in v.name or 'hypbias' in v.name:
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
            if 'mu/bias' in v.name or 'P_mlr' in v.name or 'hypbias' in v.name:
            #if 'mu/bias' in v.name or 'sigma/bias' in v.name or 'hypbias' in v.name:
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
        reproj_assign_op = tf.group([tf.assign(param,tf.squeeze(param/tf_norm(tf.expand_dims(param,0)),0)-eps) for param in parameter_list])
        reproj_op = tf.cond(global_norm>=1,true_fn=lambda:reproj_assign_op,false_fn=tf.no_op)

        return reproj_op

    def summarize(self):
        t = tf.trainable_variables()
        summaries = []
        for var in t:
            summaries.append(tf.summary.scalar("var/" + var.name, tf.norm(var)))
        
        return tf.summary.merge(summaries)
    
    #execute single forward and backward pass    
    #report loss
    def run_single_step(self,x):
        #writer = tf.summary.FileWriter('./graphs', self.sess.graph)
        _, logmap_norm, norms, x, x_hat, loss, rloss, mloss, kloss, z_norm, zp_norm, mu, sigma, z = self.sess.run([self.train_op_adam_rsgd,
                                    self.logmap_z_norm,
                                    self.norms,
                                    self.x,
                                    self.x_hat_s,
                                    self.loss,
                                    self.recon_loss,
                                    self.mmd_loss,
                                    self.kld_loss,
                                    self.z_norm,
                                    self.z_prior_norm,
                                    self.mu,
                                    self.log_sigma,                                                                     
                                    self.z], feed_dict={self.x: x})
        
        #print(logmap_norm)
        #print("Encoder output: Mu norm", np.linalg.norm(mu))
        #print("Encoder output: Log sigma norm", np.linalg.norm(sigma))

        #step = tf.train.get_global_step()
        #writer.add_summary(summary,step)
        #rep = self.sess.run(self.reproj_op)
        #print(np.linalg.norm(mu),np.linalg.norm(sigma))

        return loss, mloss, rloss, kloss, np.mean(mu,0), np.mean(sigma,0)
        
    #reconstruction
    #x->x_hat
    def reconstruction(self,x):
        return self.sess.run([self.x_hat,self.z], feed_dict={self.x: x})
        
    #generation
    #z->x_hat
    def generate(self,z):
        return self.sess.run(self.x_hat,feed_dict={self.z: z})

    #transformation of features
    #x->z
    def transformer(self,x):
        return self.sess.run(self.z,feed_dict={self.x: x})
    
    def transformer_mu(self,x):
        return self.sess.run(self.mu,feed_dict={self.x: x})