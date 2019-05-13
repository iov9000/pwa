import tensorflow as tf
from hae_utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def apply_rsgd(loss,lr):
    hyp_vars = get_latent_var()
    hyp_grads = tf.gradients(loss, hyp_vars)
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
                      
def scale_grad(gradient_list,c=1.0):
    grad_rsgd = []
    grad_adam = []
    for g,v in gradient_list:
        if 'mu/bias' in v.name or 'sigma/bias' in v.name or 'hypbias' in v.name:
            grad_rsgd.append((g*riemannian_gradient_c(v,c),v))
        else:
            grad_adam.append((g,v))
    return grad_rsgd, grad_adam
    
def get_latent_var():
        var_list = []
        for v in tf.trainable_variables():
            if 'mu/bias' in v.name or 'sigma/bias' in v.name or 'hypbias' in v.name:
                var_list.append(v)
                
        return var_list

class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

class OptimizerHVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, mmd_beta):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optim_adam = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_adam)  # Adam Optimizer
        self.optim_rsgd = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr_rsgd)  # RSGD Optimizer

        # Latent loss
        self.log_lik = self.cost
        #self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
        #                                                           tf.square(tf.exp(model.z_log_std)), 1))
        #self.kld_loss = kld_loss(model.z_mean,model.z_log_std,d=dim)        

        params = {'nr_mmd_samples':1, 
            'batch_size': 1,
            'mmd_kernel':'RBF',
            'embedding_dim': FLAGS.z_dim,
            'c':1.0
        }
        self.mmd_loss, self.res1, self.res2, self.res3 = mmd_penalty(params,60000,model.z_prior,model.z,FLAGS.sample_expmap)
        #self.cost -= self.kl
        self.cost += mmd_beta*self.mmd_loss

        self.grads_vars = self.optim_adam.compute_gradients(self.cost)
        self.grad_rsgd, self.grad_adam = scale_grad(self.grads_vars)
        gradients, variables = zip(*self.grad_adam)
        #gradients = [tf.check_numerics(gr,'nan'+gr.name) for gr in gradients]
        gradients = [tf.clip_by_norm(gr, params['c']-1e-5) for gr in gradients]
        grad_adam_clipped = list(zip(gradients,variables)) 
        # Do two sets of gradients - one with just (mu,sigma) and one without for adam
        apply_rsgd_op, self.rsgd_steps = apply_rsgd(self.cost,FLAGS.lr_rsgd)

        self.train_op_1 = self.optim_adam.apply_gradients(grad_adam_clipped)
        self.train_op_2 = tf.group([apply_rsgd_op,
                                    self.optim_adam.apply_gradients(grad_adam_clipped)]) 
        self.opt_op = self.train_op_2

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
