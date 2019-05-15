import tensorflow as tf
import numpy as np
fc = tf.layers.dense
from rietools import *

def mlr_layer(x,d,opts,c_val=1.0):
    A_mlr = []
    P_mlr = []
    logits_list = []
    for cl in range(d):
        A_mlr.append(tf.get_variable('A_mlr' + str(cl),
                                     dtype=tf.float32,
                                     shape=[1, opts['dim_z']],
                                     initializer=tf.contrib.layers.xavier_initializer()))

        P_mlr.append(tf.get_variable('P_mlr' + str(cl),
                                         dtype=tf.float32,
                                         shape=[1, opts['dim_z']],
                                         initializer=tf.constant_initializer(0.0)))

        minus_p_plus_x = mobius_add(-P_mlr[cl], x, c_val)
        norm_a = tf_norm(A_mlr[cl])
        lambda_px = tf_lambda_x(minus_p_plus_x, c_val)
        px_dot_a = tf_dot(minus_p_plus_x, tf.nn.l2_normalize(A_mlr[cl]))
        logit = 2. / np.sqrt(c_val) * norm_a * tf.asinh(np.sqrt(c_val) * px_dot_a * lambda_px)
        logits_list.append(tf.reshape(logit, [-1]))

    logits = tf.stack(logits_list, axis=1)
    return logits

def enc_simple(x,opts):
    f1 = fc(x,200,name='enc_fc1',activation=opts['nn_act_fn'])  #scope is the name of the operation
    #f2 = fc(f1,100,name='enc_fc2',activation=tf.nn.relu)  
    #f3 = fc(f2,100,name='enc_fc3',activation=None)
    return f1

def dec_simple(z,opts):
    g1 = fc(z,200,name='dec_f1',activation=opts['nn_act_fn'])
    #g2 = fc(g1,100,name='dec_f2',activation=tf.nn.relu)
    #g3 = fc(g2,100,name='dec_f3',activation=tf.nn.relu)
    x_hat = fc(g1,opts['input_dim'],name='dec_f4',activation=None)
    return x_hat, tf.nn.sigmoid(x_hat)

def dec_mlr(z,opts):
    g1 = mlr_layer(z,200,opts)
    #g2 = fc(g1,100,name='dec_f2',activation=tf.nn.relu)
    #g3 = fc(g2,100,name='dec_f3',activation=tf.nn.relu)
    x_hat = fc(g1,opts['input_dim'],name='dec_f4',activation=None)
    return x_hat, tf.nn.sigmoid(x_hat)

def enc_mlp(x,opts):
    f1 = fc(x,200,name='enc_fc1',activation=opts['nn_act_fn'])  #scope is the name of the operation
    f2 = fc(f1,100,name='enc_fc2',activation=opts['nn_act_fn'])  
    f3 = fc(f2,100,name='enc_fc3',activation=opts['nn_act_fn'])
    return f3

def dec_mlp(z,opts):
    g1 = fc(z,200,name='dec_f1',activation=opts['nn_act_fn'])
    g2 = fc(g1,100,name='dec_f2',activation=opts['nn_act_fn'])
    g3 = fc(g2,100,name='dec_f3',activation=opts['nn_act_fn'])
    x_hat = fc(g3,opts['input_dim'],name='dec_f4',activation=None)
    return x_hat, tf.nn.sigmoid(x_hat)


def hyp_fflayer(x,dim,activation=None,name='',c=1.0):
    with tf.variable_scope("hyp_ff_layer_"+name):
        W = tf.get_variable('weight',shape=[x.get_shape().as_list()[1],dim],
                           initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('hypbias',shape=[dim], initializer=tf.constant_initializer(0.0))
        # input x is Euclidean -> project to disk
        x = exp_map(x,c=1.0) + 1e-7
        #b = exp_map(tf.expand_dims(b,0),c=1.0) + 1e-7
        b = tf.expand_dims(b,0)
        Mx = tf.matmul(x,W) + 1e-7
        Mx_norm = tf.norm(Mx,2,axis=1,keepdims=True)
        x_norm = tf.norm(x,2,axis=1,keepdims=True)
        y = 1./tf.sqrt(c)*tf_tanh(Mx_norm/x_norm*tf_atanh(tf.sqrt(c)*x_norm))*Mx/Mx_norm
        y = project_hyp_vec(y,c)
        y = mobius_add(y,b,c)
        if activation is not None:
            # hyperbolic non-linearity
            y = exp_map(activation(log_map(y,c)),c)
        return y
    