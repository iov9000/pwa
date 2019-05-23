import tensorflow as tf
import numpy as np
fc = tf.layers.dense
from rietools import *

def mlr_layer(x,d,opts,c=1.0):
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

        minus_p_plus_x = mobius_add(-P_mlr[cl], x, c)
        norm_a = tf_norm(A_mlr[cl])
        lambda_px = tf_lambda_x(minus_p_plus_x, c)
        px_dot_a = tf_dot(minus_p_plus_x, tf.nn.l2_normalize(A_mlr[cl]))
        logit = 2. / np.sqrt(c) * norm_a * tf.asinh(np.sqrt(c) * px_dot_a * lambda_px)
        logits_list.append(tf.reshape(logit, [-1]))

    logits = tf.stack(logits_list, axis=1)
    return logits

def enc_simple(x,opts):
    f1 = fc(x,200,name='enc_fc1',activation=opts['nn_act_fn'],use_bias=opts['use_bias'])  #scope is the name of the operation
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

fc = tf.layers.dense
def enc_mnist(x):
    f1 = fc(x,512,name='enc_fc1',activation=tf.nn.relu)  #scope is the name of the operation
    f2 = fc(f1,384,name='enc_fc2',activation=tf.nn.relu)  
    f3 = fc(f2,256,name='enc_fc3',activation=tf.nn.tanh)
    return f3

def dec_mnist(z):
    g1 = fc(z,256,name='dec_f1',activation=tf.nn.relu)
    g2 = fc(g1,384,name='dec_f2',activation=tf.nn.relu)
    g3 = fc(g2,512,name='dec_f3',activation=tf.nn.relu)
    x_hat = fc(g3,784,name='dec_f4',activation=None)
    return x_hat, tf.nn.sigmoid(x_hat)

def enc_mnist_small(x):
    f1 = fc(x,256,name='enc_fc1',activation=tf.nn.relu)  #scope is the name of the operation
    f2 = fc(f1,128,name='enc_fc2',activation=tf.nn.tanh)  
    return f2

def dec_mnist_small(z):
    g1 = fc(z,128,name='dec_f1',activation=tf.nn.relu)
    g2 = fc(g1,256,name='dec_f2',activation=tf.nn.relu)
    x_hat = fc(g2,784,name='dec_f4',activation=None)
    return x_hat, tf.nn.sigmoid(x_hat)

def batch_norm(_input, is_train, reuse, scope, scale=True):
    return tf.contrib.layers.batch_norm(
        _input, center=True, scale=scale,
        is_training=is_train, reuse=reuse, updates_collections=None,
        scope=scope, fused=False)

def conv2d(opts, input_, output_dim, d_h=2, d_w=2, scope=None,
                      conv_filters_dim=None, padding='SAME', l2_norm=False):
    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d works only with 4d tensors.'

    with tf.variable_scope(scope or 'conv2d'):
        w = tf.get_variable(
            'filter', [k_h, k_w, shape[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if l2_norm:
            w = tf.nn.l2_normalize(w, 2)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_dim],
            initializer=tf.constant_initializer(bias_start))
        conv = tf.nn.bias_add(conv, biases)

    return conv

def deconv2d(opts, input_, output_shape, d_h=2, d_w=2, scope=None, 
             conv_filters_dim=None, padding='SAME'):
    stddev = opts['init_std']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d_transpose works only with 4d tensors.'
    assert len(output_shape) == 4, 'outut_shape should be 4dimensional'

    with tf.variable_scope(scope or "deconv2d"):
        w = tf.get_variable(
            'filter', [k_h, k_w, output_shape[-1], shape[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape,
            strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
           
    return deconv

def enc_conv(opts, inputs, bn=False, is_training=False, reuse=False):
    num_units = opts['num_filters']
    num_layers = opts['num_layers']
    layer_x = inputs
    for i in range(num_layers):
        scale = 2**(num_layers - i - 1)
        layer_x = conv2d(opts, layer_x, num_units / scale,
                             scope='h%d_conv' % i)
        if bn:
            layer_x = batch_norm(layer_x, is_training,
                                     reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    
    return layer_x

def dec_conv(opts, z, output_shape, is_training=False, bn=False, reuse=False):
    num_units = opts['num_filters']
    batch_size = tf.shape(z)[0]
    num_layers = opts['num_layers']
    height = int(output_shape[0] / 2**num_layers)
    width = int(output_shape[1] / 2**num_layers)
    
    h0 = fc(z, num_units * height * width, name='h0_lin')
    h0 = tf.reshape(h0, [-1, height, width, num_units])
    h0 = tf.nn.relu(h0)
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2**(i + 1)
        _out_shape = [batch_size, int(height * scale),
                      int(width * scale), int(num_units / scale)]
        layer_x = deconv2d(opts, layer_x, _out_shape,
                               scope='h%d_deconv' % i)
        if bn:
            layer_x = ops.batch_norm(opts, layer_x,
                                     is_training, reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    _out_shape = [batch_size] + list(output_shape)
    last_h = deconv2d(
            opts, layer_x, _out_shape, scope='hfinal_deconv')
    return tf.nn.sigmoid(last_h), last_h