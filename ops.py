import tensorflow as tf
import numpy as np

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