"""APL 2.0 code from github.com/pkmital/tensorflow_tutorials w/ permission
from Parag K. Mital.
"""
import math
import tensorflow as tf
from tensorflow.python import control_flow_ops


def batch_norm(x, phase_train, scope='bn', affine=True):
    with tf.variable_scope(scope):
        og_shape = x.get_shape().as_list()
        if len(og_shape) == 2:
            x = tf.reshape(x, [-1, 1, 1, og_shape[1]])
        shape = x.get_shape().as_list()
        beta = tf.Variable(tf.constant(0.0, shape=[shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[shape[-1]]), name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, affine)
        if len(og_shape) == 2:
            normed = tf.reshape(normed, [-1, og_shape[-1]])
    return normed


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def linear(x, n_units, scope=None, stddev=0.02, activation=lambda x: x):
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], n_units], tf.float32, tf.random_normal_initializer(stddev=stddev))
        return activation(tf.matmul(x, matrix))

def conv2d(x, n_filters, k_h=5, k_w=5, stride_h=2, stride_w=2, stddev=0.02, activation=None, bias=True, padding='SAME', name="Conv2D"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], n_filters], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
        if bias:
            b = tf.get_variable('b', [n_filters], initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.bias_add(conv, b)
        if activation:
            conv = activation(conv)
        return conv
