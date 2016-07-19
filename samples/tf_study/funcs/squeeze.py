# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.Variable(tf.constant(0.0, shape=[1, 32, 32, 16]), name='x', trainable=True)
y = tf.squeeze(x)
y_shape = tf.shape(y)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(y_shape))