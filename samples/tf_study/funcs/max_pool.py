# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

kernal = 5
stride = 2
x = tf.Variable(tf.constant(0.0, shape=[1, 32, 32, 16]), name='x', trainable=True)
# pool = tf.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
pool = tf.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
with tf.control_dependencies([pool]):
	og_shape = pool.get_shape().as_list()
	print(og_shape)

with tf.Session() as session:
	session.run(tf.initialize_all_variables())
	session.run(pool)