# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.Variable([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=tf.float32)
x_0 = tf.reduce_sum(x, 0)
x_1 = tf.reduce_sum(x, 1)
x_all = tf.reduce_sum(x)

mean_0, var_0 = tf.nn.moments(x, [0])
mean_1, var_1 = tf.nn.moments(x, [1])

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print sess.run(x)
	print
	print sess.run(x_0)
	print sess.run(x_1)
	print sess.run(x_all)
	print
	print sess.run(mean_0)
	print sess.run(var_0)
	print
	print sess.run(mean_1)
	print sess.run(var_1)