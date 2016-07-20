# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.Variable([[1, 2, 3, 4], [1, 1, 1, 1]], dtype=tf.float32)
# x = tf.cast(x, tf.float32)
mean, var = tf.nn.moments(x, [0])

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(x))

	print
	mean_val, var_val = sess.run([mean, var])
	print(mean_val)
	print(var_val)

