# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.Variable([[1, 2, 3, 4], [5, 6, 7, 8]])
x_sum = tf.reduce_sum(x, 1)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print sess.run(x)
	print sess.run(x_sum)