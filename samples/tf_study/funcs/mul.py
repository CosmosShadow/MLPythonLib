# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.Variable([1, 2, 3, 4])
y = tf.Variable([1, 2, 3, 4])
z = tf.mul(tf.reshape(x, [1, -1]), tf.reshape(y, [-1, 1]))

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print sess.run(z)