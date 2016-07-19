# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.Variable(0)
x_plus_1 = tf.assign_add(x, 1)

with tf.control_dependencies([x_plus_1]):
    y = tf.identity(x)

with tf.Session() as session:
	session.run(tf.initialize_all_variables())
	for i in xrange(5):
		print(y.eval())