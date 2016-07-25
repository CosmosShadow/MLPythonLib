# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.Variable([1, 2, 3])
x_pt = pt.wrap(x)
y = x_pt.apply(tf.mul, 5)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(y))