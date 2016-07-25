# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.Variable(tf.ones([2, 10], tf.float32))
x_pt = pt.wrap(x).sequential()
x_pt.apply(tf.mul, 5)
x_pt.fully_connected(10, activation_fn=tf.nn.relu)
x_pt.fully_connected(1, activation_fn=tf.nn.relu)
y = x_pt.as_layer()		# 转成非sequential格式

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(y))