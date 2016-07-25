# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

@pt.Register
def leaky_relu(input_pt):
	return tf.select(tf.greater(input_pt, 0.0), input_pt, 0.01*input_pt)

x = tf.Variable([1, 2, 3, -3, -2, -1], dtype=tf.float32)
x_pretty = pt.wrap(x)
y = x_pretty.leaky_relu()

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(y))