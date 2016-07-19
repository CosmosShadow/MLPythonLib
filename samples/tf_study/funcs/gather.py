# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

data = tf.Variable([1, 2, 3, 4])
# indices = tf.Variable([-1])		-1不能取到
indices = tf.Variable([1, 1])
left = tf.gather(data, indices)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(left))