# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

data = tf.Variable([1, 2, 3, 4])
indices = [1, 2]
updates = [2, 3]
data = tf.scatter_add(data, indices, updates)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(data))