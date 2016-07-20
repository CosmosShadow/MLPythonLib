# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

data1 = tf.Variable([1, 2, 3, 4])
data2 = tf.Variable([1, 2, 3, 4])
data3 = tf.Variable([1, 2, 3, 4])
data_list = [data1, data2, data3]
data_added = tf.add_n(data_list)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(data_added))