# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

data1 = tf.Variable([1, 2, 3, 4])
data2 = tf.Variable([1, 2, 3, 4])
data3 = tf.Variable([1, 2, 3, 4])
data_list1 = [data1]
data_list2 = [data1, data2]
reduce_data1 = reduce(lambda x, y: x*y, data_list1)
reduce_data2 = reduce(lambda x, y: x*y, data_list2)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(reduce_data1))
	print(sess.run(reduce_data2))