# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.reshape(tf.range(1, 100+1, 1), [20, 5])
y = tf.reverse(x, [True, True])

with tf.Session() as session:
	session.run(tf.initialize_all_variables())
	print(session.run(x))
	print(session.run(y))