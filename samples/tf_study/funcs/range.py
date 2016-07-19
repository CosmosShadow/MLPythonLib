# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

x = tf.range(1, 100+1, 2)

with tf.Session() as session:
	session.run(tf.initialize_all_variables())
	print(session.run(x))