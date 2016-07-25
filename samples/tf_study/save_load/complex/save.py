# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist



with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())