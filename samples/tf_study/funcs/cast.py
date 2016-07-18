# coding: utf-8
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
	a = tf.Variable('12')
	b = tf.string_to_number(a)
	sess.run(tf.initialize_all_variables())
	print(b.eval())