# coding: utf-8

import tensorflow as tf
import numpy as np

with tf.Session() as sess:
	indices1 = tf.constant([0, 3])		#对应数据元素应放的重组后所在位置
	indices2 = tf.constant([1, 2])
	data1 = tf.constant([-1, -2])
	data2 = tf.constant([-3, -4])		#数据
	
	result = tf.dynamic_stitch([indices1, indices2], [data1, data2])
	print(result.eval())