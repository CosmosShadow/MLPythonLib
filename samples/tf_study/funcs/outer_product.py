# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

def outer_product(*inputs):
    inputs = list(inputs)
    for idx, input_ in enumerate(inputs):
        if len(input_.get_shape()) == 1:
            inputs[idx] = tf.reshape(input_, [-1, 1] if idx % 2 == 0 else [1, -1])
    output = tf.mul(inputs[0], inputs[1])
    return output

x = tf.Variable([1, 2, 3, 4, 5])
y = tf.Variable([4, 5, 6])
x = tf.reshape(x, [-1, 1])
y = tf.reshape(y, [1, -1])
z = tf.mul(x, y)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(z))