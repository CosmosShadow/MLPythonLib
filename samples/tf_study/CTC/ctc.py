# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

inputs = tf.random_uniform([100, 1, 10], -1.0, 1.0)
labels = tf.SparseTensor(indices=[[0, 0], [0, 1], [0, 2]], values=[1, 2, 3], shape=[3, 4])
sequence_length = tf.constant(np.array([100]*1), tf.int32)
loss = tf.nn.ctc_loss(inputs, labels, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True)


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())

print sess.run(loss)


sess.close()