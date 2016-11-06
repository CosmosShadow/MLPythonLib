# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np

state_is_tuple = True
batch_size = 1

data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
data_input = data[:-1].reshape(1, -1)
data_label = data[1:].reshape(1, -1)

x = tf.placeholder(tf.int32, shape=[None, None])		#batch size, time step, intput size(embedding_lookup)
y = tf.placeholder(tf.int32, shape=[None, None])

embeddings = tf.Variable(tf.random_uniform([10, 4], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, x)

cell1 = tf.nn.rnn_cell.BasicLSTMCell(32, state_is_tuple=state_is_tuple)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(32, state_is_tuple=state_is_tuple)
cells = [cell1, cell2]

cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=None, initial_state=initial_state, parallel_iterations=1, swap_memory=True)
logits_flat = tf.contrib.layers.linear(outputs, 10)


# GPU使用率
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8    #固定比例
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	sess.run(tf.initialize_all_variables())
	outputs_ = sess.run(logits_flat, feed_dict={x: data_input, y: data_label})
	print outputs_.shape






















