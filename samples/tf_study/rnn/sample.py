# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np


# 数据
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
data_input = data[:-1].reshape(1, -1)
data_label = data[1:]


# 网络结构
batch_size = 1

x = tf.placeholder(tf.int64, shape=[None, None])		#batch size, time step, intput size(embedding_lookup)
y = tf.placeholder(tf.int64, shape=[None])

embeddings = tf.Variable(tf.random_uniform([10, 4], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, x)

cell1 = tf.nn.rnn_cell.BasicLSTMCell(32)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(32)
cells = [cell1, cell2]

cell = tf.nn.rnn_cell.MultiRNNCell(cells)
initial_state = cell.zero_state(batch_size, tf.float32)

rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, sequence_length=None, initial_state=initial_state, parallel_iterations=1, swap_memory=True)
rnn_outputs = pt.wrap(tf.reshape(rnn_outputs, [-1, 32]))
outputs = rnn_outputs.fully_connected(10, activation_fn=None)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, y))
train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs,1), y), "float"))
train_op = tf.train.MomentumOptimizer(0.1, 0.9, use_nesterov=True).minimize(loss)


# GPU使用率
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5    #固定比例
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	sess.run(tf.initialize_all_variables())
	# train
	for epoch in range(100):
		_, loss_ = sess.run([train_op, loss], feed_dict={x: data_input, y: data_label})
		if epoch%10 == 0:
			print loss_

	print "generate start with 0: "
	initial_state_ = sess.run(initial_state)
	input_x = np.zeros((1, 1))
	for _ in range(8):
		feed_dict = {x: input_x, initial_state: initial_state_}
		outputs_, final_state_ = sess.run([outputs, final_state], feed_dict)
		initial_state_ = final_state_
		output_class = outputs_[0].argmax()
		input_x[0][0] = output_class
		print output_class
























