# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np


data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# data = np.ones([100])
data_input = data[:-1].reshape(1, -1)
data_label = data[1:]

# Hyparameters
input_class_num = 10
embed_vector_size = 4
lstm_size_1 = 32
lstm_size_2 = 32
fc_size = 10


# Train Network
batch_size = 1

train_x = tf.placeholder(tf.int64, shape=[batch_size, None])		#batch size, time-steps, intput size(embedding_lookup)
train_y = tf.placeholder(tf.int64, shape=[None])					#time-steps

embeddings = tf.Variable(tf.random_uniform([input_class_num, embed_vector_size], -1.0, 1.0))
train_embed = tf.nn.embedding_lookup(embeddings, train_x)

cell1 = tf.nn.rnn_cell.BasicLSTMCell(lstm_size_1)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(lstm_size_2)
cells = [cell1, cell2]
multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
train_initial_state = multi_cell.zero_state(batch_size, tf.float32)
train_rnn_outputs, train_final_state = tf.nn.dynamic_rnn(multi_cell, train_embed, sequence_length=None, initial_state=train_initial_state, parallel_iterations=1, swap_memory=True)

def full_connect(rnn_outputs):
	rnn_outputs_wrap = pt.wrap(tf.reshape(rnn_outputs, [-1, lstm_size_2]))
	pt_outputs = rnn_outputs_wrap.fully_connected(fc_size, activation_fn=None)
	return pt_outputs

with tf.variable_scope('full_connect'):
	train_outputs = full_connect(train_rnn_outputs)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(train_outputs, train_y))
train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_outputs,1), train_y), "float"))
train_op = tf.train.MomentumOptimizer(0.1, 0.9, use_nesterov=True).minimize(loss)


# Run Network
x = tf.placeholder(tf.int64, shape=[1], name="x")
embed = tf.nn.embedding_lookup(embeddings, x)
initial_state = tf.Variable(tf.zeros(train_initial_state.get_shape()))
with tf.variable_scope('RNN', reuse=True):
	rnn_outputs, final_state_base = multi_cell(embed, initial_state)
with tf.variable_scope('full_connect', reuse=True):
	pt_outputs = full_connect(rnn_outputs)
# assign_op = initial_state.assign(final_state)
# with tf.control_dependencies([assign_op]):
final_state = tf.identity(final_state_base, name="final_state")
outputs = tf.identity(pt_outputs, name="outputs")

for var in tf.all_variables():
	print var.name, var.get_shape()

# GPU setting
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5    #固定比例
config.gpu_options.allow_growth = True

saver = tf.train.Saver()

with tf.Session(config=config) as sess:
	sess.run(tf.initialize_all_variables())
	# 存graph
	tf.train.write_graph(sess.graph.as_graph_def(), './model/', 'graph_def', as_text=False)
	# Train
	for epoch in range(100):
		_, loss_ = sess.run([train_op, loss], feed_dict={train_x: data_input, train_y: data_label})
		if epoch%10 == 0:
			print loss_
	# 存checkpoint
	saver.save(sess, './model/checkpoint.ckpt')
	#Evaluate
	input_x = np.zeros((1))
	initial_state_ = sess.run(initial_state)
	for _ in range(8):
		outputs_, initial_state_ = sess.run([outputs, final_state], feed_dict={x: input_x, initial_state: initial_state_})
		output_class = outputs_.argmax()
		input_x[0] = output_class
		print output_class























