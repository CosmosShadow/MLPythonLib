# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np


data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# data = np.ones([100])
data_input = data[:-1].reshape(1, -1)
data_label = data[1:]


# Network
batch_size = 1

x = tf.placeholder(tf.int64, shape=[batch_size, None])		#batch size, time-steps, intput size(embedding_lookup)
y = tf.placeholder(tf.int64, shape=[None])					#time-steps

embeddings = tf.Variable(tf.random_uniform([10, 4], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, x)

cell1 = tf.nn.rnn_cell.BasicLSTMCell(32)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(32)
cells = [cell1, cell2]
cell = tf.nn.rnn_cell.MultiRNNCell(cells)
initial_state_base = cell.zero_state(batch_size, tf.float32)
initial_state = tf.identity(initial_state_base, name='initial_state')
rnn_outputs, final_state_base = tf.nn.dynamic_rnn(cell, embed, sequence_length=None, initial_state=initial_state, parallel_iterations=1, swap_memory=True)
final_state = tf.identity(final_state_base, name='final_state')

# print cell1.state_size
# print cell.state_size
# print embed.get_shape()
print initial_state.get_shape()
other_input = tf.placeholder(tf.int64, shape=[1])
other_embed = tf.nn.embedding_lookup(embeddings, other_input)
other_state = tf.Variable(tf.zeros([1, 128])) 
# tf.placeholder(tf.float32, shape=[1, 128])
with tf.variable_scope('RNN', reuse=True):
	other_output, other_state_out = cell(other_embed, other_state)

rnn_outputs = pt.wrap(tf.reshape(rnn_outputs, [-1, 32]))
pt_outputs = rnn_outputs.fully_connected(10, activation_fn=None)
outputs = tf.identity(pt_outputs, name='outputs')

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, y))
train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(outputs,1), y), "float"))
train_op = tf.train.MomentumOptimizer(0.1, 0.9, use_nesterov=True).minimize(loss)

for var in tf.all_variables():
	print var.name, var.get_shape()
print initial_state.get_shape()
print final_state.get_shape()

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
		_, loss_ = sess.run([train_op, loss], feed_dict={x: data_input, y: data_label})
		if epoch%10 == 0:
			print loss_
	# 存checkpoint
	saver.save(sess, './model/checkpoint.ckpt')
	#Evaluate
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























