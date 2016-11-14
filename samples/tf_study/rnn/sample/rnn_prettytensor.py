# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np


data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
data_input = data[:-1].reshape(1, -1)
data_label = data[1:]


# Network
batch_size = 1
length = 1
class_num = 7

x = tf.placeholder(tf.int32, shape=[1, 1], name='x')
y = tf.placeholder(tf.int32, shape=[1])

with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
	x_wrap = pt.wrap(x)
	embedded = x_wrap.embedding_lookup(10, [class_num]).cleave_sequence(1)
	pt_outputs = (embedded.sequence_lstm(32).sequence_lstm(32).squash_sequence().fully_connected(class_num, activation_fn=None))

outputs = tf.identity(pt_outputs, name='outputs')

tf.add_to_collection('x', x)
tf.add_to_collection('outputs', outputs)

# GPU setting
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5    #固定比例
config.gpu_options.allow_growth = True

saver = tf.train.Saver()

with tf.Session(config=config) as sess:
	sess.run(tf.initialize_all_variables())
	# 存graph
	tf.train.write_graph(sess.graph.as_graph_def(), './model/', 'graph_def', as_text=False)
	saver.save(sess, './model/checkpoint.ckpt')
	#Evaluate
	input_x = np.zeros((1, 1))
	for _ in range(8):
		outputs_ = sess.run([outputs], feed_dict={x: [[1]]})
		print outputs_[0].argmax()























