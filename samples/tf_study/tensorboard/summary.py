# coding: utf-8
import os
import tensorflow as tf
import cmtf.data.data_mnist as data_mnist


save_path = 'output/checkpoint.ckpt'


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None,10])
W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
b = tf.Variable(tf.zeros([10]))
output = tf.nn.softmax(tf.matmul(x,W) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(output), [1]))

correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

global_step = tf.Variable(0, dtype=tf.float32)
learning_rate = tf.train.exponential_decay(0.1, global_step, 1000, 0.9, staircase=True, name='lr')
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

tf.scalar_summary('loss', loss)
tf.scalar_summary('accuracy', accuracy)
merged_summary = tf.merge_all_summaries()



mnist = data_mnist.read_data_sets(one_hot=True)


# run
with tf.Session() as sess:
	# init
	saver = tf.train.Saver()
	sess.run(tf.initialize_all_variables())
	sum_writer = tf.train.SummaryWriter('log', sess.graph)

	# restore
	if os.path.isfile(save_path):
		saver.restore(sess, save_path)

	# train
	global_step_ = sess.run(global_step)
	while global_step_ < 1000:
		batch_xs, batch_ys = mnist.train.next_batch(100)
		_, ac, merged_summary_, lr, global_step_ = sess.run(
		                                                    [train_op, accuracy, merged_summary, learning_rate, global_step], 
		                                                    feed_dict={x: batch_xs, y: batch_ys})
		sum_writer.add_summary(merged_summary_, global_step_)

		summary = tf.Summary(value=[tf.Summary.Value(tag="step", simple_value=float(global_step_))])
		sum_writer.add_summary(summary, global_step_)

		if global_step_ % 100 == 0:
			print 'epoch: %d   lr: %.2f   acc: %.3f' % (global_step_, lr, ac)
			saver.save(sess, save_path)
	# test
	test_ac = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
	print test_ac

sum_writer.close()













