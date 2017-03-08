# coding: utf-8
import tensorflow as tf

# Import MNIST data
import cmtf.data.data_mnist as data_mnist

# Import itchat & threading
import itchat
import threading

# Create a running status flag
lock = threading.Lock()
running = False

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 100

from_name = '@c07ae10facf50d77e27756b79d8c2f6efedd5f5ac40e5e5bb250f7985d6be4b8'
to_name = '@c07ae10facf50d77e27756b79d8c2f6efedd5f5ac40e5e5bb250f7985d6be4b8'

def nn_train(to_name, param):
	global lock, running
	itchat.send(u'开工了...', to_name)
	# Lock
	with lock:
		running = True

	# mnist data reading
	mnist = data_mnist.read_data_sets(one_hot=True)

	# Parameters
	# learning_rate = 0.001
	# training_iters = 200000
	# batch_size = 128
	# display_step = 10
	learning_rate, training_iters, batch_size, display_step = param

	# Network Parameters
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_classes = 10 # MNIST total classes (0-9 digits)
	dropout = 0.75 # Dropout, probability to keep units

	# tf Graph input
	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.float32, [None, n_classes])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


	# Create some wrappers for simplicity
	def conv2d(x, W, b, strides=1):
		# Conv2D wrapper, with bias and relu activation
		x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
		x = tf.nn.bias_add(x, b)
		return tf.nn.relu(x)


	def maxpool2d(x, k=2):
		# MaxPool2D wrapper
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


	# Create model
	def conv_net(x, weights, biases, dropout):
		# Reshape input picture
		x = tf.reshape(x, shape=[-1, 28, 28, 1])

		# Convolution Layer
		conv1 = conv2d(x, weights['wc1'], biases['bc1'])
		# Max Pooling (down-sampling)
		conv1 = maxpool2d(conv1, k=2)

		# Convolution Layer
		conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
		# Max Pooling (down-sampling)
		conv2 = maxpool2d(conv2, k=2)

		# Fully connected layer
		# Reshape conv2 output to fit fully connected layer input
		fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
		fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
		fc1 = tf.nn.relu(fc1)
		# Apply Dropout
		fc1 = tf.nn.dropout(fc1, dropout)

		# Output, class prediction
		out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
		return out

	# Store layers weight & bias
	weights = {
		# 5x5 conv, 1 input, 32 outputs
		'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
		# 5x5 conv, 32 inputs, 64 outputs
		'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
		# fully connected, 7*7*64 inputs, 1024 outputs
		'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
		# 1024 inputs, 10 outputs (class prediction)
		'out': tf.Variable(tf.random_normal([1024, n_classes]))
	}

	biases = {
		'bc1': tf.Variable(tf.random_normal([32])),
		'bc2': tf.Variable(tf.random_normal([64])),
		'bd1': tf.Variable(tf.random_normal([1024])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}

	# Construct model
	pred = conv_net(x, weights, biases, keep_prob)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# Initializing the variables
	init = tf.initialize_all_variables()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		# Keep training until reach max iterations
		print('Wait for lock')
		with lock:
			run_state = running
		print('Start')
		while step * batch_size < training_iters and run_state:
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
			if step % display_step == 0:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
				print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
					"{:.6f}".format(loss) + ", Training Accuracy= " + \
					"{:.5f}".format(acc))
				itchat.send("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
					"{:.6f}".format(loss) + ", Training Accuracy= " + \
							"{:.5f}".format(acc), to_name)
			step += 1
			with lock:
				run_state = running
		print("Optimization Finished!")
		itchat.send("Optimization Finished!", to_name)

		# Calculate accuracy for 256 mnist test images
		print("Testing Accuracy:", \
			sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
		itchat.send("Testing Accuracy: %s" %
			sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}), to_name)

	with lock:
		running = False

@itchat.msg_register([itchat.content.TEXT])
def chat_trigger(msg):
	global lock, running, learning_rate, training_iters, batch_size, display_step, from_name, to_name
	print msg
	if msg['FromUserName'] == from_name:
		if msg['Text'] == u'开始':
			print('Starting')
			with lock:
				run_state = running
			if not run_state:
				try:
					threading.Thread(target=nn_train, args=(to_name, (learning_rate, training_iters, batch_size, display_step))).start()
				except:
					msg.reply('Running')
		elif msg['Text'] == u'停止':
			print('Stopping')
			with lock:
				running = False
		elif msg['Text'] == u'参数':
			itchat.send('lr=%f, ti=%d, bs=%d, ds=%d'%(learning_rate, training_iters, batch_size, display_step), to_name)
		else:
			try:
				print '---------params------------'
				param = msg['Text'].split()
				key, value = param
				print(key, value)
				if key == 'lr':
					learning_rate = float(value)
					print 'lr: ', learning_rate
				elif key == 'ti':
					training_iters = int(value)
				elif key == 'bs':
					batch_size = int(value)
				elif key == 'ds':
					display_step = int(value)
			except:
				pass


if __name__ == '__main__':
	itchat.auto_login(hotReload=True)
	itchat.run()
