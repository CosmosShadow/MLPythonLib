# coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cmtf.data.data_mnist as data_mnist

mnist = data_mnist.read_data_sets(one_hot=True)

print(mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(np.min(mnist.train.images), np.max(mnist.train.images))

# %% we can visualize any one of the images by reshaping it to a 28x28 image
plt.imshow(np.reshape(mnist.train.images[100, :], (28, 28)), cmap='gray')

n_input = 784
n_output = 10
net_input = tf.placeholder(tf.float32, [None, n_input])

# %% We can write a simple regression (y = W*x + b) as:
W = tf.Variable(tf.zeros([n_input, n_output]))
b = tf.Variable(tf.zeros([n_output]))
net_output = tf.nn.softmax(tf.matmul(net_input, W) + b)
y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_true * tf.log(net_output))
correct_prediction = tf.equal(tf.argmax(net_output, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# %% Now actually do some training:
batch_size = 100
n_epochs = 10
for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={net_input: batch_xs, y_true: batch_ys})
    print(sess.run(accuracy, feed_dict={net_input: mnist.validation.images,y_true: mnist.validation.labels}))

# %% Print final test accuracy:
print(sess.run(accuracy,feed_dict={net_input: mnist.test.images,y_true: mnist.test.labels}))

