# coding: utf-8
from __future__ import division
import os.path
import time
import tensorflow as tf
import cmtf.data.data_mnist as data_mnist
import mnist

# 参数
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')

# 生成placeholder
def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS), name='images')
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size), name='labels')
  return images_placeholder, labels_placeholder

# 填充feed_dict
def fill_feed_dict(data_set, images_pl, labels_pl):
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
  feed_dict = {images_pl: images_feed,labels_pl: labels_feed}
  return feed_dict

# 评估
def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

# 训练
def run_training():
  data_sets = data_mnist.read_data_sets()
  with tf.Graph().as_default():
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
    logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
    loss = mnist.loss(logits, labels_placeholder)
    train_op = mnist.training(loss, FLAGS.learning_rate)
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver()
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # Start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      duration = time.time() - start_time

      if step % 100 == 0:
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)
        do_eval(sess,eval_correct, images_placeholder, labels_placeholder, data_sets.train)
        do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
        do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)

def main(_):
    # 清掉文件夹
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  run_training()

if __name__ == '__main__':
  tf.app.run()
