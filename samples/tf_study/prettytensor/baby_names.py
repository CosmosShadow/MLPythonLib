# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
import prettytensor as pt
import cmtf.func.lstm_func as lstm_func
import cmtf.func.permute as permute
import cmtf.data.data_baby_names as data_baby_names

tf.app.flags.DEFINE_string('save_path', None, 'Where to save the model checkpoints on local disk. Checkpoints are in LevelDb.')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 32
CHARS = 128
TIMESTEPS = 15
SEXES = 2
EMBEDDING_SIZE = 16


def create_model(text_in, labels, timesteps, per_example_weights, phase=pt.Phase.train):
  with pt.defaults_scope(phase=phase, l2loss=0.00001):
    with tf.device('/cpu:0'):
      embedded = text_in.embedding_lookup(CHARS, [EMBEDDING_SIZE])
    lstm = (embedded
      .cleave_sequence(timesteps)
      # .sequence_lstm(CHARS)
      .sequence_lstm(CHARS))
    return (lstm
      .squash_sequence()
      .fully_connected(32, activation_fn=tf.nn.relu)
      .dropout(0.7)
      .softmax_classifier(SEXES, labels, per_example_weights=per_example_weights))


def main(_=None):
  print('Starting Baby Names')
  input_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE, TIMESTEPS])
  output_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, SEXES])
  inp = lstm_func.reshape_data_to_lstm_format(input_placeholder)

  # Create a label for each timestep.
  lables_1 = tf.reshape(tf.tile(output_placeholder, [1, TIMESTEPS]), [BATCH_SIZE, TIMESTEPS, SEXES])
  labels = lstm_func.reshape_data_to_lstm_format(lables_1, per_example_length=2)

  length_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE, 1])
  t = tf.concat(1, [tf.constant(numpy.arange(BATCH_SIZE).reshape((BATCH_SIZE, 1)), dtype=tf.int32), length_placeholder])
  per_example_weights = lstm_func.reshape_data_to_lstm_format(tf.sparse_to_dense(t, [BATCH_SIZE, TIMESTEPS], 1.0, default_value=0.0)).squeeze()

  with tf.variable_scope('baby_names'):
    result = create_model(inp, labels, TIMESTEPS, per_example_weights)
  with tf.variable_scope('baby_names', reuse=True):
    test_result = create_model(inp, labels, TIMESTEPS, per_example_weights, phase=pt.Phase.test)

  accuracy = test_result.softmax.evaluate_classifier(labels, phase=pt.Phase.test, per_example_weights=per_example_weights)
  batch_accuracy = result.softmax.evaluate_classifier(labels, phase=pt.Phase.train, per_example_weights=per_example_weights)

  names, sex, lengths = data_baby_names.baby_names(TIMESTEPS)

  epoch_size = len(names) // BATCH_SIZE
  optimizer = tf.train.AdagradOptimizer(tf.train.exponential_decay(1.0, pt.global_step(), epoch_size, 0.95, staircase=True))
  train_op = pt.apply_optimizer(optimizer, losses=[result.loss])

  # We can set a save_path in the runner to automatically checkpoint every so
  # often.  Otherwise at the end of the session, the model will be lost.
  runner = pt.train.Runner(save_path=FLAGS.save_path)
  
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
  sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

  # with tf.Session():
  for epoch in xrange(100):
    # Shuffle the training data.
    names, sex, lengths = permute.permute_data((names, sex, lengths))

    runner.train_model(
        train_op,
        [result.loss, batch_accuracy],
        epoch_size,
        feed_vars=(input_placeholder, output_placeholder, length_placeholder),
        feed_data=pt.train.feed_numpy(BATCH_SIZE, names, sex, lengths),
        print_every=100)
    classification_accuracy = runner.evaluate_model(
        accuracy,
        epoch_size,
        print_every=0,
        feed_vars=(input_placeholder, output_placeholder, length_placeholder),
        feed_data=pt.train.feed_numpy(BATCH_SIZE, names, sex, lengths))

    print('Accuracy after epoch %d: %g%%' % (
        epoch + 1, classification_accuracy * 100))


if __name__ == '__main__':
  tf.app.run()
