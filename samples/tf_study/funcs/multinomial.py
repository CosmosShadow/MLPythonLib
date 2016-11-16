# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cmtf.data.data_mnist as data_mnist

student_count = 3
question_count = 2
logit = tf.truncated_normal([student_count, question_count])
# possiblity = tf.sigmoid(logit)
possiblity = tf.Variable([[1, 0], [0.5, 0.5], [0.8, 0.2]])

one_possibility = tf.expand_dims(possiblity, -1)
zero_possibility = 1 - one_possibility

tow_possibility = tf.concat(2, [zero_possibility, one_possibility])
tow_possibility_flat = tf.reshape(tow_possibility, [-1, 2])

# samples = tf.multinomial(tf.log([[10., 10.], [0, 10]]), 10)
samples_flat = tf.multinomial(tf.log(tow_possibility_flat), 1)
# samples_flat = tf.multinomial(tow_possibility_flat, 1)
samples = tf.reshape(samples_flat, [student_count, question_count])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())

# saver = tf.train.Saver()
# saver.restore(sess, checkpoint_path)
# tf.train.write_graph(sess.graph.as_graph_def(), graph_dir, graph_name, as_text=False)
# saver.save(sess, ckpt_path)
samples_sumary = None
count = 10000
for _ in range(count):
	pass
	possiblity_, tow_possibility_, samples_ = sess.run([possiblity, tow_possibility, samples])
# print possiblity_
# print tow_possibility_[:, :, 0]
# print tow_possibility_[:, :, 1]
	# print samples_
	# print ''
	if samples_sumary == None:
		samples_sumary = samples_
	else:
		samples_sumary += samples_

print samples_sumary / (count*1.0)

sess.close()