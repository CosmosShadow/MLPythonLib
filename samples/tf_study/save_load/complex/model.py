# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import cmtf.func.lstm_func as lstm_func
import numpy as np

def create_model(timesteps):
	input_timesteps_dim = 5
	x = tf.placeholder(tf.float32, [None, timesteps, input_timesteps_dim])
	x_reshape = lstm_func.reshape_data_to_lstm_format(x, per_example_length=input_timesteps_dim)
	y = (x_reshape
		.cleave_sequence(timesteps)
		.sequence_lstm(10, init=tf.truncated_normal)
		.squash_sequence()
		.fully_connected(1, activation_fn=tf.nn.relu))

	return x, y