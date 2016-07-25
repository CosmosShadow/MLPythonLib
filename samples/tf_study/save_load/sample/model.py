# coding: utf-8
import tensorflow as tf
import prettytensor as pt
import numpy as np

def create_model():
	x = tf.placeholder("float", [2,10])
	W = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))
	b = tf.Variable(tf.truncated_normal([1], stddev=0.1))
	y = tf.matmul(x,W) + b
	return x, y