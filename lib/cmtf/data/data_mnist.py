# 
import os
from File.FilePath import *
import tensorflow as tf
import cmtf.data.data_mnist_google as data_mnist_google

def read_data_sets(fake_data=False, one_hot=False, dtype=tf.float32):
	data_path = os.environ['HOME'] + '/data/tf/mnist/'
	NoExistsCreateDir(data_path)
	return data_mnist_google.read_data_sets(data_path, fake_data, one_hot, dtype)