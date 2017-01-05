# coding: utf-8

import tensorflow as tf

class TFObject(object):
	def restore(self, session, checkpoint_file, checkpoint_scope):
		with self.graph.as_default():
			var_dict = dict()
			for var in self.variables():
				inner_name = self.get_inner_name(var.name)
				var_dict[checkpoint_scope + '/' + inner_name] = var

			saver = tf.train.Saver(var_list=var_dict)
			saver.restore(session, checkpoint_file)

	def get_inner_name(self, scope_str):
		scope_str = scope_str[scope_str.find('/') + 1:]
		scope_str = scope_str[:scope_str.find(':')]
		return scope_str

class ClassName(object):
	"""docstring for ClassName"""
	def __init__(self, arg):
		super(ClassName, self).__init__()
		self.arg = arg
		