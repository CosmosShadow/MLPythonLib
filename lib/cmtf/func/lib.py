# coding: utf-8

class HParams(object):
	def __init__(self, **init_hparams):
		object.__setattr__(self, 'keyvals', init_hparams)

	def __getattr__(self, key):
		return self.keyvals.get(key)

	def __setattr__(self, key, value):
		self.keyvals[key] = value

	def update(self, values_dict):
		self.keyvals.update(values_dict)

	def parse(self, values_string):
		self.update(ast.literal_eval(values_string))

	def values(self):
		return self.keyvals
















