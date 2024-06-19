import tensorflow as tf
import numpy as np
from collections import defaultdict

class TensorLogger(object):
	"""docstring for TensorLogger"""
	def __init__(self, log_dir, parameters, scalars, delimiter = "."):
		super(TensorLogger, self).__init__()
		self.parameters = parameters
		self.scalars = scalars
		self.log_dir = log_dir
		self.pplaceholders = {}
		self.splaceholders = {}
		self.delimiter = delimiter

		with tf.Graph().as_default(), tf.device('/cpu:0'):
			parameter_list, scalar_list = self._make_placeholders(parameters, scalars)

			for var in parameter_list:
				tf.summary.histogram(var.op.name, var)

			for var in scalar_list:
				tf.summary.scalar(var.op.name, var)

			with tf.control_dependencies(parameter_list + scalar_list):
				self.no_op = tf.no_op()

			self.summary_op = tf.summary.merge_all()
			self.sess = tf.Session()
			self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

	def _make_placeholders(self, parameters, scalars):
		parameter_list = []
		scalar_list = []

		name_dict = defaultdict(list)

		for name in parameters:
			idx = name.find(".")
			scope = name[:idx]
			var_name = name[idx + 1:]

			name_dict[scope].append(name)

		for scope in name_dict:
			with tf.variable_scope(scope):
				for name in name_dict[scope]:
					idx = name.find(self.delimiter)
					scope = name[:idx]
					var_name = name[idx + 1:]

					var = tf.placeholder(tf.float32, shape = parameters[name], name = var_name)
					parameter_list.append(var)
					self.pplaceholders[name] = var


		for name, shape in scalars.items():
			var = tf.placeholder(tf.float32, shape = shape, name = name)
			scalar_list.append(var)
			self.splaceholders[name] = var

		return parameter_list, scalar_list

	def _make_feed_dict(self, parameters, scalars):
		assert len(parameters) == len(self.pplaceholders) and len(scalars) == len(self.splaceholders)

		feed_dict = {}
		for name, holder in self.pplaceholders.items():
			try:
				feed_dict[holder] = parameters[name]
			except KeyError:
				print "Please provide a value dict with the same parameters you have provided"
				return None

		for name, holder in self.splaceholders.items():
			try:
				feed_dict[holder] = scalars[name]
			except KeyError:
				print "Please provide a value dict with the same scalars you have provided"
				return None

		return feed_dict

	def update(self, parameters, scalars, step):
		feed_dict = self._make_feed_dict(parameters, scalars)
		if feed_dict is None:
			print "Please provide a value dict with the same parameters you have provided"
			return

		with tf.Graph().as_default(), tf.device('/cpu:0'):
			self.sess.run(self.no_op, feed_dict = feed_dict)
			summary_str = self.sess.run(self.summary_op, feed_dict = feed_dict)
			self.summary_writer.add_summary(summary_str, step)





		


