import math 
import numpy as np
import random

class Experiment(object):
	"""docstring for Experiment"""
	def __init__(self, R, eta, logger, params = None):
		super(Experiment, self).__init__()
		self.R = R
		self.eta = eta
		self.logger = logger

		self.s_max = int(math.log(R, eta))
		self.B = (self.s_max + 1) * R
		self.params = params
		print "Experiment is initialize with R = {}, eta = {}, s_max = {}, B = {}".format(self.R, self.eta, self.s_max, self.B)

	def run(self, run_and_eval):
		if self.params is None:
			print "set hyperparameter before run."

		best = [float('inf'), None]

		for s in xrange(self.s_max, -1, -1):
			discount = self.eta ** s
			n = int(math.ceil(self.B * discount / float(self.R * (s+1))))

			r = self.R / float(discount)
			self.logger.info("Begin successive halfing with n = {}, r= {}".format(n, r))
			param_list = self.get_params(n)
			loss = []

			for i in xrange(s):
				n_i = int(n / self.eta ** i)
				r_i = r * self.eta ** i
				for param in param_list:
					l = run_and_eval(param, r_i)
					loss.append(l)
					if l < best[0]:
						best[0] = l
						best[1] = param
					self.log(param, l, "Logging param with budget {}".format(r_i))

				param_list = self.top_k(param_list, loss, int(n_i / self.eta))

		self.log(best[1], best[0], "Logging best parameter")
		return best

	def log(self, param, loss, message):
		self.logger.info(message)
		for k, v in param.items():
			self.logger.info("{}: {}".format(k, v))
		self.logger.info("The validation loss is {}".format(loss))

	def top_k(self, param_list, losses, k):
		sorted_list = sorted(zip(param_list, losses), key = lambda x: x[1])
		top = map(lambda x: x[0], sorted_list[:k])
		return top


	def set_parames(self, params):
		self.params = params

	def get_params(self, n):
		result = []
		for _ in xrange(n):
			params = {}
			for k, v in self.params.items():
				if "choice" in v:
					params[k] = random.choice(v['choice'])
				elif 'min' in v and 'max' in v:
					if type(v['min']) is float:
						params[k] = random.uniform(v['min'], v['max'])
					elif type(v['min']) is int:
						params[k] = np.random.randint(v['min'], v['max'])
					else:
						print "Type of min and max is neither float nor int"
				else:
					print "You provide a keyword that is not in implementation"
			result.append(params)

		return result

		