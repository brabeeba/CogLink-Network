import numpy as np
import logging

logger = logging.getLogger('Bandit')

class DynamicBandit(object):
	"""docstring for DynamicBandit"""
	def __init__(self, opt, name = "DynamicBandit"):
		super(DynamicBandit, self).__init__()
		self.opt = opt
		self.trial = 0
		self.context = 0
		self.context_num = opt["context_num"]
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.stimuli = 0
		self.max_trial = opt["max_trial"]
		self.name = name
		
		self.prob = np.random.rand(self.context_num, self.stimuli_num, self.class_num)
		#self.prob = np.array([[[0.3, 0.4, 0.5, 0.6, 0.7]]])
		#self.prob = np.array([[[0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]]])
		#self.prob = np.array([[[0.3, 0.7]]])
		self.prob = np.array([[[0.7, 0.3]], [[0.3, 0.7]]])
		#self.prob = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]])
	
	def setprob(self, prob):
		self.prob = prob
		for x in range(0, self.context_num):
			print("The bandit task starts with prob {} in context {}".format(self.prob[x], x))


	def stimuli(self):
		return self.stimuli

	def expectation(self, action):
		return self.prob[self.context][self.stimuli][action]

	def step(self, action):
		last = False
		reward = np.random.binomial(1, self.prob[self.context][self.stimuli][action])
		regret = np.max(self.prob[self.context][self.stimuli]) - self.prob[self.context][self.stimuli][action]

		self.trial += 1
		
		
		if self.trial % self.opt["block_size"] == 0:
			if self.context_num > 1:
				context = np.random.randint(0, self.context_num - 1)
				if context < self.context:
					self.context = context
				else:
					self.context = context + 1

		if self.trial >= self.opt["max_trial"]:
			logger.info("Reach the last trial at trial {}".format(self.opt["max_trial"]))
			last = True
		self.stimuli = np.random.randint(0, self.stimuli_num)

		return reward, regret, last, self.stimuli

	def reset(self):
		self.trial = 0
		self.context = 0
		self.stimuli = 0
		logging.info("Reset the task")


class SimpleMDP(object):
	"""docstring for SimpleMDP"""
	def __init__(self, opt, name = "SimpleMDP"):
		super(SimpleMDP, self).__init__()
		self.opt = opt
		self.trial = 0
		self.context = 0
		self.context_num = opt["context_num"]
		self.class_num = opt["class_num"]
		self.max_trial = opt["max_trial"]
		self.name = name
		self.init_param()


	def init_param(self):
		self.prob1 = np.array([[1/3, 2/3]])
		self.prob2 = np.array([[1, 2/3, 1/3, 0]])
		self.stimuli = 0

	def sum_ev(self):
		result = []
		for i in range(self.context_num):
			temp = [self.prob1[i][0] + self.prob2[i][0], self.prob1[i][0] + self.prob2[i][1], self.prob1[i][1] + self.prob2[i][2], self.prob1[i][1] + self.prob2[i][3] ]
			result.append(temp)
		return result


	def stimuli(self):
		return self.stimuli

	def expectation(self, action):
		return self.prob[self.context].dot(self.stimuli)[action]

	def step(self, action):
		last = False
		self.action = action

		
		if self.stimuli == 0:
			self.reward1 = np.random.binomial(1, self.prob1[self.context][action])
		else:
			self.reward2 = np.random.binomial(1, self.prob2[self.context][2 * (self.stimuli - 1) + action])
		
		

		if self.stimuli != 0:
			regret = np.max(self.sum_ev()[self.context]) - (self.reward1 + self.reward2)
			self.best_action = np.argmax(self.sum_ev()[self.context])
			self.trial += 1


		if self.trial % self.opt["block_size"] == 0:
			if self.context_num > 1:
				context = np.random.randint(0, self.context_num - 1)
				if context < self.context:
					self.context = context
				else:
					self.context = context + 1

		if self.trial >= self.opt["max_trial"]:
			logger.info("Reach the last trial at trial {}".format(self.opt["max_trial"]))
			last = True

		if self.stimuli == 0:
			self.stimuli =  self.action + 1
			self.end = False
			return self.reward1, last, self.end, self.stimuli
		else:
			self.stimuli = 0
			self.end = True
			return self.reward2, last, self.end, self.stimuli
		

	def reset(self):
		self.trial = 0
		self.context = 0
		self.init_param()

		logging.info("Reset the task")


class UnstructuredBandit(object):
	"""docstring for DynamicBandit"""
	def __init__(self, opt, name = "UnstructuredBandit"):
		super(UnstructuredBandit, self).__init__()
		self.opt = opt
		self.trial = 0
		self.context = 0
		self.context_num = 10
		self.class_num = opt["class_num"]
		self.stimuli_num = opt["stimuli_num"]
		self.stimuli = 0
		self.max_trial = opt["max_trial"]
		self.name = name
		
		self.prob = np.random.rand(self.context_num, self.stimuli_num, self.opt["class_num"])
		#self.prob[:, :, 1] = 1 - self.prob[:, :, 0]
		#self.prob = np.array([[[0.3, 0.4, 0.5, 0.6, 0.7]]])
		#self.prob = np.array([[[0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]]])
		#self.prob = np.array([[[0.3, 0.7]]])
		#self.prob = np.array([[[0.7, 0.3]], [[0.3, 0.7]]])
		#self.prob = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]])
		for x in range(0, self.context_num):
			print("The bandit task starts with prob {} in context {}".format(self.prob[x], x))

	def stimuli(self):
		return self.stimuli

	def expectation(self, action):
		return self.prob[self.context][self.stimuli][action]

	def step(self, action):
		last = False
		reward = np.random.binomial(1, self.prob[self.context][self.stimuli][action])
		self.trial += 1
		self.stimuli = np.random.randint(0, self.stimuli_num)
		regret = np.max(self.prob[self.context][self.stimuli]) - self.prob[self.context][self.stimuli][action]

		if self.trial % self.opt["block_size"] == 0:
			self.context = self.context + 1

		if self.trial >= self.opt["max_trial"]:
			logger.info("Reach the last trial at trial {}".format(self.opt["max_trial"]))
			last = True

		return reward, regret, last, self.stimuli

	def reset(self):
		self.trial = 0
		self.context = 0
		self.stimuli = 0
		logging.info("Reset the task")

class VolatileBandit(object):
	"""docstring for DynamicBandit"""
	def __init__(self, opt, name = "VolatileBandit"):
		super(VolatileBandit, self).__init__()
		self.opt = opt
		self.trial = 0
		self.context = 0
		self.context_num = opt["context_num"]
		self.stimuli_num = opt["stimuli_num"]
		self.stimuli = 0
		self.max_trial = opt["max_trial"]
		self.name = name
		
		self.prob = np.random.rand(self.context_num, self.stimuli_num, self.opt["class_num"])
		self.prob = np.array([[[0.7, 0.3]], [[0.3, 0.7]]])
		#self.prob = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]])
		for x in range(0, self.context_num):
			print("The bandit task starts with prob {} in context {}".format(self.prob[x], x))

	def stimuli(self):
		return self.stimuli

	def expectation(self, action):
		return self.prob[self.context][self.stimuli][action]

	def step(self, action):
		last = False
		reward = np.random.binomial(1, self.prob[self.context][self.stimuli][action])
		self.trial += 1
		self.stimuli = np.random.randint(0, self.stimuli_num)
		regret = np.max(self.prob[self.context][self.stimuli]) - self.prob[self.context][self.stimuli][action]

		condition1 = (self.trial >= self.max_trial / 2) and (self.trial % self.opt["block_size"] == 0)
		
		if condition1:
		
			if self.context_num > 1:
				context = np.random.randint(0, self.context_num - 1)
				if context < self.context:
					self.context = context
				else:
					self.context = context + 1

		if self.trial >= self.opt["max_trial"]:
			logger.info("Reach the last trial at trial {}".format(self.opt["max_trial"]))
			last = True

		return reward, regret, last, self.stimuli

	def reset(self):
		self.trial = 0
		self.context = 0
		self.stimuli = 0
		logging.info("Reset the task")


class HMMBandit(object):
	"""docstring for HMMBandit"""
	def __init__(self, opt):
		super(HMMBandit, self).__init__()
		self.opt = opt
		self.trial = 0
		self.context = 0
		self.context_num = opt["context_num"]
		self.stimuli_num = opt["stimuli_num"]
		self.stimuli = 0
		self.max_trial = opt["max_trial"]
		
		self.prob = np.random.rand(self.context_num, self.stimuli_num, self.opt["class_num"])
		self.prob = np.array([[[0.7, 0.3]], [[0.3, 0.7]]])
		self.switch_prob = opt["switch_prob"] 

		for x in range(0, self.context_num):
			print("The bandit task starts with prob {} in context {}".format(self.prob[x], x))

	def stimuli(self):
		return self.stimuli

	def expectation(self, action):
		return self.prob[self.context][self.stimuli][action]

	def step(self, action):
		last = False
		reward = np.random.binomial(1, self.prob[self.context][self.stimuli][action])
		self.trial += 1
		self.stimuli = np.random.randint(0, self.stimuli_num)
		regret = np.max(self.prob[self.context][self.stimuli]) - self.prob[self.context][self.stimuli][action]

		if np.random.rand() < self.switch_prob:
			print("switch at trial {}".format(self.trial))
			if self.context_num > 1:
				context = np.random.randint(0, self.context_num - 1)
				if context < self.context:
					self.context = context
				else:
					self.context = context + 1

		if self.trial >= self.opt["max_trial"]:
			logger.info("Reach the last trial at trial {}".format(self.opt["max_trial"]))
			last = True

		return reward, regret, last, self.stimuli

	def reset(self):
		self.trial = 0
		self.context = 0
		self.stimuli = 0
		logging.info("Reset the task")


