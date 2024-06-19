import numpy as np
import logging
from layer import *
import scipy.special as special
from util import *
from scipy.special import expit
from scipy.stats import norm

logger = logging.getLogger('Bandit')

class ThompsonDCAgent(object):
	"""docstring for ThompsonDCAgent"""
	def __init__(self, opt, name = "Bayesian RL"):
		super(ThompsonDCAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.gamma = opt["gamma"]
		self.prob = np.zeros( (opt["stimuli_num"], opt["class_num"], 2))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name

		self.past_choice = [0]


	def ev(self):
		return (self.prob[:, :, 1] + 1) / (2 + np.sum(self.prob[:, : ,:], axis = 2))

	def get_ev(self):
		return self.ev()

	def get_choice_prob(self):
		return sum(self.past_choice) / len(self.past_choice)

	def scalars(self):
		hist = {}
		ev = self.ev()
		
		for s in range(self.stimuli_num):
			for a in range(self.class_num):
				hist["fast-reward/simuli-{}/action-{}".format(s, a)] = ev[s, a]
		hist["choice_prob"] = self.get_choice_prob()
		return hist
	def histogram(self):
		return {}
	def forward(self, x):
		self.stimuli = x
		sample = np.random.beta(  self.prob[x, :, 1] + 1,   self.prob[x, :, 0] + 1)
		action = np.argmax(sample)

		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action
		return action
		
	def update(self, r):
		self.prob[self.stimuli, self.action, :] *= self.gamma
		self.prob[self.stimuli, self.action, r] += 1
	



	def reset(self):
		self.prob = np.zeros( (self.opt["stimuli_num"], self.opt["class_num"], 2))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.past_choice = [0]




		

class TwoTimeScaleAgent(object):
	"""docstring for TwoTimeScaleAgent"""
	def __init__(self, opt, name = ""):
		super(TwoTimeScaleAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.context_num = opt["context_num"]
		self.history = opt["history"]
		self.gamma1 = opt["gamma1"]
		self.gamma2 = opt["gamma2"]
		self.temperature = opt["temperature"]
		self.lr = opt["lr"]

		self.prob = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2))
		self.context = 0
		self.action = 0
		self.stimuli = 0

		self.R = 0
		self.r = 0

		self.state = np.zeros(self.context_num)
		self.state[0] = self.history / 2
		self.time = 0
		self.confidence = 0

		self.past_choice = [0]


	def ev(self):
		return (self.prob[:, :, :, 0] + 1) / (2 + np.sum(self.prob[:, :, : ,:], axis = 3))

	def weights(self):
		return np.array([(self.state[0] - self.state[1] + self.history) / (2 * self.history), (self.history + self.state[1] - self.state[0]) / (2 * self.history)])


	def histogram(self):
		return {}

	def scalars(self):
		hist = {}
		ev = self.ev()
		for c in range(self.context_num):
			for s in range(self.stimuli_num):
				for a in range(self.class_num):
					hist["fast-reward/context-{}/simuli-{}/action-{}".format(c, s, a)] = ev[c, s, a]
					hist["negative-log-likelihood/context-{}".format(c)] = self.state[c]

		hist["choice_prob"] = sum(self.past_choice) / len(self.past_choice)
		hist["fast-reward"] = self.r
		hist["context-difference"] = self.state[0] - self.state[1]
		hist["confidence"] = self.confidence
		return hist

	def forward(self, x):
		self.time += 1
		self.stimuli = x

		# print(self.prob[:, x, :, 0] + 1)
		# print(self.weights())
		# print((self.prob[:, x, :, 0] + 1).T.dot(self.weights()))

		if self.state[0] - self.state[1] >= 0:
			self.context = 0
		else:
			self.context = 1
		confidence =  np.abs(self.state[0] - self.state[1]) / self.history
		f = lambda x: 2 * x
		self.confidence = f(confidence)

		sample = np.random.beta((self.prob[self.context, x, :, 0]) * self.confidence + 1, (self.prob[self.context, x, :, 1]) * self.confidence  + 1 )
		#sample = np.random.beta((self.prob[:, x, :, 0] + 1).T.dot(self.weights()), (self.prob[:, x, :, 1] + 1).T.dot(self.weights()))
		self.action = np.argmax(sample)

		self.past_choice.append(self.action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]

		return self.action

	def update(self, r):
		self.r = r
		

		self.R *= self.gamma2
		self.R += r

		


		# if self.context == 0:
		# 	confidence = (self.state[0] - self.state[1] + self.history) / (2 * self.history)
		# if self.context == 1:
		# 	confidence = (self.state[1] - self.state[0] + self.history) / (2 * self.history)
		# f = lambda x: x**1.5
		# confidence = f(confidence)
		confidence =  np.abs(self.state[0] - self.state[1]) / self.history
		self.confidence = confidence


		self.prob[self.context, self.stimuli, self.action, :] *=  self.gamma1
		if r == 1:
			self.prob[self.context, self.stimuli, self.action, 0] += 1 * self.confidence
		elif r == 0:
			self.prob[self.context, self.stimuli, self.action, 1] += 1 * self.confidence

		self.state *= self.gamma2
		if r == 1:
			self.state +=  0.9 * relu(3+ np.log((self.prob[:, self.stimuli, self.action, 0]+1) / (2+np.sum(self.prob[:, self.stimuli, self.action, :], axis = 1))))		
		elif r == 0:
			self.state += 0.9 * relu(3 + np.log((self.prob[:, self.stimuli, self.action, 1]+1) / (2+np.sum(self.prob[:, self.stimuli, self.action, :], axis = 1))))

		if self.state[0] - self.state[1] >= self.history:
			self.state[0] =  self.history
			self.state[1] = 0
		elif self.state[1] - self.state[0] >= self.history:
			self.state[1] = self.history
			self.state[0] = 0
		
		if self.state[0] - self.state[1] >= 0:
			self.context = 0
		else:
			self.context = 1

class TwoTimeScaleNeuralAgent(object):
	"""docstring for TwoTimeScaleAgent"""
	def __init__(self, opt, name = "Thalamocortical Model"):
		super(TwoTimeScaleNeuralAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.context_num = opt["context_num"]
		self.gamma1 = opt["gamma1"]
		self.tau = opt["tau"]
		self.s = 1.0 / (2 * self.tau)
	
		self.temperature = opt["temperature"]
		self.lr = opt["lr"]
		self.a = opt["a"]
		self.name = name

		self.learning = opt["learning"]
		self.nonlinear = opt["nonlinear"]
		
		self.dt = opt["dt"]
		self.a1 = opt["a1"]
		self.b1 = opt["b1"]
		self.a2 = opt["a2"]
		self.b2 = opt["b2"]

		self.tau = opt["tau"]
		self.eta = opt["eta"]
		self.tau1 = opt["tau1"]
		self.threshold = opt["threshold"]
		self.d_interval = opt["d_interval"]
		self.K = opt["K"]

		self.inhibit = opt["inhibit"]
		self.d2 = opt["d2"]
		self.rescue = opt["rescue"]
	






		self.stimuli = 0

		self.thalamus = np.zeros(self.context_num)
		self.thalamus[0] = 2 * self.a * self.tau + 4
		self.thalamus[1] = 4

		self.thalamus[0] = 2 * self.a+ 2
		self.thalamus[1] = 2


		if self.d2:
			self.thalamus[0] = 8

		self.vip = np.zeros(self.context_num)
		self.pv = np.zeros(self.context_num)

		self.ct = 0.5 * np.ones((self.context_num, self.stimuli_num, self.class_num, 2))

		self.tt = -  self.s * np.ones((self.context_num, self.context_num))
		for i in range(0, self.context_num):
			self.tt[i, i] = self.s 


		self.quantile_num = opt["quantile_num"]
		
		self.conf = 0

		self.prob = np.ones((self.context_num, self.stimuli_num, self.class_num, self.quantile_num))
		self.prob *= np.expand_dims((np.arange(self.quantile_num)+1) / float(self.quantile_num), (0, 1, 2))


		self.sample_neurons = [np.zeros((self.context_num, self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]



		
	
		self.sample_w = -  self.b1 * np.ones((self.quantile_num, self.quantile_num))
		for i in range(self.quantile_num):
			self.sample_w[i, i] = self.a1

		self.value_neurons = np.zeros((self.context_num, self.quantile_num, self.class_num))

		

		self.decision_neurons = np.zeros(self.class_num)
	
		self.decision_w = -  self.b2 * np.ones((self.class_num, self.class_num))
		for i in range(self.class_num):
			self.decision_w[i, i] = self.a2

		



		self.count = np.zeros( (opt["context_num"], opt["stimuli_num"], opt["class_num"]))
		self.count1 = np.zeros((opt["context_num"], opt["stimuli_num"], opt["class_num"]))
		self.stimuli = 0
		self.action = 0
		self.context = 0



		self.R = 0
		self.r = 0

		self.time = 0
		self.confidence = 0

		self.past_choice = [0]
		self.lr = 0


	def f_scalar(self, x):
		if x >= self.a:
			return 2 * self.a
		elif x <= -self.a:
			return 0
		else:
			return x + self.a

	def f(self, x):
		return np.vectorize(self.f_scalar)(x)
		

	def g(self, x):
		return relu(2.7 + np.log(x))
		#return relu(3 + np.log(x))

	def h(self, x):
		return np.minimum(1, relu(x))

	def sig(self, x):
		return 1 / (1 + np.exp(-10 * x + 5))

	#1 / (1 + np.exp(-10 * x + 5)) 

	def md_f(self, x):
		# return relu(2 /  (1 + np.exp(4*self.a - 4*(x-2)))- 1)
		return relu(2 /  (1 + np.exp(4*self.a*self.tau - 4*(x-4)))- 1)

	def in_f(self, x):
		return relu(2 /  (1 + np.exp( - 2*(x +0.25)))-1)



	def ev(self):
		return np.mean(self.prob, axis = 3)

	def get_ev(self):
		return np.mean(self.prob, axis = 3)[self.context, :, :]

	def get_choice_prob(self):
		return sum(self.past_choice) / len(self.past_choice)

	def histogram(self):
		return {"PFC": self.pfc(), "PFC/MD": self.ct.copy(), "MD": self.thalamus.copy(), "ALM": self.sample_neurons.copy(), "ALM/BG": self.prob.copy(), "BG": self.value_neurons.copy(), "M1": self.decision_neurons.copy(), "PV": self.pv.copy(), "VIP": self.vip.copy()}

	def pfc(self):
		pfc = np.zeros((self.stimuli_num, self.class_num, 2))
		pfc[self.stimuli, self.action, self.r] = 1
		return pfc

	def scalars(self):
		hist = {}
		ev = self.ev()
		for c in range(self.context_num):
			for s in range(self.stimuli_num):
				for a in range(self.class_num):
					hist["fast-reward/context-{}/simuli-{}/action-{}".format(c, s, a)] = ev[c, s, a]
					for r in range(2):
						hist["likelihood/context-{}/simuli-{}/action-{}/reward-{}".format(c, s, a, r)] = self.ct[c, s, a, r]
			hist["context-{}".format(c)] = self.thalamus[c]

			
		
		hist["fast-reward"] = self.r
		hist["confidence"] = self.confidence 
		hist["context-difference"] = self.thalamus[0] - self.thalamus[1]
		hist["choice_prob"] = self.get_choice_prob()
		hist["smooth_confidence"] = self.conf
		hist["learning_rate"] = self.lr
		hist["md-nonlinearity"] = self.md_f(self.thalamus[0])
		hist["in-nonlinearity"] = self.in_f(self.vip[0]-self.pv[0])
		hist["action"] = self.action
		hist["context"] = self.context

		return hist

	def forward(self, x):
		self.time += 1
		self.stimuli = x

		

		self.sample_neurons = [np.zeros((self.context_num, self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
		self.value_neurons = np.zeros((self.context_num, self.quantile_num, self.class_num))
		self.decision_neurons = np.zeros(self.class_num)

		action_bool = False



		for i in range(self.d_interval):

			if self.thalamus[0] - self.thalamus[1] >= 0:
				self.context = 0
			else:
				self.context = 1


			self.vip += self.dt * self.tau1 * (-self.vip + self.thalamus)
			self.pv += self.dt * self.tau1 * (-self.pv + np.flip(self.thalamus))

			self.confidence = self.sig(np.abs(self.thalamus[0] - self.thalamus[1]) / (2 * self.a * self.tau))
			self.conf += self.dt * self.tau1 * (-self.conf + self.confidence)


			if self.d2:
				gain = 0.85
			else:
				gain = 1
			
			if self.rescue:
				rescue = 0.45
			else:
				rescue = 0

			if self.nonlinear:
				self.thalamus += self.dt * 0.2 * (-1.0 / self.tau * self.thalamus +  gain * self.f(self.tt.dot(self.thalamus)) +  gain * self.g(self.ct[:, self.stimuli, self.action, self.r]) + rescue)
			else:
				self.thalamus += self.dt * 0.2 * (-1.0 / self.tau * self.thalamus + relu(self.tt.dot(self.thalamus)) + self.g(self.ct[:, self.stimuli, self.action, self.r]))

			if self.inhibit:
				self.thalamus = np.zeros(self.context_num)
			
			if self.learning:

				for k in range(self.context_num):


				
					self.sample_neurons[x][k] +=  self.dt * 6 * (-self.sample_neurons[x][k] +  self.sample_w.dot(self.h(self.sample_neurons[x][k])) +  (self.K-0.25)*self.b1 + 0.2 * np.random.normal(size = (self.quantile_num, self.class_num)))
					# if k == self.context:
					# 	self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k] +  relu(self.h(self.sample_neurons[x][k]) * self.prob[k, x, :, :].T * self.conf ))
					# else:
					# 	self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k])
					self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k] +  relu(self.h(self.sample_neurons[x][k]) * self.prob[k, x, :, :].T * self.in_f(self.vip[k] - self.pv[k])))
				self.decision_neurons += self.dt * 0.1 * self.tau1 * (-self.decision_neurons + relu(self.decision_w.dot(self.decision_neurons)) + 1.0 / self.K * np.sum(self.value_neurons, axis = (0, 1)) )
			
			else:
				for k in range(self.context_num):
				
					self.sample_neurons[x][k] +=  0.03 * (-self.sample_neurons[x][k] +  self.sample_w.dot(self.h(self.sample_neurons[x][k])) +  (self.K-0.25)*self.b1 + 0.2 * np.random.normal(size = (self.quantile_num, self.class_num)))
					if k == self.context:
						self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k] +  relu(self.h(self.sample_neurons[x][k]) * self.prob[k, x, :, :].T ))
					else:
						self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k])
				
				self.decision_neurons += self.dt * 0.1 * self.tau1 * (-self.decision_neurons + relu(self.decision_w.dot(self.decision_neurons)) + 1.0 / self.K * np.sum(self.value_neurons, axis = (0, 1)) )
			


			if np.max(self.decision_neurons) > self.threshold:
				action = np.argmax(self.decision_neurons)
				action_bool = True
				



		if not action_bool:
			
			#action = np.random.randint(self.class_num)
			#action = np.argmax(self.decision_neurons)
			action = np.random.choice(self.class_num, p= special.softmax(30 * self.decision_neurons))
			
			# print("Random decision is made {}".format(self.name))
			# print(self.decision_neurons, self.threshold, self.context)
			# print(np.sort(self.value_neurons[self.context, :, 0])[-4:])
			

		if self.learning:
			#self.count[self.context, x, action] += self.conf
			self.count[:, x, action] +=  0.5 * self.in_f(self.vip-self.pv)
		else:
			self.count[self.context, x, action] += 1

		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action


		if self.time % 200 == 0:
			print(self.time)



		return self.action

	def update(self, r):
		self.r = r

		# self.thalamus += -1.0 / self.tau * self.thalamus + self.f(self.tt.dot(self.thalamus)) + self.g(self.ct[:, self.stimuli, self.action, r])
		if self.thalamus[0] - self.thalamus[1] >= 0:
			self.context = 0
		else:
			self.context = 1
		self.confidence = self.sig(np.abs(self.thalamus[0] - self.thalamus[1]) / (2 * self.a * self.tau))
	

		if self.learning:
			#self.prob[self.context, self.stimuli, self.action, :-1] += (r - self.prob[self.context, self.stimuli, self.action, :-1]) * self.conf / (3+self.count[self.context, self.stimuli, self.action])
			self.prob[:, self.stimuli, self.action, :-1] += (r - self.prob[:, self.stimuli, self.action, :-1]) * np.expand_dims(self.in_f(self.vip-self.pv), 1) / (7+np.expand_dims(self.count[:, self.stimuli, self.action], 1))
			
			#self.count1[self.context, self.stimuli, self.action] += self.confidence
			self.count1[:, self.stimuli, self.action] += self.md_f(self.thalamus)

			inputs = np.zeros(2)
			inputs[r] = 1
			#self.ct[self.context, self.stimuli, self.action, r] +=   self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])
			#self.lr = self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])

			self.ct[:, self.stimuli, self.action, r] +=   self.md_f(self.thalamus) / (4+ self.count1[:, self.stimuli, self.action])
			self.lr = self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])
			
			self.ct[:, self.stimuli, self.action, :] = self.ct[:, self.stimuli, self.action, :] / (np.sum(self.ct[:, self.stimuli, self.action, :], axis = 1, keepdims = True)+1e-8 ) 

		else:
			self.prob[self.context, self.stimuli, self.action, :-1] += (r - self.prob[self.context, self.stimuli, self.action, :-1]) / (2+self.count[self.context, self.stimuli, self.action])
		
			self.count1[self.context, self.stimuli, self.action] += 1

			inputs = np.zeros(2)
			inputs[r] = 1
			self.ct[self.context, self.stimuli, self.action, r] +=   1 / (2+ self.count1[self.context, self.stimuli, self.action])
			self.lr = 1 / (2+ self.count1[self.context, self.stimuli, self.action])
			
			self.ct[:, self.stimuli, self.action, :] = self.ct[:, self.stimuli, self.action, :] / (np.sum(self.ct[:, self.stimuli, self.action, :], axis = 1, keepdims = True)+1e-8 ) 

		

		# if self.learning:
		# 	

		# else:
		# 	self.count1[self.context, self.stim*uli, self.action] += 1

		# inputs = np.zeros(2)
		# inputs[r] = 1
		
		# if self.learning:
		# 	self.ct[self.context, self.stimuli, self.action, :] +=  0.8 * self.confidence / (5+ self.count1[self.context, self.stimuli, self.action]) * (inputs  - self.ct[self.context, self.stimuli, self.action, :]) * self.ct[self.context, self.stimuli, self.action, r]
		# else:
		# 	self.ct[self.context, self.stimuli, self.action, :] +=  1 / (1+self.count1[self.context, self.stimuli, self.action]) * (inputs * self.ct[self.context, self.stimuli, self.action, r] - self.ct[self.context, self.stimuli, self.action, r] * self.ct[self.context, self.stimuli, self.action, :])


	def reset(self):
		self.thalamus = np.zeros(self.context_num)
		self.thalamus[0] = 2 * self.a * self.tau
		self.ct = 0.8 * np.ones((self.context_num, self.stimuli_num, self.class_num, 2))

		# self.prob = np.ones(( self.context_num, self.stimuli_num, self.class_num, self.quantile_num))
		
		self.sample_neurons = [np.zeros((self.context_num, self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
		self.value_neurons = np.zeros((self.context_num, self.quantile_num, self.class_num))
		self.decision_neurons = np.zeros(self.class_num)
		
		self.prob = np.ones((self.context_num, self.stimuli_num, self.class_num, self.quantile_num))
		self.prob *= np.expand_dims((np.arange(self.quantile_num)+1) / float(self.quantile_num), (0, 1, 2))



		self.count = np.zeros( (self.context_num, self.stimuli_num, self.class_num))
		self.count1 = np.zeros((self.context_num, self.stimuli_num, self.class_num))
		self.stimuli = 0
		self.action = 0
		self.context = 0

		self.R = 0
		self.r = 0
		self.time = 0
		self.confidence = 0
		self.past_choice = [0]



# class TwoTimeScaleNeuralAgent(object):
# 	"""docstring for TwoTimeScaleAgent"""
# 	def __init__(self, opt, name = "Thalamocortical Model"):
# 		super(TwoTimeScaleNeuralAgent, self).__init__()
# 		self.opt = opt
# 		self.stimuli_num = opt["stimuli_num"]
# 		self.class_num = opt["class_num"]
# 		self.context_num = opt["context_num"]
# 		self.gamma1 = opt["gamma1"]
# 		self.tau = opt["tau"]
# 		self.s = 1.0 / (2 * self.tau)
# 		self.temperature = opt["temperature"]
# 		self.lr = opt["lr"]
# 		self.a = opt["a"]
# 		self.name = name

# 		self.learning = opt["learning"]
# 		self.modulate = opt["modulate"]


# 		self.stimuli = 0

# 		self.thalamus = np.zeros(self.context_num)
# 		self.thalamus[0] = 2 * self.a * self.tau

# 		self.ct = 0.8 * np.ones((self.context_num, self.stimuli_num, self.class_num, 2))

# 		self.tt = -  self.s * np.ones((self.context_num, self.context_num))
# 		for i in range(0, self.context_num):
# 			self.tt[i, i] = self.s 

# 		self.prob = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2))
# 		self.context = 0
# 		self.action = 0
		

# 		self.R = 0
# 		self.r = 0

# 		self.time = 0
# 		self.confidence = 0

# 		self.past_choice = [0]

# 	def f_scalar(self, x):
# 		if x >= self.a:
# 			return 2 * self.a
# 		elif x <= -self.a:
# 			return 0
# 		else:
# 			return x + self.a

# 	def f(self, x):
# 		return np.vectorize(self.f_scalar)(x)
		

# 	def g(self, x):
# 		return 0.9 * relu(3 + np.log(x))



# 	def ev(self):
# 		return (self.prob[:, :, :, 1] + 1) / (2 + np.sum(self.prob[:, :, : ,:], axis = 3))

# 	def get_ev(self):
# 		return (self.prob[self.context, :, :, 1] + 1) / (2 + np.sum(self.prob[self.context, :, : ,:], axis = 2))

# 	def get_choice_prob(self):
# 		return sum(self.past_choice) / len(self.past_choice)

# 	def histogram(self):
# 		return {}

# 	def scalars(self):
# 		hist = {}
# 		ev = self.ev()
# 		for c in range(self.context_num):
# 			for s in range(self.stimuli_num):
# 				for a in range(self.class_num):
# 					hist["fast-reward/context-{}/simuli-{}/action-{}".format(c, s, a)] = ev[c, s, a]
# 					for r in range(2):
# 						hist["likelihood/context-{}/simuli-{}/action-{}/reward-{}".format(c, s, a, r)] = self.ct[c, s, a, r]
# 			hist["context-{}".format(c)] = self.thalamus[c]

			
		
# 		hist["fast-reward"] = self.r
# 		hist["confidence"] = self.confidence 
# 		hist["context-difference"] = self.thalamus[0] - self.thalamus[1]
# 		hist["choice_prob"] = self.get_choice_prob()

# 		return hist

# 	def forward(self, x):
# 		self.time += 1
# 		self.stimuli = x

# 		if self.thalamus[0] - self.thalamus[1] >= 0:
# 			self.context = 0
# 		else:
# 			self.context = 1

# 		self.confidence = np.abs(self.thalamus[0] - self.thalamus[1]) / (2 * self.a * self.tau)
# 		#sample = np.random.beta((2 * self.prob[self.context, x, :, 0]) * self.confidence + 1, (2 * self.prob[self.context, x, :, 1]) * self.confidence  + 1 )
# 		if self.modulate:
# 			sample = np.random.beta((3 * self.prob[self.context, x, :, 1]) * self.confidence + 1, (3 * self.prob[self.context, x, :, 0]) * self.confidence  + 1 )
# 		else:
# 			sample = np.random.beta((3 * self.prob[self.context, x, :, 1]) + 1, (3 * self.prob[self.context, x, :, 0])  + 1 )


# 		self.action = np.argmax(sample)

# 		self.past_choice.append(self.action)
# 		if len(self.past_choice) > 10:
# 			del self.past_choice[0]


# 		return self.action

# 	def update(self, r):
# 		self.r = r
		

		
# 		self.prob[self.context, self.stimuli, self.action, :] *=  self.gamma1
		
# 		if self.learning:
# 			self.prob[self.context, self.stimuli, self.action, r] += 1 * self.confidence
# 		else:
# 			self.prob[self.context, self.stimuli, self.action, r] += 1


# 		#print(self.thalamus, self.tt.dot(self.f(self.thalamus)),self.ct[:, self.stimuli, self.action, 0] )

# 		#q = (self.prob[:, self.stimuli, self.action, 1-r]+1) / (2+np.sum(self.prob[:, self.stimuli, self.action, :], axis = 1))
# 		self.thalamus += -1.0 / self.tau * self.thalamus + self.f(self.tt.dot(self.thalamus)) + self.g(self.ct[:, self.stimuli, self.action, r])

# 		#self.gain * (self.tt.dot(self.thalamus) + relu( np.log(2 + self.ct[:, self.stimuli, self.action, 1-r])) )
# 		#print(self.thalamus)
# 		#self.thalamus = self.f(self.thalamus)
# 		#print(self.thalamus)
		
# 		if self.learning:
# 			self.ct[self.context, self.stimuli, self.action, r] +=  0.03 * np.abs(self.thalamus[0] - self.thalamus[1]) / (2 * self.a * self.tau)
# 		else:
# 			self.ct[self.context, self.stimuli, self.action, r] +=  0.03
		
# 		self.ct[:, self.stimuli, self.action, :] = self.ct[:, self.stimuli, self.action, :] / (np.sum(self.ct[:, self.stimuli, self.action, :], axis = 1, keepdims = True)+1e-8 ) 

# 	def reset(self):
# 		self.thalamus = np.zeros(self.context_num)
# 		self.thalamus[0] = 2 * self.a * self.tau
# 		self.ct = 0.8 * np.ones((self.context_num, self.stimuli_num, self.class_num, 2))
# 		self.prob = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2))
# 		self.context = 0
# 		self.action = 0
# 		self.R = 0
# 		self.r = 0
# 		self.time = 0
# 		self.confidence = 0
# 		self.past_choice = [0]




		
		

			
class HMMAgent(object):
	"""docstring for TwoTimeScaleAgent"""
	def __init__(self, opt, name = "HMM Model"):
		super(HMMAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.context_num = opt["context_num"]
		self.iter = opt["iter"]
		self.name = name

		self.s = 0.9

		self.prob = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2))
		self.prob[:, :, :,0] = np.random.rand(self.context_num, self.stimuli_num, self.class_num)
		self.prob[:, :, :, 1] = 1 - self.prob[:, :, :,0]
	
		self.tt = (1-self.s) / (self.context_num - 1) * np.ones((self.context_num, self.context_num))
		for i in range(0, self.context_num):
			self.tt[i, i] = self.s 



		self.context = 0
		self.action = 0
		self.stimuli = 0


		# self.state =  np.ones(self.context_num) / self.context_num
		# self.initial = np.ones(self.context_num) / self.context_num

		self.state = np.zeros(self.context_num)
		self.initial = np.zeros(self.context_num)
		self.initial[0] = self.s
		self.state[0] = self.s
		for i in range(1, self.context_num):
			self.initial[i] = (1 - self.s) / (self.context_num - 1)
			self.state[i] = (1 - self.s) / (self.context_num - 1)

		self.time = 0
		self.confidence = 0

		self.past_choice = [0]
		self.trajectory = []


	def ev(self):
		return self.prob[:, :, :, 1] 

	def get_ev(self):
		return np.sum(self.prob[:, :, :, 1] * np.expand_dims(self.state, axis = [1, 2]), axis = 0)

	def get_choice_prob(self):
		return sum(self.past_choice) / len(self.past_choice)


	def histogram(self):
		return {}

	def scalars(self):
		
		hist = {}
		ev = self.ev()
		for c in range(self.context_num):
			for s in range(self.stimuli_num):
				for a in range(self.class_num):
					hist["fast-reward/context-{}/simuli-{}/action-{}".format(c, s, a)] = ev[c, s, a]
					for r in range(2):
						hist["likelihood/context-{}/simuli-{}/action-{}/reward-{}".format(c, s, a, r)] = self.prob[c, s, a, r]
				
		hist["choice_prob"] = self.get_choice_prob()
		hist["fast-reward"] = self.r
		hist["context-difference"] = self.state[0] - self.state[1]
	
		return hist

	def forward_backward(self):
		T = len(self.trajectory)
		if T == 0:
			return
		self.alpha = np.zeros((T, self.context_num))
		self.beta = np.zeros((T, self.context_num))
		self.temp = np.zeros((T, self.context_num))
		
		for i, data in enumerate(self.trajectory):
			self.temp[i] = self.prob[:, data[0], data[1], data[2]]

			if i == 0:
				self.alpha[0] = np.log(self.initial + 1e-8) + np.log(self.prob[:, data[0], data[1], data[2]])
			else:

				broadcast = np.log(self.tt + 1e-8) + np.expand_dims(self.alpha[i-1], axis = 1)
				self.alpha[i] = np.log(self.prob[:, data[0], data[1], data[2]]) + special.logsumexp(broadcast, axis = 0)


		for i in range(T):
			if i == 0:
				pass
			else:
				data = self.trajectory[T-i]
				broadcast = np.log(self.tt + 1e-8) + np.expand_dims(self.beta[T-i], axis = 0) + np.expand_dims(np.log(self.prob[:, data[0], data[1], data[2]]), axis = 0)
				self.beta[T-1-i] = special.logsumexp(broadcast, axis = 1)


	def forward(self, x):
		self.time += 1
		self.stimuli = x

		T = len(self.trajectory)
		if T == 0:
			self.action = np.random.randint(self.class_num)
			self.past_choice.append(self.action)
			if len(self.past_choice) > 10:
				del self.past_choice[0]
			return self.action

		self.forward_backward()
		self.gamma = self.alpha + self.beta
		self.gamma = np.exp(self.gamma - special.logsumexp(self.gamma, axis = 1, keepdims = True))
		self.state = self.gamma[-1]

		if self.state[0] - self.state[1] >= 0:
			self.context = 0
		else:
			self.context = 1


		self.occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2)) + 1
		self.total_occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num)) + 2
		for i, data in enumerate(self.trajectory):
			self.occupancy[:, data[0], data[1], data[2]] += self.gamma[i]
			self.total_occupancy[:, data[0], data[1]] += self.gamma[i]

		sample = np.random.beta(self.occupancy[:, x, :, 1].T.dot(self.state) + 1, self.occupancy[:, x, :, 0].T.dot(self.state) + 1 )
		self.action = np.argmax(sample)

		self.past_choice.append(self.action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]

		return self.action

	def update(self, r):

		self.r = r 
		self.trajectory.append((self.stimuli, self.action, self.r))

		if self.time % 100 == 0:
			print(self.time)


		# if self.time % 5 != 0:
		# 	return
		# if self.time > 600:
		# 	if self.time % 10 != 0:
		# 		return


		for it in range(self.iter):
			self.forward_backward()
			T = len(self.trajectory)
			if T == 1:
				self.gamma = self.alpha + self.beta
				self.gamma = np.exp(self.gamma - special.logsumexp(self.gamma, axis = 1))
				self.occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2)) + 1
				self.total_occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num)) + 2
				for i, data in enumerate(self.trajectory):
					self.occupancy[:, data[0], data[1], data[2]] += self.gamma[i]
					self.total_occupancy[:, data[0], data[1]] += self.gamma[i]

				self.initial = self.gamma[0]
				self.prob = self.occupancy / np.expand_dims(self.total_occupancy,axis = 3)
				return

			self.gamma = self.alpha + self.beta

			self.gamma = np.exp(self.gamma - special.logsumexp(self.gamma, axis = 1, keepdims = True))

			self.eta = np.expand_dims(self.alpha[:-1], axis = 2) + np.expand_dims(np.log(self.tt + 1e-8), axis = 0) + np.expand_dims(self.beta[1:], axis = 1) + np.expand_dims(np.log(self.temp[1:]), axis = 1)
			
			self.eta = np.exp(self.eta - special.logsumexp(special.logsumexp(self.eta, axis = 2, keepdims = True), axis = 1, keepdims = True))

			new_state = self.gamma[-1]

			self.occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2)) + 1
			self.total_occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num)) + 2
			#self.prob = self.occupancy / np.expand_dims(self.total_occupancy,axis = 3)
	

			for i, data in enumerate(self.trajectory):
				self.occupancy[:, data[0], data[1], data[2]] += self.gamma[i] 
				self.total_occupancy[:, data[0], data[1]] += self.gamma[i] 

			new_initial = self.gamma[0]
			new_tt = np.sum(self.eta, axis = 0) / np.expand_dims(np.sum(self.gamma[:-1], axis = 0), axis = 1)

			new_prob = self.occupancy / np.expand_dims(self.total_occupancy,axis = 3)

			error =  np.linalg.norm(self.state - new_state) + np.linalg.norm(self.initial - new_initial) + np.linalg.norm(self.tt - new_tt) + np.linalg.norm(self.prob - new_prob) 
			
			if error< 1e-7:
				print(it, self.time)
				return

			self.initial = new_initial
			self.state  = new_state
			self.prob = new_prob
			self.tt = new_tt

		
	def reset(self):
		self.prob = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2))
		self.prob[:, :, :,0] = np.random.rand(self.context_num, self.stimuli_num, self.class_num)
		self.prob[:, :, :, 1] = 1 - self.prob[:, :, :,0]

		self.tt = (1-self.s) / (self.context_num - 1) * np.ones((self.context_num, self.context_num))
		for i in range(0, self.context_num):
			self.tt[i, i] = self.s 


		self.state = np.zeros(self.context_num)
		self.initial = np.zeros(self.context_num)
		self.initial[0] = self.s
		self.state[0] = self.s
		for i in range(1, self.context_num):
			self.initial[i] = (1 - self.s) / (self.context_num - 1)
			self.state[i] = (1 - self.s) / (self.context_num - 1)

		self.context = 0
		self.action = 0
		self.R = 0
		self.r = 0
		self.time = 0
		self.confidence = 0
		self.past_choice = [0]
		self.trajectory = []

class HMM(object):
	"""docstring for HMM"""
	def __init__(self, opt):
		super(HMM, self).__init__()
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.context_num = opt["context_num"]
		self.iter = opt["iter"]

		self.s = 0.9

		self.prob = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2))
		self.prob[:, :, :,0] = np.random.rand(self.context_num, self.stimuli_num, self.class_num)
		self.prob[:, :, :, 1] = 1 - self.prob[:, :, :,0]
	
		self.tt = (1-self.s) / (self.context_num - 1) * np.ones((self.context_num, self.context_num))
		for i in range(0, self.context_num):
			self.tt[i, i] = self.s 


		self.context = 0
		self.action = 0
		self.stimuli = 0


		# self.state =  np.ones(self.context_num) / self.context_num
		# self.initial = np.ones(self.context_num) / self.context_num

		self.state = np.zeros(self.context_num)
		self.initial = np.zeros(self.context_num)
		self.initial[0] = self.s
		self.state[0] = self.s
		for i in range(1, self.context_num):
			self.initial[i] = (1 - self.s) / (self.context_num - 1)
			self.state[i] = (1 - self.s) / (self.context_num - 1)

		self.initial_beta = np.zeros((self.class_num, 2))

		self.time = 0
		self.confidence = 0

		self.past_choice = [0]
		self.trajectory = []

	def set_trajectory(self, trajectory):
		self.trajectory = trajectory

	def forward_backward(self):
		T = len(self.trajectory)
		if T == 0:
			return
		self.alpha = np.zeros((T, self.context_num))
		self.beta = np.zeros((T, self.context_num))
		self.temp = np.zeros((T, self.context_num))
		
		for i, data in enumerate(self.trajectory):
			self.temp[i] = self.prob[:, data[0], data[1], data[2]]

			if i == 0:
				self.alpha[0] = np.log(self.initial + 1e-8) + np.log(self.prob[:, data[0], data[1], data[2]])
			else:

				broadcast = np.log(self.tt + 1e-8) + np.expand_dims(self.alpha[i-1], axis = 1)
				self.alpha[i] = np.log(self.prob[:, data[0], data[1], data[2]]) + special.logsumexp(broadcast, axis = 0)


		for i in range(T):
			if i == 0:
				pass
			else:
				data = self.trajectory[T-i]
				broadcast = np.log(self.tt + 1e-8) + np.expand_dims(self.beta[T-i], axis = 0) + np.expand_dims(np.log(self.prob[:, data[0], data[1], data[2]]), axis = 0)
				self.beta[T-1-i] = special.logsumexp(broadcast, axis = 1)


	def forward(self):
		
		T = len(self.trajectory)
		if T == 0:
			return 

		self.forward_backward()
		self.gamma = self.alpha + self.beta
		self.gamma = np.exp(self.gamma - special.logsumexp(self.gamma, axis = 1, keepdims = True))
		self.state = self.gamma[-1]


		self.occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2)) + 1
		self.total_occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num)) + 2
		for i, data in enumerate(self.trajectory):
			self.occupancy[:, data[0], data[1], data[2]] += self.gamma[i]
			self.total_occupancy[:, data[0], data[1]] += self.gamma[i]



		sample = np.random.beta(self.occupancy[:, x, :, 1].T.dot(self.state) + 1, self.occupancy[:, x, :, 0].T.dot(self.state) + 1 )
		
		return 

	def update(self):
		self.forward()


		for it in range(self.iter):
			self.forward_backward()
			T = len(self.trajectory)
			if T == 1:
				self.gamma = self.alpha + self.beta
				self.gamma = np.exp(self.gamma - special.logsumexp(self.gamma, axis = 1))
				self.occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2)) + 1
				self.total_occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num)) + 2
				for i, data in enumerate(self.trajectory):
					self.occupancy[:, data[0], data[1], data[2]] += self.gamma[i]
					self.total_occupancy[:, data[0], data[1]] += self.gamma[i]

				self.initial = self.gamma[0]
				self.prob = self.occupancy / np.expand_dims(self.total_occupancy,axis = 3)

				return

			self.gamma = self.alpha + self.beta

			self.gamma = np.exp(self.gamma - special.logsumexp(self.gamma, axis = 1, keepdims = True))

			self.eta = np.expand_dims(self.alpha[:-1], axis = 2) + np.expand_dims(np.log(self.tt + 1e-8), axis = 0) + np.expand_dims(self.beta[1:], axis = 1) + np.expand_dims(np.log(self.temp[1:]), axis = 1)
			
			self.eta = np.exp(self.eta - special.logsumexp(special.logsumexp(self.eta, axis = 2, keepdims = True), axis = 1, keepdims = True))

			new_state = self.gamma[-1]

			self.occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2)) + 1
			self.total_occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num)) + 2
			#self.prob = self.occupancy / np.expand_dims(self.total_occupancy,axis = 3)
	

			for i, data in enumerate(self.trajectory):
				self.occupancy[:, data[0], data[1], data[2]] += self.gamma[i] 
				self.total_occupancy[:, data[0], data[1]] += self.gamma[i] 

			new_initial = self.gamma[0]
			new_tt = np.sum(self.eta, axis = 0) / np.expand_dims(np.sum(self.gamma[:-1], axis = 0), axis = 1)

			new_prob = self.occupancy / np.expand_dims(self.total_occupancy,axis = 3)

			error =  np.linalg.norm(self.state - new_state) + np.linalg.norm(self.initial - new_initial) + np.linalg.norm(self.tt - new_tt) + np.linalg.norm(self.prob - new_prob) 
			
			if error< 1e-7:
				print(it, self.time)
				return

			self.initial = new_initial
			self.state  = new_state
			self.prob = new_prob
			self.tt = new_tt





class QuantileAgent(object):
	"""docstring for QuantileAgent"""
	def __init__(self, opt, name = "Quantile"):
		super(QuantileAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.quantile_num = opt["quantile_num"]
		self.prob = np.ones( (opt["stimuli_num"], opt["class_num"], opt["quantile_num"]))
		self.A = opt["A"]


		quantile = (np.arange(self.quantile_num)+1) * 1.0 / self.quantile_num
		self.prob = self.prob * np.expand_dims(quantile, (0, 1)) * self.A

		self.count = np.zeros( (opt["stimuli_num"], opt["class_num"]))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name

		self.past_choice = [0]


	def ev(self):
		return np.mean(self.prob, axis = 2)

	def get_ev(self):
		return self.ev()

	def get_choice_prob(self):
		return sum(self.past_choice) / len(self.past_choice)

	def scalars(self):
		hist = {}
		ev = self.ev()
		
		for s in range(self.stimuli_num):
			for a in range(self.class_num):
				hist["fast-reward/simuli-{}/action-{}".format(s, a)] = ev[s, a]
		hist["choice_prob"] = self.get_choice_prob()
		return hist

	def histogram(self):
		return {}
	def forward(self, x):
		self.stimuli = x

		idx = np.random.randint(0, self.quantile_num, (self.class_num))
		
		sample = []
		for i in range(self.class_num):
			sample.append(self.prob[x, i, idx[i]])

		action = np.argmax(sample)
		self.count[x, action] += 1

		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action
		return action
		
	def update(self, r):
		self.r = r
		self.prob[self.stimuli, self.action, :] += (r - self.prob[self.stimuli, self.action, :]) / (1+self.count[self.stimuli, self.action])
		

	



	def reset(self):
		self.prob = np.ones( (self.stimuli_num, self.class_num, self.quantile_num))
		

		quantile = (np.arange(self.quantile_num)+1) * 1.0 / self.quantile_num
		self.prob = self.prob * np.expand_dims(quantile, (0, 1)) * self.A

		self.count = np.zeros( (self.stimuli_num, self.class_num))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.past_choice = [0]

class QuantileLRAgent(object):
	"""docstring for QuantileAgent"""
	def __init__(self, opt, name = "Quantile"):
		super(QuantileLRAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.quantile_num = opt["quantile_num"]
		self.prob = np.ones( (opt["stimuli_num"], opt["class_num"], opt["quantile_num"]))
		
		quantile = ((np.arange(self.quantile_num)+1) * 1.0 ) / self.quantile_num
		
		self.prob = self.prob * np.expand_dims(quantile, (0, 1))
		

		
	

		self.count = np.zeros( (opt["stimuli_num"], opt["class_num"]))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name

		self.past_choice = [0]


	def ev(self):
		return np.mean(self.prob, axis = 2)

	def get_ev(self):
		return self.ev()

	def get_choice_prob(self):
		return sum(self.past_choice) / len(self.past_choice)

	def scalars(self):
		hist = {}
		ev = self.ev()
		
		for s in range(self.stimuli_num):
			for a in range(self.class_num):
				hist["fast-reward/simuli-{}/action-{}".format(s, a)] = ev[s, a]
		hist["choice_prob"] = self.get_choice_prob()
		return hist

	def histogram(self):
		return {}
	def forward(self, x):
		self.stimuli = x

		idx = np.random.randint(0, self.quantile_num, (self.class_num))
		

		sample = []
		for i in range(self.class_num):
			indices = np.random.permutation(np.arange(self.quantile_num))[:3]

			sample.append(np.mean(self.prob[x, i, indices]))
			# if idx[i] != 0:

			# 	sample.append((self.prob[x, i, idx[i]] + self.prob[x, i, idx[i]-1] ) / 2)
			# else:
			# 	sample.append((self.prob[x, i, idx[i]] + 0 ) / 2)


		action = np.argmax(sample)
		self.count[x, action] += 1

		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action
		return action
		
	def update(self, r):
		self.r = r
		self.prob[self.stimuli, self.action, :-1] += (r - self.prob[self.stimuli, self.action, :-1])  / (2+  self.count[self.stimuli, self.action])



	def reset(self):
		self.prob = np.ones( (self.stimuli_num, self.class_num, self.quantile_num))
		quantile = ((np.arange(self.quantile_num)+1) * 1.0 ) / self.quantile_num
		self.prob = self.prob * np.expand_dims(quantile, (0, 1)) 
		

		self.count = np.zeros( (self.stimuli_num, self.class_num))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.past_choice = [0]


# class NeuralQuantileAgent(object):
# 	"""docstring for QuantileAgent"""
# 	def __init__(self, opt, name = "Neural Quantile"):
# 		super(NeuralQuantileAgent, self).__init__()
# 		self.opt = opt
# 		self.stimuli_num = opt["stimuli_num"]
# 		self.class_num = opt["class_num"]
# 		self.quantile_num = opt["quantile_num"]
# 		self.prob = np.ones( (opt["stimuli_num"], opt["class_num"], opt["quantile_num"]))
# 		self.A = opt["A"]


# 		quantile = (np.arange(self.quantile_num)+1) * 1.0 / self.quantile_num
# 		self.prob = self.prob * np.expand_dims(quantile, (0, 1)) * self.A

# 		self.dt = opt["dt"]
# 		self.sample_neurons = [np.zeros((self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]



# 		self.a1 = opt["a1"]
# 		self.b1 = opt["b1"]
	
# 		self.sample_w = -  self.b1 * np.ones((self.quantile_num, self.quantile_num))
# 		for i in range(self.quantile_num):
# 			self.sample_w[i, i] = self.a1

# 		self.value_neurons = np.zeros((self.quantile_num, self.class_num))

# 		self.a2 = opt["a2"]
# 		self.b2 = opt["b2"]

# 		self.decision_neurons = np.zeros(self.class_num)
	
# 		self.decision_w = -  self.b2 * np.ones((self.class_num, self.class_num))
# 		for i in range(self.class_num):
# 			self.decision_w[i, i] = self.a2

# 		self.tau = opt["tau"]
# 		self.eta = opt["eta"]
# 		self.threshold = opt["threshold"]
# 		self.d_interval = opt["d_interval"]



# 		self.count = np.zeros( (opt["stimuli_num"], opt["class_num"]))
# 		self.stimuli = 0
# 		self.action = 0
# 		self.context = 0
# 		self.name = name

# 		self.past_choice = [0]


# 	def ev(self):

# 		return np.mean(self.prob, axis = 2)

# 	def get_ev(self):
# 		return self.ev()

# 	def get_choice_prob(self):
# 		return sum(self.past_choice) / len(self.past_choice)

# 	def scalars(self):
# 		hist = {}
# 		ev = self.ev()
		
# 		for s in range(self.stimuli_num):
# 			for a in range(self.class_num):
# 				hist["fast-reward/simuli-{}/action-{}".format(s, a)] = ev[s, a]
# 		hist["choice_prob"] = self.get_choice_prob()
# 		return hist

# 	def histogram(self):
# 		return {}
# 	def forward(self, x):
# 		self.stimuli = x

# 		self.sample_neurons = [np.zeros((self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
# 		self.value_neurons = np.zeros((self.quantile_num, self.class_num))
# 		self.decision_neurons = np.zeros(self.class_num)

# 		for _ in range(self.d_interval):
# 			self.sample_neurons[x] += self.dt * self.tau * (-self.sample_neurons[x] +  relu(self.sample_w.dot(self.sample_neurons[x]) + 1 + self.eta * np.random.normal(size = (self.quantile_num, self.class_num))))
		
# 			self.value_neurons += self.dt * self.tau * (-self.value_neurons +  relu(self.sample_neurons[x] * self.prob[x].T))
# 			self.decision_neurons += self.dt * self.tau * (-self.decision_neurons + relu(self.decision_w.dot(self.decision_neurons)) + np.sum(self.value_neurons, axis = 0))
			
# 			if np.max(self.decision_neurons) > self.threshold:
# 				action = np.argmax(self.decision_neurons)
# 				break

# 		if np.max(self.decision_neurons) <= self.threshold:
# 			action = np.random.randint(self.class_num)
# 			print("Random decision is made {}".format(self.name))
# 			print(np.max(self.decision_neurons), self.threshold)


# 		self.count[x, action] += 1
		
# 		self.past_choice.append(action)
# 		if len(self.past_choice) > 10:
# 			del self.past_choice[0]


# 		self.action = action
# 		return action
		
# 	def update(self, r):
# 		self.r = r
# 		self.prob[self.stimuli, self.action, :] += (r - self.prob[self.stimuli, self.action, :]) / (1+self.count[self.stimuli, self.action])
		



# 	def reset(self):
# 		self.prob = np.ones( (self.stimuli_num, self.class_num, self.quantile_num))
# 		self.sample_neurons = [np.zeros((self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
# 		self.value_neurons = np.zeros((self.quantile_num, self.class_num))
# 		self.decision_neurons = np.zeros(self.class_num)
		
# 		quantile = (np.arange(self.quantile_num)+1) * 1.0 / self.quantile_num
# 		self.prob = self.prob * np.expand_dims(quantile, (0, 1)) * self.A

# 		self.count = np.zeros( (self.stimuli_num, self.class_num))
# 		self.stimuli = 0
# 		self.action = 0
# 		self.context = 0
# 		self.past_choice = [0]

class NeuralQuantileAgent(object):
	"""docstring for QuantileAgent"""
	def __init__(self, opt, name = "Neural Quantile"):
		super(NeuralQuantileAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.quantile_num = opt["quantile_num"]
		self.prob = np.ones((1, 1, opt["quantile_num"])) * np.random.rand( opt["stimuli_num"], opt["class_num"], 1)
		
		self.K = opt["K"]
		
		self.uniform = opt["uniform"]
		self.uniform_init = opt["uniform_init"]
		if not self.uniform_init:
			# quantile = (np.arange(self.quantile_num)+1) * 1.0 / self.quantile_num
			# self.prob = self.prob * np.expand_dims(quantile, (0, 1)) 

			self.prob = np.ones((self.stimuli_num, self.class_num, self.quantile_num))
			self.prob *= np.expand_dims((np.arange(self.quantile_num)+1) / float(self.quantile_num), (0, 1))
		else:
			mean = np.random.rand(self.stimuli_num, self.class_num)
			for s in range(self.stimuli_num):
				for c in range(self.class_num):
					self.prob[s,c, :] = np.random.normal(0.5, 0.05, self.quantile_num)
			
		

		self.dt = opt["dt"]
		self.sample_neurons = [np.zeros((self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]

		self.decay = 1



		self.a1 = opt["a1"]
		self.b1 = opt["b1"]
	
		self.sample_w = -  self.b1 * np.ones((self.quantile_num, self.quantile_num))
		for i in range(self.quantile_num):
			self.sample_w[i, i] = self.a1

		self.value_neurons = np.zeros((self.quantile_num, self.class_num))

		self.a2 = opt["a2"]
		self.b2 = opt["b2"]

		self.decision_neurons = np.zeros(self.class_num)
	
		self.decision_w = -  self.b2 * np.ones((self.class_num, self.class_num))
		for i in range(self.class_num):
			self.decision_w[i, i] = self.a2

		self.tau = opt["tau"]
		self.eta = opt["eta"]
		self.threshold = opt["threshold"]
		self.d_interval = opt["d_interval"]



		self.count = np.zeros( (opt["stimuli_num"], opt["class_num"]))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name

		self.past_choice = [0]


	def f(self, x):
		return np.minimum(1, relu(x))

	def ev(self):

		return np.mean(self.prob, axis = 2)

	def get_ev(self):
		return self.ev()

	def get_choice_prob(self):
		return sum(self.past_choice) / len(self.past_choice)

	def scalars(self):
		hist = {}
		ev = self.ev()
		
		for s in range(self.stimuli_num):
			for a in range(self.class_num):
				hist["fast-reward/simuli-{}/action-{}".format(s, a)] = ev[s, a]
		hist["choice_prob"] = self.get_choice_prob()
		return hist

	def histogram(self):
		return {"CS": self.prob}
	def forward(self, x):
		self.stimuli = x

		self.sample_neurons = [np.zeros((self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
		self.value_neurons = np.zeros((self.quantile_num, self.class_num))
		self.decision_neurons = np.zeros(self.class_num)

		for j in range(self.d_interval):
			
			self.sample_neurons[x] += 0.03 * (-self.sample_neurons[x] +  self.sample_w.dot(self.f(self.sample_neurons[x])) + (self.K-0.25)*self.b1 + 0.2 * np.random.normal(size = (self.quantile_num, self.class_num)))
		
			self.value_neurons += self.dt  * self.tau * (-self.value_neurons +  relu(self.f(self.sample_neurons[x]) * self.prob[x].T))
			self.decision_neurons += self.dt *0.1 * self.tau * (-self.decision_neurons + relu(self.decision_w.dot(self.decision_neurons)) + 1.0 / self.K * np.sum(self.value_neurons, axis = 0))
			
			if np.max(self.decision_neurons) > self.threshold:
				action = np.argmax(self.decision_neurons)

				break

		if np.max(self.decision_neurons) <= self.threshold:
			action = np.random.choice(self.class_num, p= special.softmax(30 * self.decision_neurons))
			# print("Random decision is made {}".format(self.name))
			# print(self.decision_neurons, self.threshold)
			# print(np.sort(self.sample_neurons[x][:, 0])[-4:])



		self.count[x, action] += 1
		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action
		return action
		
	def update(self, r):
		self.r = r
		if self.uniform:
			self.prob[self.stimuli, self.action, :] += (r - self.prob[self.stimuli, self.action, :] )  / (7**(1/self.decay)+  self.count[self.stimuli, self.action])**self.decay
		else:
			self.prob[self.stimuli, self.action, :-1] += (r - self.prob[self.stimuli, self.action, :-1] )  / (7**(1/self.decay)+  self.count[self.stimuli, self.action])**self.decay
		



	def reset(self):
		self.prob = np.ones((1, 1, self.quantile_num)) * np.random.rand( self.stimuli_num, self.class_num, 1)
		
		self.sample_neurons = [np.zeros((self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
		self.value_neurons = np.zeros((self.quantile_num, self.class_num))
		self.decision_neurons = np.zeros(self.class_num)
		
		if not self.uniform_init:
			# quantile = (np.arange(self.quantile_num)+1) * 1.0 / self.quantile_num
			# self.prob = self.prob * np.expand_dims(quantile, (0, 1)) 

			self.prob = np.ones((self.stimuli_num, self.class_num, self.quantile_num))
			self.prob *= np.expand_dims((np.arange(self.quantile_num)+1) / float(self.quantile_num), (0, 1))
		else:
			mean = np.random.rand(self.stimuli_num, self.class_num)
			for s in range(self.stimuli_num):
				for c in range(self.class_num):
					self.prob[s,c, :] = np.random.normal(0.5, 0.05, self.quantile_num)
		

		


		self.count = np.zeros( (self.stimuli_num, self.class_num))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.past_choice = [0]

class NeuralContextQuantileAgent(object):
	"""docstring for QuantileAgent"""
	def __init__(self, opt, name = "Known-Context Distributional RPE Model"):
		super(NeuralContextQuantileAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.context_num = opt["context_num"]
		self.quantile_num = opt["quantile_num"]
		self.prob = np.ones( (opt["context_num"], opt["stimuli_num"], opt["class_num"], opt["quantile_num"]))
		self.K = opt["K"]
		


		quantile = (np.arange(self.quantile_num)+1)  * 1.0 / self.quantile_num
		self.prob = self.prob * np.expand_dims(quantile, (0, 1, 2)) 

		self.dt = opt["dt"]
		self.sample_neurons = [np.zeros((self.context_num, self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]



		self.a1 = opt["a1"]
		self.b1 = opt["b1"]
	
		self.sample_w = -  self.b1 * np.ones((self.context_num, self.quantile_num, self.quantile_num))
		for j in range(self.context_num):
			for i in range(self.quantile_num):
				self.sample_w[j, i, i] = self.a1

		self.value_neurons = np.zeros((self.context_num, self.quantile_num, self.class_num))

		self.a2 = opt["a2"]
		self.b2 = opt["b2"]

		self.decision_neurons = np.zeros(self.class_num)
	
		self.decision_w = -  self.b2 * np.ones((self.class_num, self.class_num))
		for i in range(self.class_num):
			self.decision_w[i, i] = self.a2

		self.tau = opt["tau"]
		self.eta = opt["eta"]
		self.threshold = opt["threshold"]
		self.d_interval = opt["d_interval"]



		self.count = np.zeros( (self.context_num, opt["stimuli_num"], opt["class_num"]))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name
		self.trial = 0

		self.past_choice = [0]

	def f(self, x):
		return np.minimum(1, relu(x))


	def ev(self):
		return np.mean(self.prob, axis = 3)

	def get_ev(self):
		return self.ev()

	def get_choice_prob(self):
		return sum(self.past_choice) / len(self.past_choice)

	def scalars(self):
		hist = {}
		ev = self.ev()
		for c in range(self.context_num):
			for s in range(self.stimuli_num):
				for a in range(self.class_num):
					hist["fast-reward/context-{}/simuli-{}/action-{}".format(c, s, a)] = ev[c, s, a]
					for r in range(2):
						hist["likelihood/context-{}/simuli-{}/action-{}/reward-{}".format(c, s, a, r)] = self.prob[c, s, a, r]
				
		hist["choice_prob"] = self.get_choice_prob()
		hist["fast-reward"] = self.r
	
		return hist

	def histogram(self):
		return {"CS": self.prob}
	def forward(self, x):
		self.stimuli = x
		residue = self.trial % (self.opt["block_size"] * 2)
		if residue < self.opt["block_size"]:
			self.context = 0
		else:
			self.context = 1


		self.sample_neurons = [np.zeros((self.context_num, self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
		self.value_neurons = np.zeros((self.context_num, self.quantile_num, self.class_num))
		self.decision_neurons = np.zeros(self.class_num)

		for _ in range(self.d_interval):
			
			self.sample_neurons[x][self.context] +=  0.03  * (-self.sample_neurons[x][self.context] +  self.sample_w[self.context].dot(self.f(self.sample_neurons[x][self.context])) + (self.K-0.25)*self.b1 +  0.2 * np.random.normal(size = (self.quantile_num, self.class_num)))
		
			self.value_neurons[self.context] += self.dt * self.tau * (-self.value_neurons[self.context] +  relu(self.f(self.sample_neurons[x][self.context]) * self.prob[self.context, x, :, :].T))
			self.decision_neurons += self.dt * 0.1 * self.tau * (-self.decision_neurons + relu(self.decision_w.dot(self.decision_neurons)) + 1.0 / self.K * np.sum(self.value_neurons[self.context], axis = 0))
			
			if np.max(self.decision_neurons) > self.threshold:
				action = np.argmax(self.decision_neurons)
				break

		if np.max(self.decision_neurons) <= self.threshold:
			action = np.random.choice(self.class_num, p= special.softmax(30 * self.decision_neurons))
			print("Random decision is made {}".format(self.name))
			print(self.decision_neurons, self.threshold)

		
		self.count[self.context, x, action] += 1

		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action

		self.trial += 1

		return action
		
	def update(self, r):
		self.r = r
		self.prob[self.context, self.stimuli, self.action, :-1] += (r - self.prob[self.context, self.stimuli, self.action, :-1]) / (7+self.count[self.context, self.stimuli, self.action])
		



	def reset(self):
		self.prob = np.ones( (self.context_num, self.stimuli_num, self.class_num, self.quantile_num))
		self.sample_neurons = [np.zeros((self.context_num, self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
		self.value_neurons = np.zeros((self.context_num, self.quantile_num, self.class_num))
		self.decision_neurons = np.zeros(self.class_num)
		
		quantile = (np.arange(self.quantile_num)+1)  * 1.0 / self.quantile_num
		self.prob = self.prob * np.expand_dims(quantile, (0, 1, 2)) 

		self.count = np.zeros( (self.context_num, self.stimuli_num, self.class_num))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.trial = 0
		self.past_choice = [0]



class RandomAgent(object):
	"""docstring for QuantileAgent"""
	def __init__(self, opt, name = "Random"):
		super(RandomAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.prob = np.random.rand( opt["stimuli_num"], opt["class_num"])


		self.count = np.zeros( (opt["stimuli_num"], opt["class_num"]))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name

		self.past_choice = [0]


	def ev(self):
		return self.prob

	def get_ev(self):
		return self.ev()

	def get_choice_prob(self):
		return sum(self.past_choice) / len(self.past_choice)

	def scalars(self):
		hist = {}
		ev = self.ev()
		
		for s in range(self.stimuli_num):
			for a in range(self.class_num):
				hist["fast-reward/simuli-{}/action-{}".format(s, a)] = ev[s, a]
		hist["choice_prob"] = self.get_choice_prob()
		return hist

	def histogram(self):
		return {}
	def forward(self, x):
		self.stimuli = x

		noise = np.random.rand( self.opt["stimuli_num"], self.opt["class_num"]) * 0.005
		action = np.argmax(self.prob[x] + noise)
		self.count[x, action] += 1

		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action
		return action
		
	def update(self, r):
		self.r = r
		self.prob[self.stimuli, self.action] += (r - self.prob[self.stimuli, self.action]) / (1+self.count[self.stimuli, self.action])
		

	



	def reset(self):
		self.prob = np.random.rand( self.opt["stimuli_num"], self.opt["class_num"])


		self.count = np.zeros( (self.stimuli_num, self.class_num))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.past_choice = [0]


class RandomNeuralAgent(object):
	"""docstring for QuantileAgent"""
	def __init__(self, opt, name = "Random Neural"):
		super(RandomNeuralAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.prob = np.random.rand( opt["stimuli_num"], opt["class_num"])

		self.c = np.zeros(opt["class_num"])
		self.dt = opt["dt"]
		self.a = opt["a"]
		self.b = opt["b"]
	
		self.w = -  self.b * np.ones((self.class_num, self.class_num))
		for i in range(self.class_num):
			self.w[i, i] = self.a

		self.tau = opt["tau"]
		self.eta = opt["eta"]
		self.threshold = opt["threshold"]
		self.d_interval = opt["d_interval"]



		self.count = np.zeros( (opt["stimuli_num"], opt["class_num"]))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name

		self.past_choice = [0]


	def ev(self):
		return self.prob

	def get_ev(self):
		return self.ev()

	def get_choice_prob(self):
		return sum(self.past_choice) / len(self.past_choice)

	def scalars(self):
		hist = {}
		ev = self.ev()
		
		for s in range(self.stimuli_num):
			for a in range(self.class_num):
				hist["fast-reward/simuli-{}/action-{}".format(s, a)] = ev[s, a]
		hist["choice_prob"] = self.get_choice_prob()

		for a in range(self.class_num):
			hist["activity/action-{}".format(a)] = self.c[a]

		return hist

	def histogram(self):
		return {}
	def forward(self, x):
		self.stimuli = x
		self.c = np.zeros(self.class_num)


		for _ in range(self.d_interval):
			self.c += self.dt * self.tau * (-self.c +  relu(self.w.dot(self.c) + self.prob[x] + self.eta * np.random.normal(size = self.class_num)))
			
			if np.max(self.c) > self.threshold:
				action = np.argmax(self.c)
				break

		if np.max(self.c) <= self.threshold:
			action = np.random.randint(self.class_num)
			print("Random decision is made {}".format(self.name))
			print(np.max(self.c))

		self.count[x, action] += 1
	

		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action
		return action
		
	def update(self, r):
		self.r = r
		self.prob[self.stimuli, self.action] += (r - self.prob[self.stimuli, self.action]) / (1+self.count[self.stimuli, self.action])
		

	def reset(self):
		self.prob = np.random.rand( self.opt["stimuli_num"], self.opt["class_num"])

		self.c = np.zeros(self.opt["class_num"])
		self.count = np.zeros( (self.stimuli_num, self.class_num))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.past_choice = [0]




# class SimpleBG(object):
# 	"""docstring for SimpleBG"""
# 	def __init__(self, opt, name = "sBG"):
# 		super(SimpleBG, self).__init__()
# 		self.name = name
# 		self.opt = opt
# 		self.stimuli_num = opt["stimuli_num"]
# 		self.class_num = opt["class_num"]

# 		self.cortex = Sequential(opt["stimuli_num"] + opt["class_num"], opt["time_size"], opt, name + "/cortex")
# 		self.BG = EasyBG((opt["stimuli_num"] + opt["class_num"]) * opt["time_size"], opt["class_num"], opt, name + "/BG")

# 	def forward(self, x):
# 		c = self.cortex.forward(np.concatenate([x, 0.05 *(self.BG.p)]))
# 		return self.BG.forward(c)

# 	def update(self, r):
# 		self.BG.update(r)

# 	def histogram(self):
# 		return {**self.BG.histogram(), **self.cortex.histogram()}

# 	def scalars(self):
# 		return {**self.BG.scalars(), **self.cortex.scalars()}

# class SimpleActBG(object):
# 	"""docstring for SimpleActBG"""
# 	def __init__(self, opt, name = "saBG"):
# 		super(SimpleActBG, self).__init__()
# 		self.name = name
# 		self.opt = opt
# 		self.stimuli_num = opt["stimuli_num"]
# 		self.class_num = opt["class_num"]

# 		self.cortex = SequentialAction(opt["stimuli_num"], opt["class_num"], opt["time_size"], opt, name + "/cortex")
# 		self.BG = EasyBG((opt["stimuli_num"] + opt["class_num"]) * opt["time_size"], opt["class_num"], opt, name + "/BG")

# 	def forward(self, x):
# 		new_x = np.append(x, self.BG.p)
# 		c = self.cortex.forward(new_x)
# 		return self.BG.forward(c)

# 	def update(self, r):
# 		self.BG.update(r)

# 	def histogram(self):
# 		return {**self.BG.histogram(), **self.cortex.histogram()}

# 	def scalars(self):
# 		return {**self.BG.scalars(), **self.cortex.scalars()}


		
		
			

# opt = {}
# opt["context_num"] = 3
# opt["stimuli_num"] = 5
# opt["class_num"] = 3
# opt["block_size"] = 10
# opt["max_trial"] = 200
# opt["discount_rate"] = 1e-2
# db = ThompsonDCAgent(opt)
# db.act(1)









# class LogisticRegression(nn.Module):
# 	"""docstring for LogisticRegression"""

# 	def __init__(self, opt):
# 		super(LogisticRegression, self).__init__()
# 		self.opt = opt
# 		self.weight = nn.Linear(self.opt['feature_size'], self.opt['label_size'])

# 	def forward(self, x, label):
# 		logits = F.sigmoid(self.weight(x))
# 		if not self.training:
# 			return logits

# 		loss = - (torch.log(logits + 1e-8) * label + torch.log(1-logits + 1e-8) * (1-label)).mean()
# 		return loss 

# 	def save(self):
# 		param = {
# 		'state_dict': self.state_dict(),
# 		'opt': self.opt
# 		}
# 		torch.save(param, self.opt['save_file'])


# class LSTM_baseline(nn.Module):
# 	"""docstring for RNNBottleNeck"""

# 	def __init__(self, opt):
# 		super(LSTM_baseline, self).__init__()
# 		self.opt = opt
# 		self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_size'])
# 		self.birnn = nn.LSTM(input_size = opt['embedding_size'], hidden_size = opt['hidden_size'], batch_first = True, bidirectional = True)

# 		self.dropout = nn.Dropout(p=opt['dropout'])
# 		self.rnn = nn.LSTM(input_size = opt['hidden_size'] * 2, hidden_size = opt['hidden_size'] * 2, batch_first = True, dropout = opt['dropout'], num_layers = opt['num_layer'])

# 		self.fc = nn.Sequential(
# 			nn.Dropout(p=opt['dropout']),
# 			nn.Linear(2 * opt['hidden_size'], 2 * opt['hidden_size']),
# 			nn.ReLU(),
# 			nn.Dropout(p=opt['dropout'])
# 			)
		
# 		self.output_heads = nn.ModuleList([nn.Linear(2 * opt['hidden_size'], opt['label1_size']), nn.Linear(2 * opt['hidden_size'], opt['label2_size']), nn.Linear(2 * opt['hidden_size'], opt['label3_size'])])


# 	def set_embedding(self, dictionary):
# 		if not self.opt['pretrain_word'] or not os.path.isfile(self.opt['embedding_file']):
# 			logger.info('Not using embedding word')
# 			return

# 		embedding = load_embedding(self.opt, dictionary)
# 		new_size = embedding.size()
# 		old_size = self.embedding.weight.size()

# 		if new_size[0] != old_size[0] or new_size[1] != old_size[1]:
# 			raise RuntimeError('Embedding dimensions do not match.')

# 		self.embedding.weight.data = embedding
# 		for p in self.embedding.parameters():
# 			p.requires_grad = False

# 	def forward(self, x, length, label, indices, mask):

# 		current_inputs = self.embedding(x)

# 		current_inputs = self.dropout(current_inputs)

# 		packed = nn.utils.rnn.pack_padded_sequence(current_inputs, length, batch_first = True)
# 		packed_out, packed_hidden = self.birnn(packed)
# 		current_inputs, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first = True)
		
# 		current_inputs = self.dropout(current_inputs)
# 		packed = nn.utils.rnn.pack_padded_sequence(current_inputs, length, batch_first = True)
# 		packed_out, packed_hidden = self.rnn(packed)
# 		current_inputs, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first = True)

# 		current_inputs = current_inputs[:, -1, :].squeeze(1)

# 		current_inputs = self.fc(current_inputs)
# 		output1, output2, output3 = self.output_heads[0](current_inputs), self.output_heads[1](current_inputs), self.output_heads[2](current_inputs)

# 		loss1 = F.cross_entropy(output1, label[:, 0])
# 		loss2 = F.cross_entropy(output2, label[:, 1])
# 		loss3 = F.cross_entropy(output3, label[:, 2])

# 		loss = (loss1 + loss2 + loss3) / 3.0
# 		if self.training:
# 			return loss
# 		else:
# 			return loss, (output1, output2, output3)

# class LSTM_Att(nn.Module):
# 	"""docstring for RNNBottleNeck"""

# 	def __init__(self, opt):
# 		super(LSTM_Att, self).__init__()
# 		self.opt = opt
# 		self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_size'])
# 		self.birnn = nn.LSTM(input_size = opt['embedding_size'], hidden_size = opt['hidden_size'], batch_first = True, bidirectional = True, dropout = opt['dropout'], num_layers = opt['num_layer'])
		
# 		self.fc = nn.Sequential(
# 			nn.Dropout(p=opt['dropout']),
# 			SelfAtt(2 * opt['hidden_size']),
# 			nn.ReLU(),
# 			nn.Dropout(p=opt['dropout'])
# 			)
		
# 		self.output_heads = nn.Linear(2 * opt['hidden_size'], opt['label1_size'] + opt['label2_size'] + opt['label3_size'])

# 	def get_project_list(self, total_size, num_head):
# 		project_dim = (total_size + num_head - 1) / num_head
# 		last_dim = total_size - project_dim * (num_head - 1)
# 		return [ project_dim if i is not num_head - 1 else last_dim for i in xrange(num_head)]


# 	def set_embedding(self, dictionary):
# 		if not self.opt['pretrain_word'] or not os.path.isfile(self.opt['embedding_file']):
# 			logger.info('Not using embedding word')
# 			return

# 		embedding = load_embedding(self.opt, dictionary)
# 		new_size = embedding.size()
# 		old_size = self.embedding.weight.size()

# 		if new_size[0] != old_size[0] or new_size[1] != old_size[1]:
# 			raise RuntimeError('Embedding dimensions do not match.')

# 		self.embedding.weight.data = embedding
# 		for p in self.embedding.parameters():
# 			p.requires_grad = False

# 	def forward(self, x, length, label, indices, mask, bottleneck = False):
# 		length = length.data.cpu().numpy()
# 		current_inputs = self.embedding(x)

# 		packed = nn.utils.rnn.pack_padded_sequence(current_inputs, length, batch_first = True)
# 		packed_out, packed_hidden = self.birnn(packed)
# 		current_inputs, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first = True)

# 		current_inputs = self.fc(current_inputs)

# 		if bottleneck:
# 			return current_inputs


# 		output = self.output_heads(current_inputs)

# 		output1, output2, output3 = output[:, :self.opt['label1_size']], output[:, self.opt['label1_size']:self.opt['label1_size'] +self.opt['label2_size']], output[:, self.opt['label1_size'] +self.opt['label2_size']: self.opt['label1_size'] +self.opt['label2_size']+self.opt['label3_size']]

# 		loss1 = F.cross_entropy(output1, label[:, 0])
# 		loss2 = F.cross_entropy(output2, label[:, 1])
# 		loss3 = F.cross_entropy(output3, label[:, 2])

# 		loss = (loss1 + loss2 + loss3) / 3.0
# 		if self.training:
# 			return loss
# 		else:
# 			return loss, (output1, output2, output3)

# class LSTM_MultiAtt(nn.Module):
# 	"""docstring for RNNBottleNeck"""

# 	def __init__(self, opt):
# 		super(LSTM_MultiAtt, self).__init__()
# 		self.opt = opt
# 		self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_size'])
# 		self.birnn = nn.LSTM(input_size = opt['embedding_size'], hidden_size = opt['hidden_size'], batch_first = True, bidirectional = True, dropout = opt['dropout'], num_layers = opt['num_layer'])
# 		self.fc = nn.Sequential(
# 			nn.Dropout(p=opt['dropout']),
# 			SelfMultiAtt(2 * opt['hidden_size'], self.get_project_list(2 * opt['hidden_size'], opt['num_head'])),
# 			nn.ReLU(),
# 			nn.Dropout(p=opt['dropout'])
# 			)
		

# 		self.output_heads = nn.Linear(2 * opt['hidden_size'], opt['label1_size'] + opt['label2_size'] + opt['label3_size'])

# 		self.output = nn.Linear(2 * opt['hidden_size'], opt['label_size'])

# 		for p in self.output.parameters():
# 			p.requires_grad = opt['finetune']


# 	def get_project_list(self, total_size, num_head):
# 		project_dim = (total_size + num_head - 1) / num_head
# 		last_dim = total_size - project_dim * (num_head - 1)
# 		return [ project_dim if i is not num_head - 1 else last_dim for i in xrange(num_head)]

# 	def set_embedding(self, dictionary):
# 		if not self.opt['pretrain_word'] or not os.path.isfile(self.opt['embedding_file']):
# 			logger.info('Not using embedding word')
# 			return

# 		embedding = load_embedding(self.opt, dictionary)
# 		new_size = embedding.size()
# 		old_size = self.embedding.weight.size()

# 		if new_size[0] != old_size[0] or new_size[1] != old_size[1]:
# 			raise RuntimeError('Embedding dimensions do not match.')

# 		self.embedding.weight.data = embedding
# 		for p in self.embedding.parameters():
# 			p.requires_grad = False

# 	def forward(self, x, length, label, indices, mask,  bottleneck = False):
# 		length = length.data.cpu().numpy()
# 		current_inputs = self.embedding(x)

# 		packed = nn.utils.rnn.pack_padded_sequence(current_inputs, length, batch_first = True)
# 		packed_out, packed_hidden = self.birnn(packed)
# 		current_inputs, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first = True)

# 		current_inputs = self.fc(current_inputs)

# 		if bottleneck:
# 			return current_inputs

# 		if self.opt['finetune']:

# 			output = self.output_heads(current_inputs)

# 			output1, output2, output3 = output[:, :self.opt['label1_size']], output[:, self.opt['label1_size']:self.opt['label1_size'] +self.opt['label2_size']], output[:, self.opt['label1_size'] +self.opt['label2_size']: self.opt['label1_size'] +self.opt['label2_size']+self.opt['label3_size']]

# 			loss1 = F.cross_entropy(output1, label[:, 0])
# 			loss2 = F.cross_entropy(output2, label[:, 1])
# 			loss3 = F.cross_entropy(output3, label[:, 2])

# 			industry_loss = (loss1 + loss2 + loss3) / 3.0

# 			output = F.sigmoid(self.output(current_inputs))
# 			loss = (- (torch.log(output + 1e-8) * indices * self.opt['class_weight'][1] + self.opt['class_weight'][0]* torch.log(1-output + 1e-8) * (1-indices)) * mask).sum() / (mask.sum() + 1e-8)
# 			if self.training:
# 				return loss * self.opt['lambda'] + industry_loss * self.opt['alpha'] 
# 			else:
# 				return loss, industry_loss, output, output1, output2, output3
# 		else:
# 			output = self.output_heads(current_inputs)

# 			output1, output2, output3 = output[:, :self.opt['label1_size']], output[:, self.opt['label1_size']:self.opt['label1_size'] +self.opt['label2_size']], output[:, self.opt['label1_size'] +self.opt['label2_size']: self.opt['label1_size'] +self.opt['label2_size']+self.opt['label3_size']]

# 			loss1 = F.cross_entropy(output1, label[:, 0])
# 			loss2 = F.cross_entropy(output2, label[:, 1])
# 			loss3 = F.cross_entropy(output3, label[:, 2])

# 			loss = (loss1 + loss2 + loss3) / 3.0
# 			if self.training:
# 				return loss
# 			else:
# 				return loss, (output1, output2, output3)


# class ConvNet(nn.Module):
# 	"""docstring for RNNBottleNeck"""

# 	def __init__(self, opt):
# 		super(ConvNet, self).__init__()
# 		self.opt = opt
# 		self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_size'])

# 		self.conv = DilatedConv(1, opt['embedding_size'], opt['hidden_channel'], 2)

# 		self.conv_list = nn.ModuleList([DilatedConv(d, opt['hidden_channel'], opt['hidden_channel'], 2, groups = 1) for d in opt['dilation']])
# 		self.gate_list = nn.ModuleList([DilatedConv(d, opt['hidden_channel'], opt['hidden_channel'], 2, groups = 1) for d in opt['dilation']])
# 		self.batch_norm_list = nn.ModuleList([nn.BatchNorm1d(opt['hidden_channel']) for _ in opt['dilation']] )
# 		self.dropout = nn.Dropout(p=opt['dropout'])

# 		self.onexone_list = nn.ModuleList([DilatedConv(1, opt['hidden_channel'], opt['hidden_size'], 1) for _ in opt['dilation']])
# 		self.att = SelfAtt(opt['hidden_size'])

# 		self.linear = nn.Linear(opt['hidden_size'], opt['hidden_size'])
		
# 		self.output_heads = nn.Linear(opt['hidden_size'], opt['label1_size'] + opt['label2_size'] + opt['label3_size'])

# 		self.output = nn.Linear(opt['hidden_size'], opt['label_size'])
		
# 		# for p in self.output_heads.parameters():
# 		# 	p.requires_grad = not opt['finetune']
		
# 		for p in self.output.parameters():
# 			p.requires_grad = opt['finetune']
# 	# 	self.fix_parameter()

# 	# def fix_parameter(self):
# 	# 	for p in self.conv.parameters():
# 	# 		p.requires_grad = not opt['finetune']
# 	# 	for p in self.conv_list.parameters():
# 	# 		p.requires_grad = not opt['finetune']
# 	# 	for p in self.gate_list.parameters():
# 	# 		p.requires_grad = not opt['finetune']
# 	# 	for p in self.batch_norm_list.parameters():
# 	# 		p.requires_grad = not opt['finetune']
# 	# 	for p in self.onexone_list.parameters():
# 	# 		p.requires_grad = not opt['finetune']
		


# 	def set_embedding(self, dictionary):
# 		if not self.opt['pretrain_word'] or not os.path.isfile(self.opt['embedding_file']):
# 			logger.info('Not using embedding word')
# 			return

# 		embedding = load_embedding(self.opt, dictionary)
# 		new_size = embedding.size()
# 		old_size = self.embedding.weight.size()

# 		if new_size[0] != old_size[0] or new_size[1] != old_size[1]:
# 			raise RuntimeError('Embedding dimensions do not match.')

# 		self.embedding.weight.data = embedding
# 		for p in self.embedding.parameters():
# 			p.requires_grad = False

# 	def residual_block(self, x, i):

# 		forward = self.batch_norm_list[i](x)
# 		forward = self.conv_list[i](forward) * F.tanh(self.gate_list[i](forward))
# 		skip = self.onexone_list[i](forward)
# 		residual = forward + x

# 		residual = self.dropout(residual)
# 		skip = self.dropout(skip)

# 		return residual, skip 

# 	def forward(self, x, length, label, indices, mask, bottleneck = False):

# 		skip_list = []
# 		current_inputs = self.embedding(x)
# 		current_inputs = torch.transpose(current_inputs, 2, 1)
# 		current_inputs = self.conv(current_inputs)


# 		for i in xrange(len(self.conv_list)):
# 			current_inputs, skip = self.residual_block(current_inputs, i)
# 			skip_list.append(skip.unsqueeze(0))

# 		current_inputs = torch.cat(skip_list, 0)
# 		current_inputs = torch.sum(current_inputs, 0).squeeze(0)
# 		current_inputs = torch.transpose(current_inputs, 2, 1).contiguous()

# 		current_inputs = F.relu(self.att(current_inputs))
# 		current_inputs = F.relu(self.linear(current_inputs))

# 		if bottleneck:
# 			return current_inputs

# 		if self.opt['finetune']:

# 			output = self.output_heads(current_inputs)

# 			output1, output2, output3 = output[:, :self.opt['label1_size']], output[:, self.opt['label1_size']:self.opt['label1_size'] +self.opt['label2_size']], output[:, self.opt['label1_size'] +self.opt['label2_size']: self.opt['label1_size'] +self.opt['label2_size']+self.opt['label3_size']]

# 			loss1 = F.cross_entropy(output1, label[:, 0])
# 			loss2 = F.cross_entropy(output2, label[:, 1])
# 			loss3 = F.cross_entropy(output3, label[:, 2])

# 			industry_loss = (loss1 + loss2 + loss3) / 3.0

# 			output = F.sigmoid(self.output(current_inputs))
# 			loss = (- (torch.log(output + 1e-8) * indices * self.opt['class_weight'][1] + self.opt['class_weight'][0] * torch.log(1-output + 1e-8) * (1-indices)) * mask).sum() / (mask.sum() + 1e-8)
# 			if self.training:
# 				return loss * self.opt['lambda'] + industry_loss * self.opt['alpha']
# 			else:
# 				return loss, industry_loss, output, output1, output2, output3
# 		else:
# 			output = self.output_heads(current_inputs)

# 			output1, output2, output3 = output[:, :self.opt['label1_size']], output[:, self.opt['label1_size']:self.opt['label1_size'] +self.opt['label2_size']], output[:, self.opt['label1_size'] +self.opt['label2_size']: self.opt['label1_size'] +self.opt['label2_size']+self.opt['label3_size']]

# 			loss1 = F.cross_entropy(output1, label[:, 0])
# 			loss2 = F.cross_entropy(output2, label[:, 1])
# 			loss3 = F.cross_entropy(output3, label[:, 2])

# 			loss = (loss1 + loss2 + loss3) / 3.0
# 			if self.training:
# 				return loss
# 			else:
# 				return loss, (output1, output2, output3)


# class AttNet(nn.Module):
# 	"""docstring for AttNet"""
# 	def __init__(self, opt):
# 		super(AttNet, self).__init__()
# 		self.opt = opt
# 		self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_size'])
# 		self.first_att = FancySelfAtt(opt['embedding_size'], opt['key_size'], opt['hidden_size'])

# 		self.att_list = nn.ModuleList([FancySelfAtt(opt['hidden_size'], opt['key_size'], opt['hidden_size']) for _ in opt['num_layer']])
# 		self.att_norm_list = nn.ModuleList([nn.InstanceNorm1d(opt['hidden_size'], affine = True) for _ in opt['num_layer']])
# 		self.fc_list = nn.ModuleList([
# 			nn.Sequential(
# 				Linear3d(opt['hidden_size'], opt['hidden1_size']),
# 				nn.ReLU(),
# 				Linear3d(opt['hidden1_size'], opt['hidden_size']),
# 				)
# 		 for _ in opt['num_layer']])
# 		self.fc_norm_list = nn.ModuleList([nn.InstanceNorm1d(opt['hidden_size'], affine = True) for _ in opt['num_layer']])

# 		self.final_att = SelfAtt(opt['hidden_size'])
# 		self.output_heads = nn.Linear(opt['hidden_size'], opt['label1_size'] + opt['label2_size'] + opt['label3_size'])


# 	def set_embedding(self, dictionary):
# 		if not self.opt['pretrain_word'] or not os.path.isfile(self.opt['embedding_file']):
# 			logger.info('Not using embedding word')
# 			return

# 		embedding = load_embedding(self.opt, dictionary)
# 		new_size = embedding.size()
# 		old_size = self.embedding.weight.size()

# 		if new_size[0] != old_size[0] or new_size[1] != old_size[1]:
# 			raise RuntimeError('Embedding dimensions do not match.')

# 		self.embedding.weight.data = embedding
# 		for p in self.embedding.parameters():
# 			p.requires_grad = False

# 	def residual_black(self, x, i):
# 		pass

# 	def forward(self, x, length, label, indices, mask):

# 		current_inputs = self.embedding(x)



		














		
