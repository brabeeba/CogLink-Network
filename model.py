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



		
		

			
# class HMMAgent(object):
# 	"""docstring for TwoTimeScaleAgent"""
# 	def __init__(self, opt, name = "HMM Model"):
# 		super(HMMAgent, self).__init__()
# 		self.opt = opt
# 		self.stimuli_num = opt["stimuli_num"]
# 		self.class_num = opt["class_num"]
# 		self.context_num = opt["context_num"]
# 		self.iter = opt["iter"]
# 		self.name = name

# 		self.s = 0.9

# 		self.prob = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2))
# 		self.prob[:, :, :,0] = np.random.rand(self.context_num, self.stimuli_num, self.class_num)
# 		self.prob[:, :, :, 1] = 1 - self.prob[:, :, :,0]
	
# 		self.tt = (1-self.s) / (self.context_num - 1) * np.ones((self.context_num, self.context_num))
# 		for i in range(0, self.context_num):
# 			self.tt[i, i] = self.s 



# 		self.context = 0
# 		self.action = 0
# 		self.stimuli = 0


# 		# self.state =  np.ones(self.context_num) / self.context_num
# 		# self.initial = np.ones(self.context_num) / self.context_num

# 		self.state = np.zeros(self.context_num)
# 		self.initial = np.zeros(self.context_num)
# 		self.initial[0] = self.s
# 		self.state[0] = self.s
# 		for i in range(1, self.context_num):
# 			self.initial[i] = (1 - self.s) / (self.context_num - 1)
# 			self.state[i] = (1 - self.s) / (self.context_num - 1)

# 		self.time = 0
# 		self.confidence = 0

# 		self.past_choice = [0]
# 		self.trajectory = []


# 	def ev(self):
# 		return self.prob[:, :, :, 1] 

# 	def get_ev(self):
# 		return np.sum(self.prob[:, :, :, 1] * np.expand_dims(self.state, axis = [1, 2]), axis = 0)

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
# 						hist["likelihood/context-{}/simuli-{}/action-{}/reward-{}".format(c, s, a, r)] = self.prob[c, s, a, r]
				
# 		hist["choice_prob"] = self.get_choice_prob()
# 		hist["fast-reward"] = self.r
# 		hist["context-difference"] = self.state[0] - self.state[1]
	
# 		return hist

# 	def forward_backward(self):
# 		T = len(self.trajectory)
# 		if T == 0:
# 			return
# 		self.alpha = np.zeros((T, self.context_num))
# 		self.beta = np.zeros((T, self.context_num))
# 		self.temp = np.zeros((T, self.context_num))
		
# 		for i, data in enumerate(self.trajectory):
# 			self.temp[i] = self.prob[:, data[0], data[1], data[2]]

# 			if i == 0:
# 				self.alpha[0] = np.log(self.initial + 1e-8) + np.log(self.prob[:, data[0], data[1], data[2]])
# 			else:

# 				broadcast = np.log(self.tt + 1e-8) + np.expand_dims(self.alpha[i-1], axis = 1)
# 				self.alpha[i] = np.log(self.prob[:, data[0], data[1], data[2]]) + special.logsumexp(broadcast, axis = 0)


# 		for i in range(T):
# 			if i == 0:
# 				pass
# 			else:
# 				data = self.trajectory[T-i]
# 				broadcast = np.log(self.tt + 1e-8) + np.expand_dims(self.beta[T-i], axis = 0) + np.expand_dims(np.log(self.prob[:, data[0], data[1], data[2]]), axis = 0)
# 				self.beta[T-1-i] = special.logsumexp(broadcast, axis = 1)


# 	def forward(self, x):
# 		self.time += 1
# 		self.stimuli = x

# 		T = len(self.trajectory)
# 		if T == 0:
# 			self.action = np.random.randint(self.class_num)
# 			self.past_choice.append(self.action)
# 			if len(self.past_choice) > 10:
# 				del self.past_choice[0]
# 			return self.action

# 		self.forward_backward()
# 		self.gamma = self.alpha + self.beta
# 		self.gamma = np.exp(self.gamma - special.logsumexp(self.gamma, axis = 1, keepdims = True))
# 		self.state = self.gamma[-1]

# 		if self.state[0] - self.state[1] >= 0:
# 			self.context = 0
# 		else:
# 			self.context = 1


# 		self.occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2)) + 1
# 		self.total_occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num)) + 2
# 		for i, data in enumerate(self.trajectory):
# 			self.occupancy[:, data[0], data[1], data[2]] += self.gamma[i]
# 			self.total_occupancy[:, data[0], data[1]] += self.gamma[i]

# 		sample = np.random.beta(self.occupancy[:, x, :, 1].T.dot(self.state) + 1, self.occupancy[:, x, :, 0].T.dot(self.state) + 1 )
# 		self.action = np.argmax(sample)

# 		self.past_choice.append(self.action)
# 		if len(self.past_choice) > 10:
# 			del self.past_choice[0]

# 		return self.action

# 	def update(self, r):

# 		self.r = r 
# 		self.trajectory.append((self.stimuli, self.action, self.r))

# 		if self.time % 100 == 0:
# 			print(self.time)


# 		# if self.time % 5 != 0:
# 		# 	return
# 		# if self.time > 600:
# 		# 	if self.time % 10 != 0:
# 		# 		return


# 		for it in range(self.iter):
# 			self.forward_backward()
# 			T = len(self.trajectory)
# 			if T == 1:
# 				self.gamma = self.alpha + self.beta
# 				self.gamma = np.exp(self.gamma - special.logsumexp(self.gamma, axis = 1))
# 				self.occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2)) + 1
# 				self.total_occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num)) + 2
# 				for i, data in enumerate(self.trajectory):
# 					self.occupancy[:, data[0], data[1], data[2]] += self.gamma[i]
# 					self.total_occupancy[:, data[0], data[1]] += self.gamma[i]

# 				self.initial = self.gamma[0]
# 				self.prob = self.occupancy / np.expand_dims(self.total_occupancy,axis = 3)
# 				return

# 			self.gamma = self.alpha + self.beta

# 			self.gamma = np.exp(self.gamma - special.logsumexp(self.gamma, axis = 1, keepdims = True))

# 			self.eta = np.expand_dims(self.alpha[:-1], axis = 2) + np.expand_dims(np.log(self.tt + 1e-8), axis = 0) + np.expand_dims(self.beta[1:], axis = 1) + np.expand_dims(np.log(self.temp[1:]), axis = 1)
			
# 			self.eta = np.exp(self.eta - special.logsumexp(special.logsumexp(self.eta, axis = 2, keepdims = True), axis = 1, keepdims = True))

# 			new_state = self.gamma[-1]

# 			self.occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2)) + 1
# 			self.total_occupancy = np.zeros((self.context_num, self.stimuli_num, self.class_num)) + 2
# 			#self.prob = self.occupancy / np.expand_dims(self.total_occupancy,axis = 3)
	

# 			for i, data in enumerate(self.trajectory):
# 				self.occupancy[:, data[0], data[1], data[2]] += self.gamma[i] 
# 				self.total_occupancy[:, data[0], data[1]] += self.gamma[i] 

# 			new_initial = self.gamma[0]
# 			new_tt = np.sum(self.eta, axis = 0) / np.expand_dims(np.sum(self.gamma[:-1], axis = 0), axis = 1)

# 			new_prob = self.occupancy / np.expand_dims(self.total_occupancy,axis = 3)

# 			error =  np.linalg.norm(self.state - new_state) + np.linalg.norm(self.initial - new_initial) + np.linalg.norm(self.tt - new_tt) + np.linalg.norm(self.prob - new_prob) 
			
# 			if error< 1e-7:
# 				print(it, self.time)
# 				return

# 			self.initial = new_initial
# 			self.state  = new_state
# 			self.prob = new_prob
# 			self.tt = new_tt

		
# 	def reset(self):
# 		self.prob = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2))
# 		self.prob[:, :, :,0] = np.random.rand(self.context_num, self.stimuli_num, self.class_num)
# 		self.prob[:, :, :, 1] = 1 - self.prob[:, :, :,0]

# 		self.tt = (1-self.s) / (self.context_num - 1) * np.ones((self.context_num, self.context_num))
# 		for i in range(0, self.context_num):
# 			self.tt[i, i] = self.s 


# 		self.state = np.zeros(self.context_num)
# 		self.initial = np.zeros(self.context_num)
# 		self.initial[0] = self.s
# 		self.state[0] = self.s
# 		for i in range(1, self.context_num):
# 			self.initial[i] = (1 - self.s) / (self.context_num - 1)
# 			self.state[i] = (1 - self.s) / (self.context_num - 1)

# 		self.context = 0
# 		self.action = 0
# 		self.R = 0
# 		self.r = 0
# 		self.time = 0
# 		self.confidence = 0
# 		self.past_choice = [0]
# 		self.trajectory = []



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

