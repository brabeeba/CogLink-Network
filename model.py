import numpy as np
import logging
import scipy.special as special
from util import *
from scipy.special import expit
from scipy.stats import norm
from hmmlearn.hmm import CategoricalHMM

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


class HMMThompsonAgent(object):
	"""docstring for ThompsonDCAgent"""
	def __init__(self, opt, name = "HMM"):
		super(HMMThompsonAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.gamma = opt["gamma"]
		self.prob = np.zeros( (opt["context_num"], opt["stimuli_num"], opt["class_num"], 2))
		self.context_num = opt["context_num"]
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name

		self.past_choice = [0]

		H = 1 / 200
		m = CategoricalHMM(n_components = self.context_num, init_params = "")
		m.startprob_ = np.array([1., 0.])
		m.transmat_ = np.array([[ 1-H, H], [H, 1-H]])

		mat1 = np.zeros((self.class_num, 2))
		mat1[0, 0] = 0.3
		mat1[0, 1] = 0.7
		mat1[1, 0] = 0.7
		mat1[1, 1] = 0.3

		mat2 = 1-mat1

		mat1 /= 2
		mat2 /= 2

		m.emissionprob_ = np.array([mat1.flatten(),mat2.flatten()])

		self.hmm = m
		self.action_arr = []
		self.reward_arr = []
		self.state = np.array([1, 0])
		self.time = 0


	def ev(self):
		return (self.prob[:, :, :, 1] + 1) / (2 + np.sum(self.prob[:, :, : ,:], axis = 3))

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
		return {}
	def forward(self, x):
		self.stimuli = x

		if np.random.random() < self.state[0]:
			sample_state = 0
		else:
			sample_state = 1

		# sample = np.random.beta(  self.state.dot(self.prob[:, x, :, 1]) + 1,   self.state.dot(self.prob[:, x, :, 0]) + 1)
		sample = np.random.beta(  self.prob[sample_state, x, :, 1] + 1,   self.prob[sample_state, x, :, 0] + 1)
		action = np.argmax(sample)

		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action
		self.action_arr.append(action)
		self.time += 1
		
		return action
		
	def update(self, r):
		
		self.prob[:, self.stimuli, self.action, r] += self.state
		self.r = r
		self.reward_arr.append(r)
		data = np.array(self.action_arr) * 2 + np.array(self.reward_arr)
		self.state = self.hmm.predict_proba(data.astype(np.int32).reshape(1, -1))[-1]
		


	



	def reset(self):
		self.prob = np.zeros( (self.opt["context_num"], self.opt["stimuli_num"], self.opt["class_num"], 2))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.past_choice = [0]

		self.action_arr = []
		self.reward_arr = []
		self.state = np.array([1, 0])
		self.time = 0



# class TwoTimeScaleNeuralAgent(object):
# 	"""docstring for ThalamocorticalModel"""
# 	def __init__(self, opt, name = "Thalamocortical Model"):
# 		super(TwoTimeScaleNeuralAgent, self).__init__()
# 		self.opt = opt
# 		self.stimuli_num = opt["stimuli_num"]
# 		self.class_num = opt["class_num"]
# 		self.context_num = opt["context_num"]
# 		self.gamma1 = opt["gamma1"]
# 		self.tau = opt["tau"]
	
# 		self.temperature = opt["temperature"]
# 		self.lr = opt["lr"]
# 		self.a = opt["a"]
# 		self.name = name

# 		self.dt = opt["dt"]
# 		self.a1 = opt["a1"]
# 		self.b1 = opt["b1"]
# 		self.a2 = opt["a2"]
# 		self.b2 = opt["b2"]

# 		self.tau = opt["tau"]
# 		self.eta = opt["eta"]
# 		self.tau1 = opt["tau1"]
# 		self.threshold = opt["threshold"]
# 		self.d_interval = opt["d_interval"]
# 		self.K = opt["K"]
# 		self.learning = opt["learning"]

# 		self.inhibit = opt["inhibit"]
# 		self.d2 = opt["d2"]
# 		self.rescue = opt["rescue"]


# 		self.stimuli = 0

# 		self.thalamus = np.zeros(self.context_num)
	
	

# 		self.trn = 0

# 		self.pfc_core = np.zeros(self.context_num)
		

# 		# self.thalamus[0] = 2 * self.a+ 2
# 		# self.thalamus[1] = 2


# 		if self.d2:
# 			self.thalamus[0] = 4

# 		self.vip = np.zeros(self.context_num)
# 		self.pv = np.zeros(self.context_num)

		
# 		self.ct = 0.5 * np.ones((self.context_num, self.stimuli_num, self.class_num, 2))
# 		self.ct = np.random.random((self.context_num, self.stimuli_num, self.class_num, 2))
# 		for i in range(self.stimuli_num):
# 			self.ct[:, i,  :, 1] = np.eye(self.context_num)
# 			self.ct[:, i,  :, 0] = 1 - np.eye(self.context_num)
# 		self.ct *= 0.8

# 		#self.ct[0] = 0.5 * np.ones((self.stimuli_num, self.class_num, 2))
		
		
		

# 		self.quantile_num = opt["quantile_num"]
		
# 		self.conf = 0

# 		self.prob = np.ones((self.context_num, self.stimuli_num, self.class_num, self.quantile_num))
# 		self.prob *= np.expand_dims((np.arange(self.quantile_num)+1) / float(self.quantile_num), (0, 1, 2))


# 		self.sample_neurons = [np.zeros((self.context_num, self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]



		
	
# 		self.sample_w = -  self.b1 * np.ones((self.quantile_num, self.quantile_num))
# 		for i in range(self.quantile_num):
# 			self.sample_w[i, i] = self.a1

# 		self.value_neurons = np.zeros((self.context_num, self.quantile_num, self.class_num))

		

# 		self.decision_neurons = np.zeros(self.class_num)
	
# 		self.decision_w = -  self.b2 * np.ones((self.class_num, self.class_num))
# 		for i in range(self.class_num):
# 			self.decision_w[i, i] = self.a2

		



# 		self.count = np.zeros( (opt["context_num"], opt["stimuli_num"], opt["class_num"]))
# 		self.count1 = np.zeros((opt["context_num"], opt["stimuli_num"], opt["class_num"]))
# 		self.stimuli = 0
# 		self.action = 0
# 		self.context = 0



# 		self.R = 0
# 		self.r = 0

# 		self.time = 0
# 		self.confidence = 0

# 		self.past_choice = [0]
# 		self.lr = 0


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
# 		return relu(5 + 0.3 * np.log(x))
# 		#return relu(3 + np.log(x))

# 	def h(self, x):
# 		return np.minimum(1, relu(x))

# 	def hf(self, x, a):
# 		return np.minimum(a, relu(x))

# 	def sig(self, x):
# 		return 1 / (1 + np.exp(-10 * x + 5))

# 	#1 / (1 + np.exp(-10 * x + 5)) 

# 	def md_f(self, x):
# 		# return relu(2 /  (1 + np.exp(4*self.a - 4*(x-2)))- 1)
# 		return relu(2 /  (1 + np.exp(4*self.a*self.tau - 4*x))- 1)

# 	def in_f(self, x):
# 		return relu(2 /  (1 + np.exp( - 2*(x +0.25)))-1)



# 	def ev(self):
# 		return np.mean(self.prob, axis = 3)

# 	def get_ev(self):
# 		return np.mean(self.prob, axis = 3)[self.context, :, :]

# 	def get_choice_prob(self):
# 		return sum(self.past_choice) / len(self.past_choice)

# 	def histogram(self):
# 		return {"PFC": self.pfc(), "PFC/MD": self.ct.copy(), "MD": self.thalamus.copy(), "ALM": self.sample_neurons.copy(), "ALM/BG": self.prob.copy(), "BG": self.value_neurons.copy(), "M1": self.decision_neurons.copy(), "PV": self.pv.copy(), "VIP": self.vip.copy()}

# 	def pfc(self):
# 		pfc = np.zeros((self.stimuli_num, self.class_num, 2))
# 		pfc[self.stimuli, self.action, self.r] = 1
# 		return pfc

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
# 		hist["smooth_confidence"] = self.conf
# 		hist["learning_rate"] = self.lr
# 		hist["md-nonlinearity"] = self.md_f(self.thalamus[0])
# 		hist["in-nonlinearity"] = self.in_f(self.vip[0]-self.pv[0])
# 		hist["action"] = self.action
# 		hist["context"] = self.context

# 		return hist

# 	def forward(self, x):
# 		self.time += 1
# 		self.stimuli = x

		

# 		self.sample_neurons = [np.zeros((self.context_num, self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
# 		self.value_neurons = np.zeros((self.context_num, self.quantile_num, self.class_num))
# 		self.decision_neurons = np.zeros(self.class_num)

# 		action_bool = False



# 		for i in range(self.d_interval):

# 			if self.thalamus[0] - self.thalamus[1] >= 0:
# 				self.context = 0
# 			else:
# 				self.context = 1


# 			self.vip += self.dt * self.tau1 * (-self.vip + self.thalamus)
# 			for i in range(self.context_num):
# 				self.pv += self.dt * self.tau1 * -self.pv 
# 				for j in range(self.context_num):
# 					if j != i:
# 						self.pv += self.dt * self.tau1 * self.thalamus[j]

# 			self.confidence = self.sig(np.abs(self.thalamus[0] - self.thalamus[1]) / (2 * self.a * self.tau))
# 			self.conf += self.dt * self.tau1 * (-self.conf + self.confidence)


# 			if self.d2:
# 				gain = 0.85
# 			else:
# 				gain = 1
			
# 			if self.rescue:
# 				rescue = 0.45
# 			else:
# 				rescue = 0

# 			self.pfc_core += self.dt * self.tau * (-self.pfc_core + self.thalamus)
# 			self.thalamus += self.dt * self.tau * (-self.thalamus + self.pfc_core - self.trn + self.g(self.ct[:, self.stimuli, self.action, self.r]))
# 			self.trn += self.dt * self.tau * (-self.trn + relu(np.sum(self.thalamus)))

# 			if self.inhibit:
# 				self.thalamus = np.zeros(self.context_num)
			
# 			if self.learning:

# 				for k in range(self.context_num):


				
# 					self.sample_neurons[x][k] +=  self.dt * 6 * (-self.sample_neurons[x][k] +  self.sample_w.dot(self.h(self.sample_neurons[x][k])) +  (self.K-0.25)*self.b1 + 0.2 * np.random.normal(size = (self.quantile_num, self.class_num)))
# 					# if k == self.context:
# 					# 	self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k] +  relu(self.h(self.sample_neurons[x][k]) * self.prob[k, x, :, :].T * self.conf ))
# 					# else:
# 					# 	self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k])
# 					#self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k] +  relu(self.h(self.sample_neurons[x][k]) * self.prob[k, x, :, :].T * self.md_f(self.vip[k])))
# 					self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k] +  relu(self.h(self.sample_neurons[x][k]) * self.prob[k, x, :, :].T * self.in_f(self.vip[k] - self.pv[k])))
# 				self.decision_neurons += self.dt * 0.1 * self.tau1 * (-self.decision_neurons + relu(self.decision_w.dot(self.decision_neurons)) + 1.0 / self.K * np.sum(self.value_neurons, axis = (0, 1)) )
			
# 			else:
# 				for k in range(self.context_num):
				
# 					self.sample_neurons[x][k] +=  0.03 * (-self.sample_neurons[x][k] +  self.sample_w.dot(self.h(self.sample_neurons[x][k])) +  (self.K-0.25)*self.b1 + 0.2 * np.random.normal(size = (self.quantile_num, self.class_num)))
# 					if k == self.context:
# 						self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k] +  relu(self.h(self.sample_neurons[x][k]) * self.prob[k, x, :, :].T ))
# 					else:
# 						self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k])
				
# 				self.decision_neurons += self.dt * 0.1 * self.tau1 * (-self.decision_neurons + relu(self.decision_w.dot(self.decision_neurons)) + 1.0 / self.K * np.sum(self.value_neurons, axis = (0, 1)) )
			


# 			if np.max(self.decision_neurons) > self.threshold:
# 				action = np.argmax(self.decision_neurons)
# 				action_bool = True
				



# 		if not action_bool:
			
# 			#action = np.random.randint(self.class_num)
# 			#action = np.argmax(self.decision_neurons)
# 			action = np.random.choice(self.class_num, p= special.softmax(15 * self.decision_neurons))
			
# 			# print("Random decision is made {}".format(self.name))
# 			# print(self.decision_neurons, self.threshold, self.context)
# 			# print(np.sort(self.value_neurons[self.context, :, 0])[-4:])
			

# 		if self.learning:
# 			#self.count[self.context, x, action] += self.conf
# 			self.count[:, x, action] +=  0.5 * self.in_f(self.vip-self.pv)
# 		else:
# 			self.count[self.context, x, action] += 1

		
# 		self.past_choice.append(action)
# 		if len(self.past_choice) > 10:
# 			del self.past_choice[0]


# 		self.action = action


# 		if self.time % 200 == 0:
# 			print(self.time)

# 		print(self.thalamus)

# 		return self.action

# 	def update(self, r):
# 		self.r = r

# 		# self.thalamus += -1.0 / self.tau * self.thalamus + self.f(self.tt.dot(self.thalamus)) + self.g(self.ct[:, self.stimuli, self.action, r])
# 		if self.thalamus[0] - self.thalamus[1] >= 0:
# 			self.context = 0
# 		else:
# 			self.context = 1
# 		self.confidence = self.sig(np.abs(self.thalamus[0] - self.thalamus[1]) / (2 * self.a * self.tau))
	

# 		if self.learning:
# 			#self.prob[self.context, self.stimuli, self.action, :-1] += (r - self.prob[self.context, self.stimuli, self.action, :-1]) * self.conf / (3+self.count[self.context, self.stimuli, self.action])
# 			self.prob[:, self.stimuli, self.action, :-1] += (r - self.prob[:, self.stimuli, self.action, :-1]) * np.expand_dims(self.in_f(self.vip-self.pv), 1) / (14+np.expand_dims(self.count[:, self.stimuli, self.action], 1))
			
# 			#self.count1[self.context, self.stimuli, self.action] += self.confidence
# 			self.count1[:, self.stimuli, self.action] += self.md_f(self.thalamus)

# 			inputs = np.zeros(2)
# 			inputs[r] = 1
# 			#self.ct[self.context, self.stimuli, self.action, r] +=   self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])
# 			#self.lr = self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])

# 			self.ct[:, self.stimuli, self.action, r] +=   self.md_f(self.thalamus) * np.minimum(1 / (4+ self.count1[self.context, self.stimuli, self.action]), 0.075)
# 			self.lr = self.confidence * np.minimum(1 / (4+ self.count1[self.context, self.stimuli, self.action]), 0.075)
			
# 			self.ct[:, self.stimuli, self.action, :] = self.ct[:, self.stimuli, self.action, :] / (np.sum(self.ct[:, self.stimuli, self.action, :], axis = 1, keepdims = True)+1e-8 ) 

# 		else:
# 			self.prob[self.context, self.stimuli, self.action, :-1] += (r - self.prob[self.context, self.stimuli, self.action, :-1]) / (2+self.count[self.context, self.stimuli, self.action])
		
# 			self.count1[self.context, self.stimuli, self.action] += 1

# 			inputs = np.zeros(2)
# 			inputs[r] = 1
# 			self.ct[self.context, self.stimuli, self.action, r] +=   1 / (2+ self.count1[self.context, self.stimuli, self.action])
# 			self.lr = 1 / (2+ self.count1[self.context, self.stimuli, self.action])
			
# 			self.ct[:, self.stimuli, self.action, :] = self.ct[:, self.stimuli, self.action, :] / (np.sum(self.ct[:, self.stimuli, self.action, :], axis = 1, keepdims = True)+1e-8 ) 

	

# 	def reset(self):
# 		self.thalamus = np.zeros(self.context_num)
		
	

# 		self.trn = 0

# 		self.pfc_core = np.zeros(self.context_num)
	

# 		self.ct = 0.5 * np.ones((self.context_num, self.stimuli_num, self.class_num, 2))
# 		self.ct = np.random.random((self.context_num, self.stimuli_num, self.class_num, 2))
# 		for i in range(self.stimuli_num):
# 			self.ct[:, i,  :, 0] = np.eye(self.context_num)
# 			self.ct[:, i,  :, 1] = 1 - np.eye(self.context_num)
# 		#self.ct[0] = 0.5 * np.ones((self.stimuli_num, self.class_num, 2))
		

# 		# self.prob = np.ones(( self.context_num, self.stimuli_num, self.class_num, self.quantile_num))
		
# 		self.sample_neurons = [np.zeros((self.context_num, self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
# 		self.value_neurons = np.zeros((self.context_num, self.quantile_num, self.class_num))
# 		self.decision_neurons = np.zeros(self.class_num)
		
# 		self.prob = np.ones((self.context_num, self.stimuli_num, self.class_num, self.quantile_num))
# 		self.prob *= np.expand_dims((np.arange(self.quantile_num)+1) / float(self.quantile_num), (0, 1, 2))



# 		self.count = np.zeros( (self.context_num, self.stimuli_num, self.class_num))
# 		self.count1 = np.zeros((self.context_num, self.stimuli_num, self.class_num))
# 		self.stimuli = 0
# 		self.action = 0
# 		self.context = 0

# 		self.R = 0
# 		self.r = 0
# 		self.time = 0
# 		self.confidence = 0
# 		self.past_choice = [0]

class GittinsIndex(object):
	"""docstring for GittinsIndex"""
	def __init__(self, opt, name = "Gittins Index"):
		super(GittinsIndex, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		
		self.alpha = np.ones((self.stimuli_num, self.class_num), dtype = np.int64)
		self.beta = np.ones((self.stimuli_num, self.class_num), dtype = np.int64)
		self.trial = 0

		self.max_trial = opt["max_trial"]
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name
		self.value_table = np.zeros((self.max_trial, self.max_trial, self.max_trial))
		self.compute_value_table()
		print(self.value_table[:, :, 0])

		print(self.value_table[:, :, 1])		

		print("finish computing value table")




		self.past_choice = [0]


	def compute_value_table(self):
		"""
		Precompute the value table for all horizons up to max_horizon.
		"""
		for h in range(self.max_trial):
			for alpha in range(self.max_trial):
				for beta in range(self.max_trial):
					self.value_table[alpha, beta, h] = self.compute_value(alpha + 1, beta + 1, h + 1)

	def compute_value(self, alpha, beta, horizon):
		"""
		Recursively compute the value for a given alpha, beta, and horizon.
		Args:
			alpha (int): Prior successes.
			beta (int): Prior failures.
			horizon (int): Remaining time horizon.
		Returns:
			float: Value for the given state and horizon.
		"""
		if horizon == 0:
			return 0

		immediate_reward = alpha / (alpha + beta)  # Posterior mean
		continue_value = (
			(alpha / (alpha + beta)) * (1 + self.lookup_value(alpha + 1, beta, horizon - 1))
			+ (beta / (alpha + beta)) * self.lookup_value(alpha, beta + 1, horizon - 1)
		)

		return max(immediate_reward, continue_value)

	def lookup_value(self, alpha, beta, horizon):
		"""
		Lookup precomputed value in the table or return 0 if horizon is 0.
		"""
		if horizon == 0 or alpha == self.max_trial + 1 or beta == self.max_trial + 1:
			return 0

		return self.value_table[alpha-1, beta-1, horizon-1]

	def gittins_index(self, alpha, beta, horizon):
		"""
		Retrieve the Gittins index for a given alpha, beta, and horizon.
		Args:
			alpha (int): Prior successes.
			beta (int): Prior failures.
			horizon (int): Remaining time horizon.
		Returns:
			float: Gittins index for the given state and horizon.


		"""
		return self.lookup_value(alpha, beta, horizon)

	def ev(self):
		return (self.alpha + 1) / (2 + self.alpha + self.beta)

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
		remaining_trial = self.max_trial - self.trial


		gittins_indices = [ self.gittins_index(self.alpha[x, i], self.beta[x, i], remaining_trial) for i in range(self.class_num)]

		
		action = np.argmax(gittins_indices)

		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action
		self.trial += 1

		
		return action
		
	def update(self, r):
		if r == 1:
			self.alpha[self.stimuli, self.action] += 1
		else:
			self.beta[self.stimuli, self.action] += 1



	def reset(self):
		self.alpha = np.ones((self.stimuli_num, self.class_num), dtype = np.int64)
		self.beta = np.ones((self.stimuli_num, self.class_num), dtype = np.int64)
		self.trial = 0
		

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

		self.md_learning = opt["md_learning"]
		self.in_learning = opt["in_learning"]
		
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
		self.fixmd = opt["fixmd"]
	






		self.stimuli = 0

		self.thalamus = np.zeros(self.context_num)
		self.thalamus[0] =  8
		self.thalamus[1] = 4
	

		# self.thalamus[0] = 2 * self.a+ 2
		# self.thalamus[1] = 2

		# self.thalamus = np.zeros(self.context_num)
		# self.thalamus[0] = 4

		self.pfc_core = np.zeros(self.context_num)
		self.pfc_core[0] =  8
		self.pfc_core[1] = 4

		self.trn = 12

		

		if self.d2:
			self.thalamus[0] = 4
			self.pfc_core[0] =  4

		self.vip = np.zeros(self.context_num)
		self.pv = np.zeros(self.context_num)

		self.ct = np.random.normal(0.5, 0.05, (self.context_num, self.stimuli_num, self.class_num, 2))

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

	def hf(self, x, a):
		return np.minimum(a, relu(x))

	def sig(self, x):
		return 1 / (1 + np.exp(-10 * x + 5))

	#1 / (1 + np.exp(-10 * x + 5)) 

	def md_f(self, x):
		# return relu(2 /  (1 + np.exp(4*self.a - 4*(x-2)))- 1)
		return relu(2 /  (1 + np.exp(4*self.a*self.tau - 4*(x-4)))- 1)


	def in_f(self, x):
		return 1 /  (1 + np.exp( -  2 * (x  -0.5)))
		#return relu(2 /  (1 + np.exp( - 2*(x / 4 +0.5)))-1)



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

			
			if not self.fixmd:
				self.thalamus += self.dt * 0.2 * (-1.0 / self.tau * self.thalamus +  gain * self.f(self.pfc_core * self.s * 2 - self.trn * self.s) +  gain * self.g(self.ct[:, self.stimuli, self.action, self.r])  + rescue)
				#self.thalamus += self.dt * self.tau * (-self.thalamus + self.hf(gain * self.pfc_core + gain * self.g(self.ct[:, self.stimuli, self.action, self.r]) - self.trn + rescue , 4))
				self.pfc_core += self.dt * self.tau * (-self.pfc_core + self.thalamus)
				self.trn += self.dt * self.tau * (-self.trn + np.sum(self.thalamus))

		
			if self.inhibit:
				self.thalamus = np.zeros(self.context_num)


			
			

			for k in range(self.context_num):


			
				self.sample_neurons[x][k] +=  self.dt * 6 * (-self.sample_neurons[x][k] +  self.sample_w.dot(self.h(self.sample_neurons[x][k])) +  (self.K-0.25)*self.b1 + 0.2 * np.random.normal(size = (self.quantile_num, self.class_num)))
				# if k == self.context:
				# 	self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k] +  relu(self.h(self.sample_neurons[x][k]) * self.prob[k, x, :, :].T * self.conf ))
				# else:
				# 	self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k])
				self.value_neurons[k] += self.dt * self.tau1 * (-self.value_neurons[k] +  relu(self.h(self.sample_neurons[x][k]) * self.prob[k, x, :, :].T * self.in_f(self.vip[k] - self.pv[k])))
			self.decision_neurons += self.dt * 0.1 * self.tau1 * (-self.decision_neurons + relu(self.decision_w.dot(self.decision_neurons)) + 1.0 / self.K * np.sum(self.value_neurons, axis = (0, 1)) )
		
		


			if np.max(self.decision_neurons) > self.threshold:

				action = np.argmax(self.decision_neurons)
				action_bool = True

		self.stimuli = x

				



		if not action_bool:
			
			#action = np.random.randint(self.class_num)
			#action = np.argmax(self.decision_neurons)
		
			action = np.random.choice(self.class_num, p= special.softmax( self.decision_neurons))
			
			print("Random decision is made {}".format(self.name))
			print(self.decision_neurons, self.threshold, self.context, action)
			# print(np.sort(self.value_neurons[self.context, :, 0])[-4:])
			

		if self.in_learning:
			#self.count[self.context, x, action] += self.conf
			self.count[:, x, action] +=  0.5 * self.in_f(self.vip-self.pv)
		else:
			self.count[self.context, x, action] += 1

		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]

		#print(self.time, self.in_f(self.vip[0] - self.pv[0]), self.in_f(self.vip[1] - self.pv[1]))




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
	

					#self.prob[self.context, self.stimuli, self.action, :-1] += (r - self.prob[self.context, self.stimuli, self.action, :-1]) * self.conf / (3+self.count[self.context, self.stimuli, self.action])
		if self.in_learning:
			self.prob[:, self.stimuli, self.action, :] += (r - self.prob[:, self.stimuli, self.action, :]) * np.expand_dims(self.in_f(self.vip-self.pv), 1) * np.maximum(1 / (12+np.expand_dims(self.count[:, self.stimuli, self.action], 1)), 0.005)
		else:
			self.prob[:, self.stimuli, self.action, :] += (r - self.prob[:, self.stimuli, self.action, :]) * np.expand_dims(relu(self.thalamus-4)/4, 1)* np.maximum(1 / (12+np.expand_dims(self.count[:, self.stimuli, self.action], 1)), 0.005)
		
		#self.count1[self.context, self.stimuli, self.action] += self.confidence

		if self.md_learning:
			self.count1[:, self.stimuli, self.action] += self.md_f(self.thalamus)
			self.ct[:, self.stimuli, self.action, r] +=   self.md_f(self.thalamus) * np.maximum(1/ (4+ self.count1[:, self.stimuli, self.action]), 0.0)
			self.lr = self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])
		
			self.ct[:, self.stimuli, self.action, :] = self.ct[:, self.stimuli, self.action, :] / (np.sum(self.ct[:, self.stimuli, self.action, :], axis = 1, keepdims = True)+1e-8 ) 

		else:
			self.count1[:, self.stimuli, self.action] += 1
			self.ct[:, self.stimuli, self.action, r] +=  ((self.thalamus - 4) / 4) * np.maximum(1/ (4+ self.count1[:, self.stimuli, self.action]), 0.0)
			self.ct[:, self.stimuli, self.action, r] = relu(self.ct[:, self.stimuli, self.action, r])
			self.lr = self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])
		
			self.ct[:, self.stimuli, self.action, :] = self.ct[:, self.stimuli, self.action, :] / (np.sum(self.ct[:, self.stimuli, self.action, :], axis = 1, keepdims = True)+1e-8 ) 


		inputs = np.zeros(2)
		inputs[r] = 1
		#self.ct[self.context, self.stimuli, self.action, r] +=   self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])
		#self.lr = self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])

		# self.ct[:, self.stimuli, self.action, r] +=   self.md_f(self.thalamus) / (4+ self.count1[:, self.stimuli, self.action])
		# self.lr = self.confidence / (4+ self.count1[self.context, self.stimuli, self.action])
		
		# self.ct[:, self.stimuli, self.action, :] = self.ct[:, self.stimuli, self.action, :] / (np.sum(self.ct[:, self.stimuli, self.action, :], axis = 1, keepdims = True)+1e-8 ) 

		

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
		self.thalamus[0] =  8
		self.thalamus[1] = 4
	

		# self.thalamus[0] = 2 * self.a+ 2
		# self.thalamus[1] = 2

		# self.thalamus = np.zeros(self.context_num)
		# self.thalamus[0] = 4

		self.pfc_core = np.zeros(self.context_num)
		self.pfc_core[0] =  8
		self.pfc_core[1] = 4

		self.trn = 12

		self.ct = np.random.normal(0.5, 0.05, (self.context_num, self.stimuli_num, self.class_num, 2))

		
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

class DQNRL(object):
	"""docstring for DQNRL"""
	def __init__(self, opt, name = "DQN RL"):
		super(DQNRL, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.hidden_size = opt["hidden_size"]
		self.epsilon = opt['epsilon'] * np.ones(self.stimuli_num)
		self.lr = opt['lr']

		self.W1 = self.xavier_init((self.hidden_size, self.stimuli_num)) 
		self.b1 = np.zeros(self.hidden_size)
		self.W2 = self.xavier_init((self.class_num, self.hidden_size))
		self.b2 = np.zeros(self.class_num)




		
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name

		self.past_choice = [0]

	def xavier_init(self, shape):
		limit = np.sqrt(6 / (shape[0] + shape[1]))
		return np.random.uniform(-limit, limit, shape)


	def ev(self):
		result = []
		for i in range(self.stimuli_num):
			state = np.zeros(self.stimuli_num)
			state[i] = 1
			hidden = np.maximum(0, self.W1.dot(state) + self.b1)
			q_values = self.W2.dot(hidden) + self.b2
			result.append(q_values)

		return np.array(result)

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

	def get_q(self, x):
		state = np.zeros(self.stimuli_num)
		state[x] = 1
		hidden = np.maximum(0, self.W1.dot(state) + self.b1)
		return self.W2.dot(hidden) + self.b2


	def forward(self, x):
		self.stimuli = x
		state = np.zeros(self.stimuli_num)
		state[x] = 1
		self.state = state
		self.hidden = np.maximum(0, self.W1.dot(state) + self.b1)
		self.q_values = self.W2.dot(self.hidden) + self.b2

		if np.random.random() < 1./self.epsilon[self.stimuli]:
			action = np.random.randint(0, self.class_num)
		else:
			action = np.argmax(self.q_values)

		self.current_action = action
		if self.stimuli == 0:
			self.action1 = action
		else:
			self.action2 = action
			self.action = self.action1 * 2 + self.action2


		self.epsilon[self.stimuli] += 0.2


		return self.current_action
		
	def update(self, r, next_s):
		self.r = r

		target_q_value = self.q_values.copy()
		if next_s is None:
			target_q_value[self.current_action] = r
		else:
			target_q_value[self.current_action] = r + np.max(self.get_q(next_s))

		grad_output = 2 * (self.q_values - target_q_value)
		grad_W2 = np.outer(grad_output, self.hidden)
		grad_b2 = grad_output

		grad_hidden = self.W2.T.dot(grad_output) * ((self.W1.dot(self.state) + self.b1) > 0)
		grad_W1 = np.outer(grad_hidden, self.state)
		grad_b1 = grad_hidden

		self.W2 -= self.lr * grad_W2
		self.b2 -= self.lr * grad_b2
		self.W1 -= self.lr * grad_W1
		self.b1 -= self.lr * grad_b1



	def reset(self):
		self.W1 = self.xavier_init((self.hidden_size, self.stimuli_num)) 
		self.b1 = np.zeros(self.hidden_size)
		self.W2 = self.xavier_init((self.class_num, self.hidden_size))
		self.b2 = np.zeros(self.class_num)
		self.epsilon = self.opt['epsilon'] * np.ones(self.stimuli_num)
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.past_choice = [0]


class DQN(object):
	"""docstring for DQN"""
	def __init__(self, opt, name = "Bayesian RL"):
		super(DQN, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.hidden_size = opt["hidden_size"]
		self.epsilon = opt['epsilon'] * np.ones(self.stimuli_num)
		self.lr = opt['lr']
		self.volatile = opt['volatile']

		self.W1 = self.xavier_init((self.hidden_size, self.stimuli_num)) 
		self.b1 = np.zeros(self.hidden_size)
		self.W2 = self.xavier_init((self.class_num, self.hidden_size))
		self.b2 = np.zeros(self.class_num)


		
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name

		self.past_choice = [0]

	def xavier_init(self, shape):
		limit = np.sqrt(6 / (shape[0] + shape[1]))
		return np.random.uniform(-limit, limit, shape)


	def ev(self):
		result = []
		for i in range(self.stimuli_num):
			state = np.zeros(self.stimuli_num)
			state[i] = 1
			hidden = np.maximum(0, self.W1.dot(state) + self.b1)
			q_values = self.W2.dot(hidden) + self.b2
			result.append(q_values)

		return np.array(result)

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
		state = np.zeros(self.stimuli_num)
		state[x] = 1
		self.state = state
		self.hidden = np.maximum(0, self.W1.dot(state) + self.b1)
		self.q_values = self.W2.dot(self.hidden) + self.b2

		if np.random.random() <  1./self.epsilon[self.stimuli]:
			action = np.random.randint(0, self.class_num)
		else:
			action = np.argmax(self.q_values)

		self.epsilon[self.stimuli] += 0.2

		




		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action
		print(x, self.action)
		return action
		
	def update(self, r):
		self.r = r
		target_q_value = self.q_values.copy()
		target_q_value[self.action] = r

		grad_output = 2 * (self.q_values - target_q_value)
		grad_W2 = np.outer(grad_output, self.hidden)
		grad_b2 = grad_output

		grad_hidden = self.W2.T.dot(grad_output) * ((self.W1.dot(self.state) + self.b1) > 0)
		grad_W1 = np.outer(grad_hidden, self.state)
		grad_b1 = grad_hidden

		self.W2 -= self.lr * grad_W2
		self.b2 -= self.lr * grad_b2
		self.W1 -= self.lr * grad_W1
		self.b1 -= self.lr * grad_b1



	def reset(self):
		self.W1 = self.xavier_init((self.hidden_size, self.stimuli_num)) 
		self.b1 = np.zeros(self.hidden_size)
		self.W2 = self.xavier_init((self.class_num, self.hidden_size))
		self.b2 = np.zeros(self.class_num)
		self.epsilon = self.opt['epsilon'] * np.ones(self.stimuli_num)
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.past_choice = [0]


class ThompsonRLAgent(object):
	"""docstring for ThompsonDCAgent"""
	def __init__(self, opt, name = "Thompson RL"):
		super(ThompsonRLAgent, self).__init__()
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
		result = []
		result.append((self.prob[0, 0, 1] + 1) / (2 + np.sum(self.prob[0, 0 ,:], axis = 0)) + (self.prob[1, 0, 1] + 1) / (2 + np.sum(self.prob[1, 0 ,:], axis = 0)))
		result.append((self.prob[0, 0, 1] + 1) / (2 + np.sum(self.prob[0, 0 ,:], axis = 0)) + (self.prob[1, 1, 1] + 1) / (2 + np.sum(self.prob[1, 1 ,:], axis = 0)))
		result.append((self.prob[0, 1, 1] + 1) / (2 + np.sum(self.prob[0, 1 ,:], axis = 0)) + (self.prob[2, 0, 1] + 1) / (2 + np.sum(self.prob[2, 0 ,:], axis = 0)))
		result.append((self.prob[0, 1, 1] + 1) / (2 + np.sum(self.prob[0, 1 ,:], axis = 0)) + (self.prob[2, 1, 1] + 1) / (2 + np.sum(self.prob[2, 1 ,:], axis = 0)))
		return result

	def get_ev(self):
		return self.ev()

	def get_choice_prob(self):
		return sum(self.past_choice) / len(self.past_choice)

	def scalars(self):
		hist = {}
		ev = self.ev()
		

		hist["choice_prob"] = self.get_choice_prob()
		return hist
	def histogram(self):
		return {"prob": self.prob, "ev": self.ev()}

	def forward(self, x):
		self.stimuli = x
		if self.stimuli == 0:
			sample0 = np.random.beta(self.prob[0, :, 1] + 1, self.prob[0, :, 0] + 1)
			sample1 = np.random.beta(self.prob[1, :, 1] + 1, self.prob[1, :, 0] + 1)
			sample2 = np.random.beta(self.prob[2, :, 1] + 1, self.prob[2, :, 0] + 1)
			action_value = [sample0[0] + sample1[0], sample0[0] + sample1[1], sample0[1] + sample2[0], sample0[1] + sample2[1]]
			self.action = np.argmax(action_value)

			self.action1 = int(self.action / 2)
			self.action2 = self.action - 2 * self.action1


		
			self.past_choice.append(self.action)
			if len(self.past_choice) > 10:
				del self.past_choice[0]

		if self.stimuli == 0:
			self.current_action = self.action1
			
		else:
			self.current_action = self.action2
		
		return self.current_action

		
	def update(self, r, next_s):
		self.prob[self.stimuli, self.current_action, :] *= self.gamma
		self.prob[self.stimuli, self.current_action, r] += 1

		if self.stimuli == 0:
			self.r1 = r
		else:
			self.r2 = r
			self.r = self.r1 + self.r2
	



	def reset(self):
		self.prob = np.zeros( (self.opt["stimuli_num"], self.opt["class_num"], 2))
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.past_choice = [0]


			
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
		self.learned = opt["learned"]

		self.s = 0.9

		self.prob = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2))
		self.prob[:, :, :,0] = np.random.rand(self.context_num, self.stimuli_num, self.class_num)
		self.prob[:, :, :, 1] = 1 - self.prob[:, :, :,0]
	
		self.tt = (1-self.s) / (self.context_num - 1) * np.ones((self.context_num, self.context_num))
		for i in range(0, self.context_num):
			self.tt[i, i] = self.s 

		if self.learned:
			self.tt = (1- 1./opt["block_size"]) / (self.context_num - 1) * np.ones((self.context_num, self.context_num))
			for i in range(0, self.context_num):
				self.tt[i, i] = 1./opt["block_size"]

		else:
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

		print(self.state)
		print(self.prob)

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
				self.initial = new_initial
				self.state  = new_state
				self.prob = new_prob
				if not self.learned:
					self.tt = new_tt
				return

			self.initial = new_initial
			self.state  = new_state
			self.prob = new_prob
			if not self.learned:
				self.tt = new_tt

		
	def reset(self):
		self.prob = np.zeros((self.context_num, self.stimuli_num, self.class_num, 2))
		self.prob[:, :, :,0] = np.random.rand(self.context_num, self.stimuli_num, self.class_num)
		self.prob[:, :, :, 1] = 1 - self.prob[:, :, :,0]

		if self.learned:
			self.tt = (1- 1./self.opt["block_size"]) / (self.context_num - 1) * np.ones((self.context_num, self.context_num))
			for i in range(0, self.context_num):
				self.tt[i, i] = 1./self.opt["block_size"]

		else:
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
					self.prob[s,c, :] = np.random.normal(0.5, 0.1, self.quantile_num)
			
		

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
			action = np.random.choice(self.class_num, p= special.softmax(15 * self.decision_neurons))
			print("Random decision is made {}".format(self.name))
			print(self.decision_neurons, self.threshold)
			# print(np.sort(self.sample_neurons[x][:, 0])[-4:])



		self.count[x, action] += 1
		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]


		self.action = action
		return action
		
	def update(self, r):
		self.r = r
		init = 14
		
		if self.uniform:
			self.prob[self.stimuli, self.action, :] += (r - self.prob[self.stimuli, self.action, :] )  / (init**(1/self.decay)+  self.count[self.stimuli, self.action])**self.decay
		else:
			self.prob[self.stimuli, self.action, :] += (r - self.prob[self.stimuli, self.action, :] )  / (init**(1/self.decay)+  self.count[self.stimuli, self.action])**self.decay
		



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


class NeuralQuantileRLAgent(object):
	"""docstring for QuantileAgent"""
	def __init__(self, opt, name = "Corticostriatal RL"):
		super(NeuralQuantileRLAgent, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.quantile_num = opt["quantile_num"]
		self.prob = np.ones((1, 1, opt["quantile_num"])) * np.random.rand( opt["stimuli_num"], opt["class_num"], 1)
		self.r = 1
		
		self.K = opt["K"]
		self.init1 = 2.5
		self.init2 = 1.5
		
		self.uniform = opt["uniform"]
		self.uniform_init = opt["uniform_init"]
		if not self.uniform_init:
			# quantile = (np.arange(self.quantile_num)+1) * 1.0 / self.quantile_num
			# self.prob = self.prob * np.expand_dims(quantile, (0, 1)) 

			self.prob = np.ones((self.stimuli_num, self.class_num, self.quantile_num))
			self.prob *=  np.expand_dims((np.arange(self.quantile_num)+1) / float(self.quantile_num), (0, 1))
			self.prob[0] *= self.init1
			self.prob[1:] *= self.init2
			

			self.prob_q = np.ones((self.quantile_num, self.stimuli_num))
			self.prob_q *= np.expand_dims( (np.arange(self.quantile_num)+1) / float(self.quantile_num), 1)
			self.prob_q[0] *= self.init1
			self.prob_q[1:] *= self.init2
			

		else:
			mean = np.random.rand(self.stimuli_num, self.class_num)
			for s in range(self.stimuli_num):
				for c in range(self.class_num):
					self.prob[s,c, :] = np.random.normal(0.5, 0.05, self.quantile_num)
				self.prob_q[:, s] = np.random.normal(0.5, 0.05, self.quantile_num)
			
		

		self.dt = opt["dt"]
		self.sample_neurons = [np.zeros((self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]

		self.stimuli_neuron = np.zeros((self.quantile_num, self.stimuli_num))
		self.q_neuron = 0

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
		self.count_v = np.zeros(self.stimuli_num)
		self.stimuli = 0
		self.action = 0
		self.context = 0
		self.name = name

		self.past_choice = [0]
		self.last_stimuli = None
		self.last_action = None


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
		return {"prob": self.prob, "prob_q": self.prob_q}
	def forward(self, x):
		self.stimuli = x

		self.sample_neurons = [np.zeros((self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
		self.value_neurons = np.zeros((self.quantile_num, self.class_num))
		self.decision_neurons = np.zeros(self.class_num)
		self.stimuli_neuron = np.zeros((self.quantile_num, self.stimuli_num))
		self.q_neuron = 0

		for j in range(self.d_interval):
			
			self.sample_neurons[x] += 0.03 * (-self.sample_neurons[x] +  self.sample_w.dot(self.f(self.sample_neurons[x])) + (self.K-0.25)*self.b1 + 0.2 * np.random.normal(size = (self.quantile_num, self.class_num)))
		
			self.value_neurons += self.dt  * self.tau * (-self.value_neurons +  relu(self.f(self.sample_neurons[x]) * self.prob[x].T))
			self.decision_neurons += self.dt *0.1 * self.tau * (-self.decision_neurons + relu(self.decision_w.dot(self.decision_neurons)) + 1.0 / self.K * np.sum(self.value_neurons, axis = 0))
			
			if np.max(self.decision_neurons) > self.threshold:
				action = np.argmax(self.decision_neurons)

				break
		

		if np.max(self.decision_neurons) <= self.threshold:
			action = np.random.choice(self.class_num, p= special.softmax(10 * self.decision_neurons))
			print("Random decision is made {}".format(self.name))
			print(self.decision_neurons, action)
			# print(np.sort(self.sample_neurons[x][:, 0])[-4:])


		
		self.count[x, action] += 0.3
		self.count_v[x] += 0.3


		
		self.past_choice.append(action)
		if len(self.past_choice) > 10:
			del self.past_choice[0]

		if self.stimuli == 0:
			self.action1 = action
		else:
			self.action2 = action

			self.action = self.action1 * 2 + self.action2
			
			
			
		self.current_action = action
		#print(np.mean(self.prob, axis = -1))



		print(np.mean(self.prob, axis = -1))
		return self.current_action
		
	def update(self, r, next_s):

		if next_s is not None:
			for j in range(int(self.d_interval / 3)):
				self.stimuli_neuron[:, next_s] += 0.03 * (-self.stimuli_neuron[:, next_s] +  self.sample_w.dot(self.f(self.stimuli_neuron[:, next_s])) + (1-0.25)*self.b1 + 0.2 * np.random.normal(size = (self.quantile_num)))
				self.q_neuron += self.dt * self.tau * (-self.q_neuron + self.f(self.stimuli_neuron).flatten().dot(self.prob_q.flatten()))

				
			self.v_state = np.mean(self.prob_q[:, next_s])

			

		self.r = r
		init = 7
		decay_f = 1
		# print(self.stimuli, self.current_action, np.mean(self.prob[self.stimuli, self.current_action, :]))
		# print(self.stimuli, np.mean(self.prob_q[:, self.stimuli]) )

		if next_s is not None:
			
			self.prob[self.stimuli, self.current_action, :] += (self.r + self.v_state  - self.prob[self.stimuli, self.current_action, :] )  / (init**(1/(self.decay * decay_f))+  self.count[self.stimuli, self.current_action])**(self.decay *decay_f)
			self.prob_q[:, self.stimuli] += (self.r + self.v_state  - self.prob_q[:, self.stimuli] )  * 0.1
			
		else:
			
			self.prob[self.stimuli, self.current_action, :] += (self.r  - self.prob[self.stimuli, self.current_action, :] )  / (7**(1/(self.decay * decay_f))+  self.count[self.stimuli, self.current_action])**(self.decay * decay_f)
			self.prob_q[:, self.stimuli] += (self.r  - self.prob_q[:, self.stimuli] ) / (7**(1/(self.decay))+  self.count_v[self.stimuli])**(self.decay)
		
			
		



	def reset(self):
		self.prob = np.ones((1, 1, self.quantile_num)) * np.random.rand( self.stimuli_num, self.class_num, 1)
		
		self.sample_neurons = [np.zeros((self.quantile_num, self.class_num)) for _ in range(self.stimuli_num)]
		self.value_neurons = np.zeros((self.quantile_num, self.class_num))
		self.decision_neurons = np.zeros(self.class_num)
		self.stimuli_neuron = np.zeros((self.quantile_num, self.stimuli_num))
		self.q_neuron = 0
		self.last_stimuli = None
		self.last_action = None
		
		if not self.uniform_init:
			# quantile = (np.arange(self.quantile_num)+1) * 1.0 / self.quantile_num
			# self.prob = self.prob * np.expand_dims(quantile, (0, 1)) 

			self.prob = np.ones((self.stimuli_num, self.class_num, self.quantile_num))
			self.prob *=  np.expand_dims((np.arange(self.quantile_num)+1) / float(self.quantile_num), (0, 1))
			self.prob[0] *= self.init1
			self.prob[1:] *= self.init2
			

			self.prob_q = np.ones((self.quantile_num, self.stimuli_num))
			self.prob_q *=  np.expand_dims((np.arange(self.quantile_num)+1) / float(self.quantile_num), 1)
			self.prob_q[0] *= self.init1
			self.prob_q[1:] *= self.init2
			

		else:
			mean = np.random.rand(self.stimuli_num, self.class_num)
			for s in range(self.stimuli_num):
				for c in range(self.class_num):
					self.prob[s,c, :] = np.random.normal(0.5, 0.05, self.quantile_num)
				self.prob_q[:, s] = np.random.normal(0.5, 0.05, self.quantile_num)
			

		


		self.count = np.zeros( (self.stimuli_num, self.class_num))
		self.count_v = np.zeros(self.stimuli_num)
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

