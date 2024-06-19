import numpy as np
from util import one_hot, save_dict
import matplotlib.pyplot as plt
from model import *

class Experiment(object):
	"""docstring for Experiment"""
	def __init__(self, opt):
		super(Experiment, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.time = 0

		self.total_reward = 0
		self.total_regret = 0

		self.switch_time = {}


	def run(self, agent, task, writer):
		last = False

		if task.name == "DynamicBandit" or task.name == "UnstructuredBandit":
			self.switch = int(self.opt["max_trial"] / self.opt["block_size"]) - 1
			#print("hi")
		elif task.name == "VolatileBandit":
			self.switch = int(self.opt["max_trial"] / (2 * self.opt["block_size"])) +  int(self.opt["max_trial"] / (self.opt["block_size"])) - 1
		
		switch_bool = {}
		self.switch_time[agent.name] = np.zeros(self.switch)
		switch_bool[agent.name] = [False for _ in range(self.switch)]
		print(self.switch)


		S = np.zeros(300)



		while not last:
			a = agent.forward(task.stimuli)
			r, R, last, next_s = task.step(a)
			agent.update(r)


			self.total_reward += r
			self.total_regret += R
			writer.add_scalar("reward", self.total_reward, task.trial)
			writer.add_scalar("regret", self.total_regret, task.trial)


			scalars = agent.scalars()
			
			for k in scalars:
				writer.add_scalar(k, scalars[k], task.trial)

			if task.name == "DynamicBandit":
				idx = int((task.trial - 1) / self.opt["block_size"]) - 1

				if idx > 1:
					if not switch_bool[agent.name][idx-2]:
						self.switch_time[agent.name][idx-2] += self.opt["block_size"]
						switch_bool[agent.name][idx - 2] = True
				
				if idx > -1 and not switch_bool[agent.name][idx]:
					new_a = (idx+1) % 2

					if new_a == 1:
						if agent.get_choice_prob() >= 0.8:
							self.switch_time[agent.name][idx] += task.trial  - (idx + 1) * self.opt["block_size"] 
							switch_bool[agent.name][idx] = True

					else:
						if agent.get_choice_prob() <= 0.2:
							self.switch_time[agent.name][idx] += task.trial  - (idx + 1) * self.opt["block_size"] 
							switch_bool[agent.name][idx] = True
							

		# 	if task.name == "VolatileBandit":
				
		# 		if task.trial < self.max_trial / 2:
		# 			idx = int((task.trial - 1) / self.opt["block_size"]) - 1
		# 		else:
		# 			idx = int(self.max_trial / (2 * self.opt["block_size"] )) + int((task.trial - self.max_trial / 2 - 1)*2 / self.opt["block_size"]) - 1

		# 		if idx > 1:
		# 			if not switch_bool[agent.name][idx-2]:
		# 				if task.trial < self.max_trial / 2:
		# 					self.switch_time[agent.name][idx-2] += self.opt["block_size"]
		# 				else:
		# 					self.switch_time[agent.name][idx-2] += self.opt["block_size"]/2
		# 				switch_bool[agent.name][idx - 2] = True
				
		# 		if idx > -1 and not switch_bool[agent.name][idx]:

		# 			new_a = (idx+1) % 2


		# 			if task.trial < self.max_trial / 2:
		# 				block = (idx + 1) * self.opt["block_size"] 
		# 			else:
		# 				block = self.max_trial / 2 +  (idx + 1 - int(self.max_trial / (self.opt["block_size"] * 2))) * (self.opt["block_size"] / 2)
					
		# 			if new_a == 1:
		# 				if agent.get_choice_prob() >= 0.9:
						
		# 					self.switch_time[agent.name][idx] += task.trial  - block
		# 					self.switch_data[agent.name][i, idx] = task.trial  - block
		# 					switch_bool[agent.name][idx] = True
							

		# 			else:
		# 				if agent.get_choice_prob() <= 0.1:
							
		# 					self.switch_time[agent.name][idx] += task.trial  - block
		# 					self.switch_data[agent.name][i, idx] = task.trial  - block
		# 					switch_bool[agent.name][idx] = True
		
			
		# print(self.switch_time[agent.name] - 8)

		



class MultiPlotExperiment(object):
	"""docstring for MultiPlotExperiment"""
	def __init__(self, opt):
		super(MultiPlotExperiment, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.max_trial = opt["max_trial"]
		
		self.time = 0
		self.num = opt["experiment_num"]

		self.N = opt["N"]
		self.reward_arr = {}
		self.regret_arr = {}
		self.switch_time = {}
		self.action_arr = {}
		self.evidence_arr = {} 
		self.value_arr = {} 

		self.task_arr = np.zeros((2, self.max_trial))
		self.model_arr =  {}


		self.reward_data = {}
		self.regret_data = {}
		self.choice_prob_data = {}
		self.action_data = {}
		self.evidence_data = {}
		self.value_data = {}
		self.model_data = {}
		self.task_data = {}
		self.switch_data = {}
		self.quantile_data = {}

	def run(self, agents, tasks, writer):

		agent_num = len(agents[0])
		
		task_num = len(tasks)
		


		self.total_reward = np.zeros((task_num, agent_num))
		self.total_regret = np.zeros((task_num, agent_num))

		for a in agents[0]:
			self.quantile_data[a.name] = {}
			self.value_data[a.name] = {}

		for a in agents[0]:
			self.regret_arr[a.name] = np.zeros(len(tasks))
			self.reward_arr[a.name] = np.zeros(len(tasks)) 

			self.reward_data[a.name] = np.zeros((task_num, self.N, self.max_trial))
			self.regret_data[a.name] = np.zeros((task_num, self.N, self.max_trial))
			self.action_data[a.name] = np.zeros((task_num, self.N, self.max_trial), dtype = np.int64)
			self.choice_prob_data[a.name] = np.zeros((task_num, self.N, self.max_trial))


			
			if isinstance(a, NeuralQuantileAgent):
				for i, t in enumerate(tasks):
					self.quantile_data[a.name][i] = np.zeros((self.N, a.stimuli_num, t.class_num, a.quantile_num, self.max_trial))
					
			for i, t in enumerate(tasks):
				self.value_data[a.name][i] = np.zeros((self.N, a.stimuli_num, t.class_num, self.max_trial))


		
		
			
		for j, task in enumerate(tasks):
			self.task_data[task.name] = np.zeros((task_num, self.N, task.class_num, self.max_trial))
			
			for i in range(self.N):
				print("This is iteration {}".format(i))

				self.total_regret[j, :] = np.zeros(agent_num)
				self.total_reward[j, :] = np.zeros(agent_num)
				for k in range(task.class_num):
					self.task_data[task.name][j, i, k, task.trial] = task.prob[task.context][0][k]


				for k, agent in enumerate(agents[j]):
					last = False
					while not last:
						a = agent.forward(task.stimuli)
						r, R, last, next_s = task.step(a)
						agent.update(r)
						
						self.total_reward[j][k] += r
						self.total_regret[j][k] += R

						self.action_data[agent.name][j, i, task.trial-1] = a
						self.choice_prob_data[agent.name][j, i, task.trial - 1] = agent.get_choice_prob()

						if isinstance(agent, TwoTimeScaleNeuralAgent):
							self.value_data[agent.name][j][i, :, :, task.trial - 1] = agent.ev()[agent.context]
						else:
							self.value_data[agent.name][j][i, :, :, task.trial - 1] = agent.ev()
						if isinstance(agent, NeuralQuantileAgent):
							self.quantile_data[agent.name][j][i,  :, :, :, task.trial-1] = agent.prob
							



						self.reward_data[agent.name][j, i, task.trial - 1] = self.total_reward[j][k]
						self.regret_data[agent.name][j, i, task.trial-1] = self.total_regret[j][k]






					agent.reset()
					task.reset()



			for a in agents[j]:
				self.reward_arr[a.name][j] /= self.N
				self.regret_arr[a.name][j] /= self.N

		data_dict = {}
		data_dict["reward"] = self.reward_data
		data_dict["regret"] = self.regret_data
		data_dict["action"] = self.action_data
		data_dict["value"] = self.value_data
		data_dict["task"] = self.task_data
		data_dict["choice_prob"] = self.choice_prob_data
		data_dict["quantile_data"] = self.quantile_data

		save_dict(data_dict, "experiment{}_data".format(self.opt["experiment_num"]))
			
		






		

class PlotExperiment(object):
	"""docstring for Experiment"""
	def __init__(self, opt):
		super(PlotExperiment, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.max_trial = opt["max_trial"]
		
		self.time = 0

		self.N = opt["N"]
		self.reward_arr = {}
		self.regret_arr = {}
		self.switch_time = {}
		self.action_arr = {}
		self.evidence_arr = {} 
		self.value_arr = {} 

		self.task_arr = np.zeros((self.class_num, self.max_trial))
		self.model_arr =  {}

		self.scalar_arr = {}
		self.histogram_arr = {}


		self.total_reward = 0
		self.total_regret = 0


		self.reward_data = {}
		self.regret_data = {}
		self.choice_prob_data = {}
		self.action_data = {}
		self.evidence_data = {}
		self.value_data = {}
		self.model_data = {}
		self.task_data = np.zeros((self.N, self.class_num, self.max_trial))
		self.switch_data = {}
		self.quantile_data = {}



	def run(self, agents, task, writer):
		#print(task.name)
		if task.name == "DynamicBandit" or task.name == "UnstructuredBandit":
			self.switch = int(self.opt["max_trial"] / self.opt["block_size"]) - 1
			#print("hi")
		elif task.name == "VolatileBandit":
			self.switch = int(self.opt["max_trial"] / (2 * self.opt["block_size"])) 
			#print(self.switch)
		# print(int(self.opt["max_trial"] / (2 * self.opt["block_size"])), int(self.opt["max_trial"] / (self.opt["block_size"])) )
		# print(task.name)
		# print(self.switch)
		switch_bool = {}
		for a in agents:
			self.reward_arr[a.name] = np.zeros(self.max_trial)
			self.regret_arr[a.name] = np.zeros(self.max_trial)
			self.switch_time[a.name] = np.zeros(self.switch)
			self.action_arr[a.name] = np.zeros(self.max_trial, dtype = np.int64)

			self.reward_data[a.name] = np.zeros((self.N, self.max_trial))
			self.regret_data[a.name] = np.zeros((self.N, self.max_trial))
			self.switch_data[a.name] = np.zeros((self.N, self.switch))
			self.action_data[a.name] = np.zeros((self.N, self.max_trial), dtype = np.int64)
			self.choice_prob_data[a.name] = np.zeros((self.N, self.max_trial))

			self.scalar_arr[a.name] = []
			self.histogram_arr[a.name] = []

			switch_bool[a.name] = [False for _ in range(self.switch)]
			if isinstance(a, TwoTimeScaleNeuralAgent) or a.name == "HMM Model":
				self.evidence_arr[a.name] = np.zeros(self.max_trial)
				self.value_arr[a.name] = np.zeros((a.context_num, a.stimuli_num, a.class_num, self.max_trial))
				self.model_arr[a.name] = np.zeros((a.context_num, a.stimuli_num, a.class_num, 2, self.max_trial))

				self.evidence_data[a.name] = np.zeros((self.N, self.max_trial))
				self.value_data[a.name] = np.zeros((self.N, a.context_num, a.stimuli_num, a.class_num, self.max_trial))
				self.model_data[a.name] = np.zeros((self.N, a.context_num, a.stimuli_num, a.class_num, 2, self.max_trial))

			if a.name == "Thompson Sampling" or a.name == "Discounted Thompson Sampling" or a.name == "Scalar RPE Model":
				self.value_data[a.name] = np.zeros((self.N, a.stimuli_num, a.class_num, self.max_trial))


			if isinstance(a, NeuralQuantileAgent) or a.name == "Normative Model":
				self.quantile_data[a.name] = np.zeros((self.N, a.stimuli_num, a.class_num, a.quantile_num, self.max_trial))

				self.value_data[a.name] = np.zeros((self.N, a.stimuli_num, a.class_num, self.max_trial))
			if a.name == "Known-Context Distributional RPE Model" or isinstance(a, TwoTimeScaleNeuralAgent):
				self.quantile_data[a.name] = np.zeros((self.N, a.context_num, a.stimuli_num, a.class_num, a.quantile_num, self.max_trial))
				self.value_data[a.name] = np.zeros((self.N, a.context_num, a.stimuli_num, a.class_num, self.max_trial))




		for i in range(self.N):
			print("This is iteration {}".format(i))
			for agent in agents:
				last = False
				while not last:

					for j in range(self.class_num):
						
						self.task_arr[j][task.trial] = task.prob[task.context][0][j]
						self.task_data[i, j, task.trial] = task.prob[task.context][0][j]
					a = agent.forward(task.stimuli)
					r, R, last, next_s = task.step(a)
					agent.update(r)



					self.action_data[agent.name][i, task.trial-1] = a
					self.choice_prob_data[agent.name][i, task.trial - 1] = agent.get_choice_prob()

					self.scalar_arr[agent.name].append(agent.scalars())
					self.histogram_arr[agent.name].append(agent.histogram())
					data = agent.scalars()

					if agent.name == "Thompson Sampling" or agent.name == "Discounted Thompson Sampling" or agent.name == "Scalar RPE Model" or isinstance(agent, NeuralQuantileAgent)  or agent.name == "Normative Model":
						self.value_data[agent.name][i, :, :, task.trial - 1] = agent.ev()

					if agent.name == "Known-Context Distributional RPE Model":
						self.value_data[agent.name][i, :, :, :, task.trial - 1] = agent.ev()

					if isinstance(agent, NeuralQuantileAgent)  or agent.name == "Normative Model":
					
						for s in range(agent.stimuli_num):
							for a in range(agent.class_num):
								for q in range(agent.quantile_num):
									self.quantile_data[agent.name][i, s, a, q, task.trial-1] = agent.prob[s, a, q]

					if agent.name == "Known-Context Distributional RPE Model" or isinstance(agent, TwoTimeScaleNeuralAgent):
						for c in range(agent.context_num):
							for s in range(agent.stimuli_num):
								for a in range(agent.class_num):
									for q in range(agent.quantile_num):
										self.quantile_data[agent.name][i, c, s, a, q, task.trial-1] = agent.prob[c, s, a, q]

					if isinstance(agent, TwoTimeScaleNeuralAgent) or agent.name == "HMM Model":
						
						self.evidence_data[agent.name][i, task.trial-1] = data["context-difference"]
						for c in range(agent.context_num):
							for s in range(agent.stimuli_num):
								for a in range(agent.class_num):
									self.value_data[agent.name][i,c,s,a,task.trial-1] = data["fast-reward/context-{}/simuli-{}/action-{}".format(c, s, a)]
									for j in range(2):
										self.model_data[agent.name][i,c,s,a,j,task.trial-1] = data["likelihood/context-{}/simuli-{}/action-{}/reward-{}".format(c, s, a, j)]
					



					if task.name == "DynamicBandit":
						idx = int((task.trial - 1) / self.opt["block_size"]) - 1

						if idx > 1:
							if not switch_bool[agent.name][idx-2]:
								self.switch_time[agent.name][idx-2] += self.opt["block_size"]
								self.switch_data[agent.name][i, idx-2] = self.opt["block_size"]
								switch_bool[agent.name][idx - 2] = True
								#print(task.trial,self.switch_time[agent.name][idx-2])
						
						if idx > -1 and not switch_bool[agent.name][idx]:
							new_a = (idx+1) % 2


							# if agent.get_ev()[0][new_a] - agent.get_ev()[0][1-new_a] >= 0.1:
							# 	self.switch_time[agent.name][idx] += task.trial  - (idx + 1) * self.opt["block_size"] 
							# 	switch_bool[agent.name][idx] = True
							# 	print(task.trial, idx, self.switch_time[agent.name][idx])

							if new_a == 1:
								if agent.get_choice_prob() >= 0.8:
									self.switch_time[agent.name][idx] += task.trial  - (idx + 1) * self.opt["block_size"] 
									self.switch_data[agent.name][i, idx] = task.trial  - (idx + 1) * self.opt["block_size"] 
									switch_bool[agent.name][idx] = True
									#print(self.switch_time[agent.name][idx])

							else:
								if agent.get_choice_prob() <= 0.2:
									self.switch_time[agent.name][idx] += task.trial  - (idx + 1) * self.opt["block_size"] 
									self.switch_data[agent.name][i, idx] = task.trial  - (idx + 1) * self.opt["block_size"] 
									switch_bool[agent.name][idx] = True
									#print(self.switch_time[agent.name][idx])


					if task.name == "VolatileBandit":
						
						if task.trial <= self.max_trial / 2:
							idx = - 1
						else:
							idx = int((task.trial - self.max_trial / 2 - 1 ) / self.opt["block_size"]) 
						
						
						if idx > 1:
							if not switch_bool[agent.name][idx-2]:
								
								self.switch_time[agent.name][idx-2] += self.opt["block_size"]
								self.switch_data[agent.name][i, idx-2] = self.opt["block_size"]
							
								switch_bool[agent.name][idx - 2] = True
						
						if idx > -1 and not switch_bool[agent.name][idx]:

							new_a = (idx+1) % 2
							

							# if agent.get_ev()[0][new_a] - agent.get_ev()[0][1-new_a] >= 0.1:
							# 	self.switch_time[agent.name][idx] += task.trial  - (idx + 1) * self.opt["block_size"] 
							# 	switch_bool[agent.name][idx] = True
							# 	print(task.trial, idx, self.switch_time[agent.name][idx])

							
							block = self.max_trial / 2 + (idx) * self.opt["block_size"] 
							
							if new_a == 1:
								if agent.get_choice_prob() >= 0.8:
								
									self.switch_time[agent.name][idx] += task.trial  - block
									self.switch_data[agent.name][i, idx] = task.trial  - block
									
									switch_bool[agent.name][idx] = True
									

							else:
								if agent.get_choice_prob() <= 0.2:
						
									self.switch_time[agent.name][idx] += task.trial  - block
									self.switch_data[agent.name][i, idx] = task.trial  - block
									switch_bool[agent.name][idx] = True

								
					




					self.total_reward += r
					self.total_regret += R
					self.reward_arr[agent.name][task.trial - 1] += self.total_reward
					self.regret_arr[agent.name][task.trial - 1] += self.total_regret

					self.reward_data[agent.name][i, task.trial - 1] = self.total_reward
					self.regret_data[agent.name][i, task.trial-1] = self.total_regret

				if self.switch > 1:
					for j in range(2):
					
						if not switch_bool[agent.name][self.switch - 1 - j]:
							self.switch_time[agent.name][self.switch - 1 - j] += self.opt["block_size"]
							self.switch_data[agent.name][i, self.switch - 1 - j] = self.opt["block_size"]	
							switch_bool[agent.name][self.switch - 1 - j] = True

				
				switch_bool[agent.name] = [False for _ in range(self.switch)]
		
				agent.reset()
				task.reset()
				self.total_reward = 0
				self.total_regret = 0

		for a in agents:
			self.reward_arr[a.name] /= self.N
			self.regret_arr[a.name] /= self.N
			self.switch_time[a.name] /= self.N


		# for agent in agents:

		# 	last = False
		# 	while not last:
		# 		a = agent.forward(task.stimuli)
		# 		r, R, last, next_s = task.step(a)
		# 		agent.update(r)
		# 		self.action_arr[agent.name][task.trial - 1] = a
		# 		data = agent.scalars()
		# 		if agent.name == "Thalamocortical Model" or agent.name == "HMM Model":
		# 			self.evidence_arr[agent.name][task.trial-1] = data["context-difference"]
		# 			for c in range(agent.context_num):
		# 				for s in range(agent.stimuli_num):
		# 					for a in range(agent.class_num):
		# 						self.value_arr[agent.name][c][s][a][task.trial-1] = data["fast-reward/context-{}/simuli-{}/action-{}".format(c, s, a)]
		# 						for j in range(2):
		# 							self.model_arr[agent.name][c][s][a][j][task.trial-1] = data["likelihood/context-{}/simuli-{}/action-{}/reward-{}".format(c, s, a, j)]
					
		# 	agent.reset()
		# 	task.reset()


	

		data_dict = {}
		data_dict["reward"] = self.reward_data
		data_dict["regret"] = self.regret_data
		data_dict["action"] = self.action_data
		data_dict["evidence"] = self.evidence_data
		data_dict["value"] = self.value_data
		data_dict["model"] = self.model_data
		data_dict["task"] = self.task_data
		data_dict["switch"] = self.switch_data
		data_dict["choice_prob"] = self.choice_prob_data
		data_dict["quantile_data"] = self.quantile_data
		data_dict["scalars"] = self.scalar_arr
		data_dict["histogram"] = self.histogram_arr

		save_dict(data_dict, "experiment{}_data".format(self.opt["experiment_num"]))

		# t = np.arange(self.max_trial)
		# for i, a in enumerate(agents):
		# 	if i == 0:
		# 		c = "blue"
		# 	elif i == 1:
		# 		c = "red"
		# 	elif i == 2:
		# 		c = "green"
		# 	else:
		# 		c = "black"
		# 	plt.plot(t, self.reward_arr[a.name], label = a.name, c = c)
		# plt.legend(loc="upper left")
		# plt.xlabel("Trial")
		# plt.ylabel("Accumulated reward")
		# plt.title("Averaged accumulated reward over {} trials".format(self.max_trial))
		# plt.savefig("fig/experiment{}_reward.png".format(self.opt["experiment_num"]))
		# plt.close()

		# for i, a in enumerate(agents):
		# 	if i == 0:
		# 		c = "blue"
		# 	elif i == 1:
		# 		c = "red"
		# 	elif i == 2:
		# 		c = "green"
		# 	else:
		# 		c = "black"
		# 	plt.plot(t, self.regret_arr[a.name], label = a.name, c = c)
		# plt.legend(loc="upper left")
		# plt.xlabel("Trial")
		# plt.ylabel("Accumulated regret")
		# plt.title("Averaged accumulated regret over {} trials".format(self.max_trial))
		# plt.savefig("fig/experiment{}_regret.png".format(self.opt["experiment_num"]))
		# plt.close()


		# # for i, a in enumerate(agents):
		# # 	if i == 0:
		# # 		c = "blue"
		# # 	elif i == 1:
		# # 		c = "red"
		# # 	elif i == 2:
		# # 		c = "green"
		# # 	else:
		# # 		c = "black"
		# # 	plt.plot(t[600:], self.regret_arr[a.name][600:] - self.regret_arr[a.name][600], label = a.name, c = c)
		# # plt.legend(loc="upper left")
		# # plt.xlabel("Trial")
		# # plt.ylabel("Accumulated regret")
		# # plt.title("Averaged accumulated regret over {} trials after three switches".format(self.max_trial))
		# # if self.opt["experiment_num"] == 12:
		# # 	fig = plt.gcf()
		# # 	fig.set_size_inches(4.8, 4.8)
		# # plt.tight_layout()
		# # plt.savefig("fig/experiment{}_regret_after_three_switches.png".format(self.opt["experiment_num"]))
		# # plt.close()



		# for i in range(self.class_num):
			
		# 	c = (1 - i * 1.0 / (self.class_num-1), 0, i * 1.0 / (self.class_num-1) )
		
		# 	plt.plot(t, self.task_arr[i], label = "Action {}".format(i+1), c = c)
		# plt.legend(loc="upper left")
		# plt.ylim(0, 1)
		# if self.opt["experiment_num"] == 12:
		# 	fig = plt.gcf()
		# 	fig.set_size_inches(2.4, 4.8)
		# plt.xlabel("Trial")
		# plt.ylabel("Probability of reward")
		# plt.title("Task reward structure")
		# plt.tight_layout()

		
		# plt.savefig("fig/experiment{}_task.png".format(self.opt["experiment_num"]))
		# plt.close()


		# if task.name == "DynamicBandit":
		# 	bt = np.arange(200, 2000, 200)
		# if task.name == "VolatileBandit":
		# 	bt = np.concatenate((np.arange(200, 1000, 200), np.arange(1000, 2000, 100)))
		# 	print(bt)
		# if task.name == "UnstructuredBandit":
		# 	bt = np.arange(200, 2000, 200)


		# for i, a in enumerate(agents):
		# 	ax = plt.figure().gca()
		# 	ax.yaxis.get_major_locator().set_params(integer=True)
		# 	if i == 0:
		# 		c = "blue"
		# 	elif i == 1:
		# 		c = "red"
		# 	elif i == 2:
		# 		c = "green"
		# 	else:
		# 		c = "black"
		# 	plt.scatter(t, self.action_arr[a.name], label = a.name, facecolors='none', edgecolors=c, s = 3)
		# 	for v in bt:
		# 		plt.axvline(v, c = "black")


		# 	plt.xlabel("Trial")
		# 	plt.ylabel("Action of {}".format(a.name))
		# 	plt.title("Action of {} over {} trials".format(a.name, self.max_trial))
		# 	plt.savefig("fig/experiment{}_action_{}.png".format(self.opt["experiment_num"], a.name))
		# 	plt.close()


		# st = np.arange(self.switch) + 1

		# for i, a in enumerate(agents):
		# 	if i == 0:
		# 		c = "blue"
		# 	elif i == 1:
		# 		c = "red"
		# 	elif i == 2:
		# 		c = "green"
		# 	else:
		# 		c = "black"
		# 	plt.plot(st, self.switch_time[a.name], label = a.name, c = c)
		# plt.legend(loc="upper left")
		# plt.xlabel("Block")
		# plt.ylabel("Time to switch")
		# plt.title("Averaged time to switch over blocks")
		# plt.savefig("fig/experiment{}_switch_time.png".format(self.opt["experiment_num"]))
		# plt.close()


		# for i, a in enumerate(agents):
		# 	if a.name == "Thalamocortical Model" or a.name == "HMM Model":
		# 		if i == 0:
		# 			c = "lawngreen"
		# 		elif i == 1:
		# 			c = "purple"
		# 		elif i == 2:
		# 			c = "green"
		# 		else:
		# 			c = "black"

		# 		plt.plot(t, self.evidence_arr[a.name],  c = c)
		# 		for v in bt:
		# 			plt.axvline(v, c = "black")
		# 		plt.legend(loc="upper left")
		# 		plt.xlabel("Trial")
		# 		plt.ylabel("Normalized firing rate")
		# 		plt.title("Difference between two contextual populations\nfor {} over 2000 trials".format(a.name))
		# 		plt.savefig("fig/experiment{}_evidence_{}.png".format(self.opt["experiment_num"],a.name))
		# 		plt.close()

		# for i, a in enumerate(agents):
		# 	if a.name == "Thalamocortical Model" or a.name == "HMM Model":
				
		# 		for c in range(a.context_num):
		# 			for s in range(a.stimuli_num):
		# 				for j in range(a.class_num):
		# 					if j == 0:
		# 						co = "m"
		# 					elif j == 1:
		# 						co = "c"
		# 					elif j == 2:
		# 						co = "green"
		# 					else:
		# 						co = "red"
		# 					plt.plot(t, self.value_arr[a.name][c][s][j], label = "Action {}".format(j+1), c = co)
		# 				plt.legend(loc="upper left")
		# 				plt.xlabel("Trial")
		# 				plt.ylabel("Estimated value")
		# 				plt.title("Estimated value of {}\n at context {} for stimuli {} over 2000 trials".format(a.name, c, s))
		# 				plt.savefig("fig/experiment{}_value_{}_{}_{}.png".format(self.opt["experiment_num"],a.name, c, s))
		# 				plt.close()

		# for i, a in enumerate(agents):
		# 	if a.name == "Thalamocortical Model" or a.name == "HMM Model":
		# 		for c in range(a.context_num):
		# 			for s in range(a.stimuli_num):
		# 				for ac in range(a.class_num):
		# 					for j in range(2):
		# 						if j == 0:
		# 							co = "blue"
		# 							label = "No reward"
		# 						else:
		# 							co = "red"
		# 							label = "Reward"
		# 						plt.plot(t, self.model_arr[a.name][c][s][ac][j], label = label, c = co)

		# 					plt.legend(loc="upper left")
		# 					plt.xlabel("Trial")
		# 					plt.ylabel("Estimated Probability")
		# 					plt.title("Generative model of reward of {}\ntaking action {} to stimuli {} at context {} over 2000 trials".format(a.name, ac, s, c))
		# 					plt.savefig("fig/experiment{}_model_{}_{}_{}_{}.png".format(self.opt["experiment_num"],a.name, c, s, ac))
		# 					plt.close()


# class Plot2Experiment(object):
# 	"""docstring for Experiment"""
# 	def __init__(self, opt):
# 		super(Plot2Experiment, self).__init__()
# 		self.opt = opt
# 		self.stimuli_num = opt["stimuli_num"]
# 		self.class_num = opt["class_num"]
# 		self.max_trial = opt["max_trial"]
# 		self.switch = int(opt["max_trial"] / opt["block_size"]) - 1
# 		self.time = 0

# 		self.N = 10
# 		self.reward_arr = {}
# 		self.regret_arr = {}
# 		self.switch_time = {}
# 		self.action_arr = {}
# 		self.evidence_arr = np.zeros(self.max_trial)
# 		self.value_arr = np.zeros((2, self.max_trial))

# 		self.task_arr = np.zeros((2, self.max_trial))
# 		self.model_arr = np.zeros((2, self.max_trial))


# 		self.total_reward = 0
# 		self.total_regret = 0


# 	def run(self, agents, task, writer):

# 		switch_bool = {}
# 		for a in agents:
# 			self.reward_arr[a.name] = np.zeros(self.max_trial)
# 			self.regret_arr[a.name] = np.zeros(self.max_trial)
# 			self.switch_time[a.name] = np.zeros(self.switch)
# 			self.action_arr[a.name] = np.zeros(self.max_trial, dtype = np.int64)

# 			switch_bool[a.name] = [False for _ in range(self.switch)]


# 		for i in range(self.N):
# 			for agent in agents:
# 				last = False
# 				while not last:

# 					for j in range(2):
# 						self.task_arr[j][task.trial] = task.prob[task.context][0][j]
# 					a = agent.forward(task.stimuli)
# 					r, R, last, next_s = task.step(a)
# 					agent.update(r)
# 					idx = int((task.trial - 1) / self.opt["block_size"]) - 1
				
# 					if idx > -1 and not switch_bool[agent.name][idx]:
# 						new_a = (idx + 1) % 2
# 						agent_a = np.argmax(agent.get_ev()[0])
# 						if agent.get_ev()[0][new_a] - agent.get_ev()[0][1-new_a] >= 0.1:
# 							self.switch_time[agent.name][idx] += task.trial - 1 - (idx + 1) * self.opt["block_size"]
# 							switch_bool[agent.name][idx] = True

# 					self.total_reward += r
# 					self.total_regret += R
# 					self.reward_arr[agent.name][task.trial - 1] += self.total_reward
# 					self.regret_arr[agent.name][task.trial - 1] += self.total_regret
				
# 				switch_bool[agent.name] = [False for _ in range(self.switch)]
# 				agent.reset()
# 				task.reset()
# 				self.total_reward = 0
# 				self.total_regret = 0

# 		for a in agents:
# 			self.reward_arr[a.name] /= self.N
# 			self.regret_arr[a.name] /= self.N
# 			self.switch_time[a.name] /= self.N


# 		for agent in agents:

# 			last = False
# 			while not last:
# 				a = agent.forward(task.stimuli)
# 				r, R, last, next_s = task.step(a)
# 				agent.update(r)
# 				self.action_arr[agent.name][task.trial - 1] = a
# 				data = agent.scalars()
# 				if agent.name == "Thalamocortical Model":
# 					self.evidence_arr[task.trial-1] = data["context-difference"]
# 					for j in range(2):
# 						self.value_arr[j][task.trial-1] = data["fast-reward/context-0/simuli-0/action-{}".format(j)]
# 						self.model_arr[j][task.trial-1] = data["likelihood/context-0/simuli-0/action-0/reward-{}".format(j)]
					
# 			agent.reset()
# 			task.reset()

# 		t = np.arange(self.max_trial)
# 		for i, a in enumerate(agents):
# 			if i == 0:
# 				c = "blue"
# 			elif i == 1:
# 				c = "red"
# 			else:
# 				c = "black"
# 			plt.plot(t, self.reward_arr[a.name], label = a.name, c = c)
# 		plt.legend(loc="upper left")
# 		plt.xlabel("Trial")
# 		plt.ylabel("Accumulated reward")
# 		plt.title("Averaged accumulated reward over 2000 trials")
# 		plt.savefig("fig/reward.png")
# 		plt.close()

# 		for i, a in enumerate(agents):
# 			if i == 0:
# 				c = "blue"
# 			elif i == 1:
# 				c = "red"
# 			else:
# 				c = "black"
# 			plt.plot(t, self.regret_arr[a.name], label = a.name, c = c)
# 		plt.legend(loc="upper left")
# 		plt.xlabel("Trial")
# 		plt.ylabel("Accumulated regret")
# 		plt.title("Averaged accumulated regret over 2000 trials")
# 		plt.savefig("fig/regret.png")
# 		plt.close()


# 		for i in range(self.class_num):
# 			if i == 0:
# 				c = (1 - i * 1.0 / (class_num-1), i * 1.0 / (class_num-1), 0 )
# 			else:
# 				c = (i * 1.0 / (class_num-1), 1 - i * 1.0 / (class_num-1), 0)
# 			plt.plot(t, self.task_arr[i], label = "Action {}".format(i+1), c = c)
# 		plt.legend(loc="upper left")
# 		plt.ylim(0, 1)
# 		plt.xlabel("Trial")
# 		plt.ylabel("Probability of reward")
# 		plt.title("Task reward structure")
# 		plt.savefig("fig/task.png")
# 		plt.close()



# 		bt = np.arange(200, 2000, 200)

# 		for i, a in enumerate(agents):
# 			if i == 0:
# 				c = "blue"
# 			elif i == 1:
# 				c = "blue"
# 			else:
# 				c = "black"
# 			plt.scatter(t, self.action_arr[a.name], label = a.name, facecolors='none', edgecolors=c, s = 3)
# 			for v in bt:
# 				plt.axvline(v, c = "red")
# 			plt.xlabel("Trial")
# 			plt.ylabel("Action of {}".format(a.name))
# 			plt.title("Action of {} over 2000 trials".format(a.name))
# 			plt.savefig("fig/action_{}.png".format(a.name))
# 			plt.close()


# 		st = np.arange(self.switch) + 1

# 		for i, a in enumerate(agents):
# 			if i == 0:
# 				c = "blue"
# 			elif i == 1:
# 				c = "red"
# 			else:
# 				c = "black"
# 			plt.plot(st, self.switch_time[a.name], label = a.name, c = c)
# 		plt.legend(loc="upper left")
# 		plt.xlabel("Block")
# 		plt.ylabel("Time to switch")
# 		plt.title("Averaged time to switch over 10 blocks")
# 		plt.savefig("fig/switch_time.png")
# 		plt.close()


# 		plt.plot(t, self.evidence_arr,  c = "blue")
# 		for v in bt:
# 			plt.axvline(v, c = "red")
# 		plt.legend(loc="upper left")
# 		plt.xlabel("Trial")
# 		plt.ylabel("Normalized firing rate")
# 		plt.title("Difference between two thalamic populations over 2000 trials")
# 		plt.savefig("fig/evidence.png")
# 		plt.close()

# 		for i in range(2):
# 			if i == 0:
# 				c = "blue"
# 			else:
# 				c = "red"
# 			plt.plot(t, self.value_arr[i], label = "Action {}".format(i+1), c = c)

# 		plt.legend(loc="upper left")
# 		plt.xlabel("Trial")
# 		plt.ylabel("Estimated value")
# 		plt.title("Estimated value at context 0 over 2000 trials")
# 		plt.savefig("fig/value.png")
# 		plt.close()

# 		for i in range(2):
# 			if i == 0:
# 				c = "blue"
# 				label = "No reward"
# 			else:
# 				c = "red"
# 				label = "Reward"
# 			plt.plot(t, self.model_arr[i], label = label, c = c)

# 		plt.legend(loc="upper left")
# 		plt.xlabel("Trial")
# 		plt.ylabel("Estimated Probability")
# 		plt.title("Generative model of reward of taking action 1 at context 0 over 2000 trials")
# 		plt.savefig("fig/model.png")
# 		plt.close()








		



class TimeExperiment(object):
	"""docstring for TimeExperiment"""
	def __init__(self, opt):
		super(TimeExperiment, self).__init__()
		self.opt = opt
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.stimuli_time = opt["stimuli_time"] // opt["dt"]
		self.waiting_time = opt["waiting_time"] // opt["dt"]
		self.reward_time = opt["reward_time"] // opt["dt"]
		self.inter_trial = opt["inter_trial"] // opt["dt"]
		self.logging_interval = opt["logging_interval"] // opt["dt"]
		self.total_time = self.stimuli_time + self.waiting_time + self.reward_time + self.inter_trial
		self.dt = opt["dt"]

		self.time = 0

		self.total_reward = 0
		self.total_regret = 0


	def run(self, agent, task, writer):
		
		last = False
		da = []
		v = []
		while not last:
			if self.time % self.total_time == 0:
				stimuli = task.stimuli
				# if task.trial % 10 == 0:
				# 	print("da")
				# 	if len(da) > 0:
				# 		plt.plot(np.arange(len(da)), da)
				# 		plt.show()
				# 	if len(v) > 0:
				# 		plt.plot(np.arange(len(v)), v)
				# 		plt.show()
				da = []
				v = []

			if self.time % self.total_time < self.stimuli_time:
				# if self.time % 10 == 0:
				# 	print("stimuli")
				agent.forward(one_hot(task.stimuli, self.stimuli_num))
				agent.update(0)
			elif self.time % self.total_time < self.stimuli_time + self.waiting_time:
				# if self.time % 10 == 0:
				# 	print("waiting")
				agent.forward(np.zeros(self.stimuli_num))
				agent.update(0)
			elif self.time % self.total_time == self.stimuli_time + self.waiting_time:
				o = agent.forward(np.zeros(self.stimuli_num))
				a = np.argmax(o)
				# print("hi")
				# print(agent.BG.old_v)
				# print(agent.BG.v)
				# print(agent.BG.gamma * agent.BG.v -agent.BG.old_v)
				reward, regret, last, stimuli = task.step(a)
				agent.update(reward)

				self.total_reward += reward
				self.total_regret += regret

				writer.add_scalar("reward", self.total_reward, task.trial)
				writer.add_scalar("regret", self.total_regret, task.trial)

			elif self.time % self.total_time < self.stimuli_time + self.waiting_time + self.reward_time:
				# if self.time % 10 == 0:
				# 	print("reward")
				agent.forward(np.zeros(self.stimuli_num))
				agent.update(reward)
			else:
				# if self.time % 10 == 0:
				# 	print("no reward")
				agent.forward(np.zeros(self.stimuli_num))
				agent.update(0)

			self.time += 1

			da.append(agent.BG.da)
			v.append(agent.BG.v)



			# if self.time % 10 == 0:
			# 	print(self.time)
			# 	agent.cortex.plot()

			if self.time % self.logging_interval == 0:
				writer.add_scalar("dopamine", agent.BG.da, self.time )
				writer.add_scalar('value', agent.BG.v, self.time )
				scalars = agent.scalars()
				hist = agent.histogram()
				for k in hist:
					writer.add_histogram(k, hist[k], self.time )
				for k in scalars:
					writer.add_scalar(k, scalars[k], self.time )





			