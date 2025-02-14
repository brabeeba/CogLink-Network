# import os
# os.system("python train.py --experiment_num 36")
# os.system("python train.py --experiment_num 37")






# from model import *

# opt = {}
# opt["stimuli_num"] =1
# opt["class_num"] = 3
# opt["max_trial"] = 10
# agent = GittinsIndex(opt)
#dx1/dt = -x1 + x_2 + 1
#dx2/dt = -x2 + x_1 + 1

# import numpy as np
# from scipy.stats import beta
# import matplotlib.pyplot as plt
# import seaborn as sns

# quantile_num = 50
# sample_num = 3

# quantile = (np.arange(quantile_num) + 1.0 ) / quantile_num

# prob = 0.7
# max_trial = 500
# count = np.ones(2)

# timepoints = [0, 10, 50]

# fig, ax = plt.subplots(len(timepoints),  sharey=True)
# fig.set_figwidth(6.4)
# fig.set_figheight(12)

# bins = np.linspace(0, 1, 100)
# std = 0.001
# idx = 0

# samplepoints = [1, 2, 3, 4]


# def outer_sum(x, num):
	
# 	current = x
# 	for _ in range(num - 1):
# 		current = np.expand_dims(current, 0) + np.expand_dims(x, 1)
# 		current = current.flatten()

# 	return current / num


# for i in range(max_trial):


# 	if i == timepoints[idx]:
# 		df = [ outer_sum(quantile, s) for s in samplepoints]

						
# 		q_data = [np.sum(np.exp(-0.5 * (np.expand_dims(bins, 1) - np.expand_dims(df[i], 0)) ** 2 / std ** 2) / (np.sqrt(2 * np.pi) * std), axis = 1) / len(df[i])  for i, s in enumerate(samplepoints)]
# 		print(np.sum(q_data[0]) / 101)
# 		print(np.sum(q_data[1]) / 101)
# 		print(np.sum(q_data[2]) / 101)
# 		print(np.sum(beta.pdf(bins, count[1], count[0])) / 101)
# 		for i, s in enumerate(samplepoints):
# 			ax[idx].plot(bins, q_data[i], label = "quantile {}".format(s))
# 		ax[idx].plot(bins, beta.pdf(bins, count[1], count[0]), label = "Bayesian")
# 		idx += 1

# 	if idx >= len(timepoints):
# 		break


# 	sample = np.random.rand()
# 	if sample < prob:
# 		observation = 1
# 	else:
# 		observation = 0
# 	count[observation] += 1

# 	quantile[:-1] += (observation - quantile[:-1]) / (3.0 + np.sum(count))





	

# handles, labels = ax[0].get_legend_handles_labels()
				
# fig.legend(handles, labels,  bbox_to_anchor=(1, 0.95), loc = "upper right", frameon=False)
# sns.despine()
# fig.supxlabel("Values", x= 0.625, y = 0.06,  fontsize = "small")
# fig.supylabel("Probability density", x = 0.12, fontsize = "small")
# plt.tight_layout()
# plt.show()


	



# import numpy as np
# from util import *
# from scipy.special import expit
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# def preprocess(num):
# 	data = load_dict("experiment{}_data".format(num))
# 	agents = list(data["action"].keys())
# 	max_episode, max_trial = data["action"]["Thompson Sampling"].shape
	
# 	t = np.arange(max_trial)


# 	#trial_df = pd.DataFrame({"Trial": [], "Episode": [], "Reward":[], "Regret": [], "Action":[], "Evidence": [], "Context":[], "Value":[], "Action value":[], "Choice probability":[], "Model": []})
# 	trial_df = pd.DataFrame({"Trial": pd.Series(dtype='int'), "Episode": pd.Series(dtype='int'), "Model": pd.Series(dtype='str'), "Regret": pd.Series(dtype='float')})
# 	for i in range(max_trial):
# 		for a in agents:
# 			for j in range(max_episode):
# 				df = {}
# 				df["Trial"] = [i + 1]
# 				df["Episode"] = [j + 1]
# 				df["Model"] = [a]
# 				df["Regret"] = [data["reward"][a][j][i]]
# 				df = pd.DataFrame(df)
# 				trial_df = pd.concat([trial_df, df])

# 	trial_df.reset_index(drop = True)
# 	trial_df.set_index(["Trial", "Episode"])
	

# 	return trial_df

# df = preprocess(12)
# print(df)

# sns.relplot(data = df, kind = "line", x = "Trial", y = "Regret", hue = "Model")
# plt.show()



# def f(inputs):
# 	return np.minimum(1, relu(inputs))

# def g(inputs):
# 	return relu(inputs - 0.001)

# def h(inputs):

# 	return np.tanh(inputs * 2) 

# K = 4


# b1 = 1


# quantile_num = 100
# I = (K-0.25)*b1

# a1 = 0.75

# sample_w1 = -  b1 * np.ones((quantile_num, quantile_num))
# sample_w2 = np.eye(quantile_num) * a1


# for i in range(quantile_num):
	
# 	sample_w1[i, i] = a1
# #print(sample_w1)



# for j in range(100):
# 	sample_neurons = np.zeros((quantile_num)) 
# 	for i in range(500):

# 		update = (-  sample_neurons + sample_w1.dot(f(sample_neurons))  +  I  + 0.1 * np.random.normal(0, 1, quantile_num) )
	
# 		sample_neurons += 0.03 * update
		
# 	print(np.sum(sample_neurons > 1), np.sum(sample_neurons < 0))
# #print(sample_neurons)	

#fmri = sns.load_dataset("fmri")
#print(len(fmri[fmri["timepoint"] == 1]))
# sns.relplot(
#     data=fmri, kind="line",
#     x="timepoint", y="signal", col="region",
#     hue="event", style="event",
# )
# plt.show()