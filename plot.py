import matplotlib.pyplot as plt
import matplotlib
from util import load_dict, save_dict, relu
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats
from matplotlib.collections import PolyCollection
import scikit_posthocs as sp
import argparse
import os
from model import *
if not os.path.exists("./fig"):
    os.makedirs("./fig")

from sklearn.svm import LinearSVC 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_num', type=int, default=2)
opt = vars(parser.parse_args())

num = opt["experiment_num"]

sns.set_theme(context = "paper", style = "ticks")
#sns.set_theme("ticks", palette=None, context = "paper")
palette = sns.color_palette()
palette_alt = sns.color_palette("Set2")
palette_gradient = sns.color_palette("rocket")


def stars(p):
   if p < 0.0001:
       return "****"
   elif (p < 0.001):
       return "***"
   elif (p < 0.01):
       return "**"
   elif (p < 0.05):
       return "*"
   else:
       return "-"

def plot(num):
	print("Analyze the data of experiment {} and plot the corresponding figures".format(num))
	
		

	if num == 0:
		x = np.linspace(4, 8, 101)
		f = 2 / (1 + np.exp(8-4 * (x-4))) - 1
		f[f < 0] = 0 

		ratio = 0.7
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		plt.plot(x, f)
		
		sns.despine()
		plt.axvline(6, c = "grey", linewidth = 1, linestyle = "dashed")
		plt.savefig("fig/experiment{}_hebbian_nonlinearity.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.7
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		x = np.linspace(-2, 2, 101)
		f = x + 1
		f[f< 0] = 0
		f[f>2] = 2
		plt.plot(x, f)
		
		sns.despine()
		plt.axvline(1, c = "grey", linewidth = 1, linestyle = "dashed")
		plt.axvline(-1, c = "grey", linewidth = 1, linestyle = "dashed")
		plt.savefig("fig/experiment{}_md_nonlinearity.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.7
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		x = np.linspace(-4, 4, 101)
		f = 2 / (1 + np.exp(-2 * (x+0.25))) - 1
		f[f < 0] = 0 
		plt.plot(x, f)
		
		sns.despine()
		plt.axvline(0, c = "grey", linewidth = 1, linestyle = "dashed")
		plt.savefig("fig/experiment{}_in_nonlinearity.pdf".format(num), transparent = True)
		plt.close()

		data = load_dict("experiment9_data")
		quantile_num = 100

		def mean_loss(c):
			
			length = 500
			quantile_num = 100

			prob = (np.arange(quantile_num)) / float(quantile_num)
			a = 1
			b = 1

			var_result = []
			var_result1 = []
			exp_result = []
			exp_result1 = []

			for sample in range(50):
				result = []
				result1 = []

				e_result = []
				e_result1 = []
				for i in range(length):
					if np.random.random() < 0.3:
						sample = 0
					else:
						sample = 1

					prob += (sample - prob) / (c** (1/d) + i) ** d
				
					if sample == 0:
						b += 1
					else:
						a += 1
					
					exp = np.sum(prob) / float(quantile_num)
					exp_2 = np.sum(prob * prob) / float(quantile_num)
					var = exp_2 - exp ** 2
					result1.append(var)
					result.append((a*b) / ((a+b)**2 * (a+b+1)))

					e_result.append(a / (a+b))
					e_result1.append(exp)

				var_result.append(result)
				var_result1.append(result1)
				exp_result.append(e_result)
				exp_result1.append(e_result1)


			var_result = np.array(var_result)
			var_result1 = np.array(var_result1)
			exp_result = np.array(exp_result)
			exp_result1 = np.array(exp_result1)

			print(np.linalg.norm(np.mean(var_result, axis = 0) - np.mean(var_result1, axis = 0)), np.linalg.norm(np.mean(exp_result, axis = 0) - np.mean(exp_result1, axis = 0)) / 90)

			return np.linalg.norm(np.mean(var_result, axis = 0) - np.mean(var_result1, axis = 0)) + np.linalg.norm(np.mean(exp_result, axis = 0) - np.mean(exp_result1, axis = 0)) / 90

		#best_params = 6.157894736842105, 1.3157894736842106
		# min_value = float("inf")

		# x_range = np.linspace(10, 15, 20)
		# y_range = np.linspace(0.8, 1, 20) 

		# for x in x_range:
		# 	for y in y_range:
		# 		value = mean_loss(x, y)
		# 		if value < min_value:
		# 			min_value = value
		# 			best_params = (x, y)


		# # Print the results
		# print(f"Optimal parameters: {best_params}")
		# print(f"Minimum value of the function: {min_value}")


		length = 500

		prob = (np.arange(quantile_num)) / float(quantile_num)
		a = 1
		b = 1
		var_result = []
		var_result1 = []
		exp_result = []
		exp_result1 = []
		c, d, e = 10, 1, 1

		for sample in range(50):
			result = []
			result1 = []

			e_result = []
			e_result1 = []
			for i in range(length):
				exp = np.sum(prob) / quantile_num
				exp_2 = np.sum(prob * prob ) / quantile_num
				
				
				var = exp_2 - exp ** 2
				
				result1.append(var)
				result.append((a*b) / ((a+b)**2 * (a+b+1)))

				e_result.append(a / (a+b))
				e_result1.append(exp)

				if np.random.random() < 0.3:
					sample = 0
				else:
					sample = 1

				prob += (sample - prob) / (c** (1/d) + e * i) ** d
				
				if sample == 0:
					b += 1
				else:
					a += 1

				# df = np.expand_dims(prob, axis = (0, 1)) + np.expand_dims(prob, axis = (0, 2)) + np.expand_dims(prob, axis = (1, 2))
				# exp = np.sum(df) / 3 / quantile_num ** 3 
				# exp_2 = np.sum(df * df / 9) / quantile_num ** 3
		
			a, b = 1, 1
			prob = (np.arange(quantile_num)) / float(quantile_num)

				

			var_result.append(result)
			var_result1.append(result1)
			exp_result.append(e_result)
			exp_result1.append(e_result1)


		var_result = np.array(var_result)
		var_result1 = np.array(var_result1)
		exp_result = np.array(exp_result)
		exp_result1 = np.array(exp_result1)
		

		






	
		ratio = 0.7
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		x = np.arange(length) + 1


		ci = np.std(var_result, axis = 0) / np.sqrt(var_result.shape[0])
				
		plt.plot(x, np.mean(var_result, axis = 0), label = "Bayesian", c = palette[0])
		plt.fill_between(x, np.mean(var_result, axis = 0) + ci, np.mean(var_result, axis = 0) - ci, color = palette[0], alpha = 0.1)
		ci = np.std(var_result1, axis = 0) / np.sqrt(var_result1.shape[0])

		plt.plot(x, np.mean(var_result1, axis = 0), label = "Distributional RPE", c = palette[1])
		plt.fill_between(x, np.mean(var_result1, axis = 0) + ci, np.mean(var_result1, axis = 0) - ci, color = palette[1], alpha = 0.1)

		
		
		plt.legend(loc="upper right", frameon = False)
		plt.xlabel("Trial")
		plt.ylabel("Variance")
		plt.xscale('log')

		
		sns.despine()
		plt.savefig("fig/experiment{}_variance.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.7
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		x = np.arange(length) + 1


		ci = np.std(exp_result, axis = 0) / np.sqrt(exp_result.shape[0])
				
		plt.plot(x, np.mean(exp_result, axis = 0), label = "Bayesian", c = palette[0])
		plt.fill_between(x, np.mean(exp_result, axis = 0) + ci, np.mean(exp_result, axis = 0) - ci, color = palette[0], alpha = 0.1)
		ci = np.std(exp_result1, axis = 0) / np.sqrt(exp_result1.shape[0])

		plt.plot(x, np.mean(exp_result1, axis = 0), label = "Distributional RPE", c = palette[1])
		plt.fill_between(x, np.mean(exp_result1, axis = 0) + ci, np.mean(exp_result1, axis = 0) - ci, color = palette[1], alpha = 0.1)

		
		
		plt.legend(loc="upper right", frameon = False)
		plt.xlabel("Trial")
		plt.ylabel("Expectation")
		plt.xscale("log")

		
		sns.despine()
		plt.savefig("fig/experiment{}_expectation.pdf".format(num), transparent = True)
		plt.close()




	
	if num == 8:
		ratio = 0.7

		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())

		
		
		
		max_trial = len(data["action"][agents[0]][0, 0, :])
		t = np.arange(max_trial)


		
		delta = [ data["task"][x][i, 0, 1, 0] - data["task"][x][i, 0, 0, 0] for i, x in enumerate(data["task"].keys())]
		task_num = len(delta)
	
		label = np.linspace(0.5, 0.2, len(delta))
		label = ["{0:.2f}".format(x) for x in label]
		
		x = np.arange(len(delta))
		

		print("There are four stationary A-AFC tasks tested in this experiment with A=[2,2, 2, 2], Delta = {} respectively".format(label))
		print("Here is the models tested: {}".format(agents))

		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.25
		multiplier = 0
		regret_data = []

		for i, a in enumerate(agents):
			error_bar = np.std(data["regret"][a][:, :, -1], axis = 1) / np.sqrt(data["regret"][a][:, :, -1].shape[1])
			# offset  = width * multiplier
			offset  = width * multiplier 
			# rects = ax.bar(x + offset, np.mean(data["regret"][a][idx, :, -1], axis = 1), width, label = a, yerr = error_bar)
			if a == "Distributional RPE Model":
				
				rects = ax.bar(x + offset, np.mean(data["regret"][a][:, :, -1], axis = 1), width, label = "Our Model", yerr = error_bar)
			else:
				rects = ax.bar(x + offset, np.mean(data["regret"][a][:, :, -1], axis = 1), width, label = a, yerr = error_bar)
			regret_data.append(data["regret"][a][:, :, -1])

			# for j in range(len(delta)):
			#  	ax.boxplot(data["regret"][a][j, :, -1], positions = [x[j]+offset])
			
			multiplier += 1
			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		
		for i in range(len(delta)):
		# 	y_max = np.max([np.mean(data["regret"][a][i, :, -1]) +  np.std(data["regret"][a][i, :, -1]) / np.sqrt(len(data["regret"][a][i, :, -1]))  for a in agents[1:]])
		# 	y_min = np.min([data["regret"][a][i, :, -1] for a in agents[1:]])
			
		# 	ax.annotate("", xy=(j+0.25, y_max + 2), xycoords='data',
	       # xytext=(j+0.5, y_max + 2), textcoords='data',
	       # arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	       #                 connectionstyle="bar,fraction=0.2"))
			# z, p = scipy.stats.mannwhitneyu(data["regret"][agents[1]][i, :, -1], data["regret"][agents[2]][i, :, -1])

			y_max = np.max([np.mean(data["regret"][a][i, :, -1]) + np.std(data["regret"][a][i, :, -1]) / np.sqrt(len(data["regret"][a][i, :, -1]))  for a in agents])
			y_min = np.min([data["regret"][a][i, :, -1] for a in agents])
			
			ax.annotate("", xy=(i+width / 2, y_max + 1), xycoords='data',
	       xytext=(i+width * 3 /2, y_max + 1), textcoords='data',
	       arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	                       connectionstyle="bar,fraction=0.2"))
			res = scipy.stats.shapiro(regret_data[0][i])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = 2, Delta = {} is {}".format(agents[0],label[i],  res.pvalue))
			
			res = scipy.stats.shapiro(regret_data[1][i])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = 2, Delta = {} is {}".format(agents[1],label[i],  res.pvalue))
			
			res = scipy.stats.shapiro(regret_data[2][i])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = 2, Delta = {} is {}".format(agents[2],label[i],  res.pvalue))
			

			z, p = scipy.stats.kruskal(regret_data[0][i], regret_data[1][i], regret_data[2][i])
			print("The p value of Kruskal Wallis test on the accumulated regret in environment A = 2, Delta = {} is {}".format(label[i], p))
		 
			
			p = sp.posthoc_dunn([x[i] for x in regret_data], p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the accumulated regret in environment A = 2, Delta = {} is detailed in the following matrix:".format(label[i]))
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			
			print(p)
			
		# plt.legend(loc="upper right", frameon = False)
		plt.legend(loc="upper left", frameon = False)
		ax.set_xticks(x + width, labels = label)
		plt.xlabel("\u0394")
		plt.ylabel("Regret")
		plt.ylim(bottom = 0)
		
		# plt.title("Averaged regret over different \u0394 between two alternatives".format(max_trial))
		sns.despine()
		plt.tight_layout()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.25
		multiplier = 0

		accuracy = {}
		accuracy_data = []
		for i, a in enumerate(agents):
			
			# offset  = width * multiplier
			offset  = width * multiplier 
			# rects = ax.bar(x + offset, np.mean(data["regret"][a][idx, :, -1], axis = 1), width, label = a, yerr = error_bar)
			accuracy[a] = np.copy(data["action"][a])
			accuracy[a][accuracy[a] < 1] = 0
			accuracy[a][accuracy[a] == 1] = 1
			
			accuracy[a] = np.mean(accuracy[a], axis = 2) 
			error_bar = np.std(accuracy[a], axis = 1) / np.sqrt(accuracy[a].shape[1])

			if a == "Distributional RPE Model":
				
				rects = ax.bar(x + offset, np.mean(accuracy[a], axis = 1), width, label = "Our Model", yerr = error_bar)
			else:
				rects = ax.bar(x + offset, np.mean(accuracy[a], axis = 1), width, label = a, yerr = error_bar)
			accuracy_data.append(accuracy[a])
			# for j in range(len(delta)):
			#   	ax.boxplot(accuracy[a][j], positions = [x[j]+offset])

			
			
			multiplier += 1
			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		for i in range(len(delta)):
		# 	y_max = np.max([np.mean(data["regret"][a][i, :, -1]) +  np.std(data["regret"][a][i, :, -1]) / np.sqrt(len(data["regret"][a][i, :, -1]))  for a in agents[1:]])
		# 	y_min = np.min([data["regret"][a][i, :, -1] for a in agents[1:]])
			
		# 	ax.annotate("", xy=(j+0.25, y_max + 2), xycoords='data',
	       # xytext=(j+0.5, y_max + 2), textcoords='data',
	       # arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	       #                 connectionstyle="bar,fraction=0.2"))
			# z, p = scipy.stats.mannwhitneyu(data["regret"][agents[1]][i, :, -1], data["regret"][agents[2]][i, :, -1])

			y_max = np.max([np.mean(accuracy[a][i])  for a in agents])
		
			
			ax.annotate("", xy=(i+width / 2, y_max + 1), xycoords='data',
	       xytext=(i+width * 3 /2, y_max + 1), textcoords='data',
	       arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	                       connectionstyle="bar,fraction=0.2"))
			res = scipy.stats.shapiro(accuracy_data[0][i])
			print("The p value of shapiro test on the accuracy per run of {} in environment A = 2, Delta = {} is {}".format(agents[0],label[i],  res.pvalue))
	
			res = scipy.stats.shapiro(accuracy_data[1][i])
			print("The p value of shapiro test on the accuracy per run of {} in environment A = 2, Delta = {} is {}".format(agents[1],label[i],  res.pvalue))
	
			res = scipy.stats.shapiro(accuracy_data[2][i])
			print("The p value of shapiro test on the accuracy per run of {} in environment A = 2, Delta = {} is {}".format(agents[2],label[i],  res.pvalue))
	

			z, p = scipy.stats.kruskal(accuracy_data[0][i], accuracy_data[1][i], accuracy_data[2][i])
			print("The p value of Kruskal Wallis test on the accuracy per run in environment A = 2, Delta = {} is {}".format(label[i], p))
		 
			
			p = sp.posthoc_dunn([x[i] for x in accuracy_data], p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the accuracy per run in environment A = 2, Delta = {} is detailed in the following matrix:".format(label[i]))
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			
			print(p)

			
			
		# plt.legend(loc="upper right", frameon = False)
		plt.legend(loc="upper left", frameon = False)
		ax.set_xticks(x + width, labels = label)
		plt.xlabel("\u0394")
		plt.ylabel("Accuracy")
		ylim = 0.93
		plt.ylim(bottom = ylim)
		plt.ylim(top = 1)
		
		# plt.title("Averaged regret over different \u0394 between two alternatives".format(max_trial))
		sns.despine()
		plt.tight_layout()
		plt.savefig("fig/experiment{}_accuracy.pdf".format(num), transparent = True)
		plt.close()

		for i in range(task_num):
			
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			for j, a in enumerate(agents):
				
				accuracy = np.copy(data["action"][a][i])
				accuracy[accuracy < 1] = 0
				accuracy[accuracy == 1] = 1
				choice_prob = np.mean(accuracy, axis = 0)
				ci = np.std(accuracy, axis = 0) / np.sqrt(accuracy.shape[0])
				
				plt.plot(t + 1, choice_prob, label = a, c = palette[j])
				plt.fill_between(t + 1, choice_prob + ci, choice_prob - ci, color = palette[j], alpha = 0.1)
			#plt.xscale('log')
			plt.legend(loc="lower right", frameon=False)
			plt.xlabel("Trial")
			plt.ylim(bottom = 0.15, top = 1)
			plt.ylabel("Accurate choice probability")
			plt.xscale("log")
			ax.set_xlim(right = max_trial)
			plt.title("Choice probability over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_task_{}action.pdf".format(num, i), transparent = True)
			plt.close()

			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)

			accuracy_data = []
			label_data = []
			for j, a in enumerate(agents):
				accuracy =  np.copy(data["action"][a][i])

				accuracy[accuracy != 1] = 0
				accuracy[accuracy == 1] = 1
				accuracy = np.mean(accuracy, axis = 1)

				accuracy_data.append(accuracy)
				label_data.append(a)
			ax.boxplot(accuracy_data, sym = '', widths = 0.7, showcaps = False, 
	                     vert=True,  # vertical box alignment
	                     labels=label_data)  # will be used to label x-ticks
			res = scipy.stats.shapiro(accuracy_data[0])
			print("The p value of shapiro test on the accuracy per run of {} in environment A = 2, Delta = {} is {}".format(agents[0],label[i],  res.pvalue))
	
			res = scipy.stats.shapiro(accuracy_data[1])
			print("The p value of shapiro test on the accuracy per run of {} in environment A = 2, Delta = {} is {}".format(agents[1],label[i],  res.pvalue))
	
			res = scipy.stats.shapiro(accuracy_data[2])
			print("The p value of shapiro test on the accuracy per run of {} in environment A = 2, Delta = {} is {}".format(agents[2],label[i],  res.pvalue))
	
			
			z, p = scipy.stats.kruskal(accuracy_data[0], accuracy_data[1], accuracy_data[2])
			print("The p value of Kruskal Wallis test on the accuracy per run in environment A = 2, Delta = {} is {}".format(label[i], p))
		 
			p = sp.posthoc_dunn(accuracy_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the accuracy per run in environment A = 2, Delta = {} is detailed in the following matrix:".format(label[i]))
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			
			print(p)


			print("accuracy for {} = {}, sem = {}".format(agents[0], np.mean(accuracy_data[0]), np.std(accuracy_data[0]) / np.sqrt(len(accuracy_data[0]))))
			print("accuracy for {} = {}, sem = {}".format(agents[1],  np.mean(accuracy_data[1]), np.std(accuracy_data[1]) / np.sqrt(len(accuracy_data[1]))))
			print("accuracy for {} = {}, sem = {}".format(agents[2],  np.mean(accuracy_data[2]), np.std(accuracy_data[2]) / np.sqrt(len(accuracy_data[2]))))

			
				
			plt.legend(loc="upper left", frameon=False)
			plt.xlabel("Trial")
			plt.ylabel("Accuracy")
			plt.title("Accuracy over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_task_{}_accuracy_box.pdf".format(num, i), transparent = True)
			plt.close()

			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)

			regret_data = []
			label_data = []
			for j, a in enumerate(agents):
				regret =  np.copy(data["regret"][a][i])


				regret_data.append(regret[:, -1])
				label_data.append(a)
			ax.boxplot(regret_data, sym = '', widths = 0.7, showcaps = False, 
	                     vert=True,  # vertical box alignment
	                     labels=label_data)  # will be used to label x-ticks

			res = scipy.stats.shapiro(regret_data[0])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = 2, Delta = {} is {}".format(agents[0],label[i],  res.pvalue))
	
			res = scipy.stats.shapiro(regret_data[1])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = 2, Delta = {} is {}".format(agents[1],label[i],  res.pvalue))
	
			res = scipy.stats.shapiro(regret_data[2])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = 2, Delta = {} is {}".format(agents[2],label[i],  res.pvalue))
	
			
			
			z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
			print("The p value of Kruskal Wallis test on the accumulated regret in environment A = 2, Delta = {} is {}".format(label[i], p))
		 
			
			p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the accumulated regret in environment A = 2, Delta = {} is detailed in the following matrix:".format(label[i]))
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			
			print(p)


			print("regret for {} = {}, sem = {}".format(agents[0], np.mean(regret_data[0]), np.std(regret_data[0]) / np.sqrt(len(regret_data[0]))))
			print("regret for {} = {}, sem = {}".format(agents[1],  np.mean(regret_data[1]), np.std(regret_data[1]) / np.sqrt(len(regret_data[1]))))
			print("regret for {} = {}, sem = {}".format(agents[2],  np.mean(regret_data[2]), np.std(regret_data[2]) / np.sqrt(len(regret_data[2]))))

			
				
			plt.legend(loc="upper left", frameon=False)
			plt.xlabel("Trial")
			plt.ylabel("Accumulated regret")
			plt.title("Averaged accumulated regret over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_task_{}_regret_box.pdf".format(num, i), transparent = True)
			plt.close()






			
	if num == 12:
		ratio = 0.7
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		max_trial = len(data["action"][agents[0]][0, :])
		t = np.arange(max_trial)



		
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		

		for j, a in enumerate(agents):
			ax.scatter(j + 0.8 * (np.random.random(data["regret"][a][:, -1].shape) -0.5), data["regret"][a][:, -1], label = a, color = palette[j])

		plt.legend(loc="upper left", frameon = False)
		sns.despine()
		plt.tight_layout()
		plt.savefig("fig/experiment{}_regret_task.pdf".format(num), transparent = True)
		print("hi")
		plt.close()


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		regret_data = []
		label_data = []
		for i, a in enumerate(agents):
			regret_data.append(data["regret"][a][:, -1])
			label_data.append(a)
		ax.boxplot(regret_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data, showfliers = True)  # will be used to label x-ticks
		
		res = scipy.stats.shapiro(regret_data[0])
		print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[0], res.pvalue))

		res = scipy.stats.shapiro(regret_data[1])
		print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[1], res.pvalue))

		res = scipy.stats.shapiro(regret_data[2])
		print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[2], res.pvalue))

		z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
		print("The p value of Kruskal Wallis test on the accumulated regret in a probability reversal task is {}".format(p))
	
		p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
		print("The p value of posthoc dunn test with bonferroni correction on the accumulated regret is detailed in the following matrix:")
		print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
	
		print(p)
		
		print("regret for {} = {}, sem = {}".format(agents[0], np.mean(regret_data[0]), np.std(regret_data[0]) / np.sqrt(len(regret_data[0]))))
		print("regret for {} = {}, sem = {}".format(agents[1],  np.mean(regret_data[1]), np.std(regret_data[1]) / np.sqrt(len(regret_data[1]))))
		print("regret for {} = {}, sem = {}".format(agents[2], np.mean(regret_data[2]), np.std(regret_data[2]) / np.sqrt(len(regret_data[2]))))
			


			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret_box.pdf".format(num), transparent = True)
		plt.close()

		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		

		for i, a in enumerate(agents):

			action_data = np.copy(data["action"][a])
			action_data[action_data < 2] = 0
			action_data[action_data == 2] = 1
				
			choice_prob = np.mean(action_data, axis = 0)
			ci =  np.std(action_data, axis = 0) / np.sqrt(action_data.shape[0])
			
			plt.plot(t + 1, choice_prob, label = a, c = palette[i])
			plt.fill_between(t + 1, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
		
			#ax.legend(loc = "upper left")
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		plt.xlim(left = 0)
		plt.legend(loc="upper left", frameon=False)
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		for i, a in enumerate(agents):
			error_bar = np.std(data["regret"][a][:, -1]) / np.sqrt(data["regret"][a][:, -1].shape[0])
			ax.bar(i, np.mean(data["regret"][a][:, -1]),label = a, width = 0.8, yerr = error_bar)

		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret_bar.pdf".format(num), transparent = True)
		plt.close()


		res = scipy.stats.permutation_test((data["regret"][agents[1]][:, -1], data["regret"][agents[0]][:, -1]), lambda x, y: np.mean(x)-np.mean(y), n_resamples=100000)
		print("The mean difference of {}'s to the original model is {} with p value {} in 10^6 resamples permutation test".format(agents[1], res.statistic, res.pvalue))
		res = scipy.stats.permutation_test((data["regret"][agents[2]][:, -1], data["regret"][agents[0]][:, -1]), lambda x, y: np.mean(x)-np.mean(y), n_resamples=100000)
		print("The mean difference of {}'s to the original model is {} with p value {} in 10^6 resamples permutation test".format(agents[2], res.statistic, res.pvalue))
		
			

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		

		for i, a in enumerate(agents):

			action_data = np.copy(data["action"][a])
			action_data[action_data < 2] = 0
			action_data[action_data == 2] = 1
				
			choice_prob = np.mean(action_data, axis = 0)
			ci =  np.std(action_data, axis = 0) / np.sqrt(action_data.shape[0])
			
			plt.plot(t + 1, choice_prob, label = a, c = palette[i])
			plt.fill_between(t + 1, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
		
			#ax.legend(loc = "upper left")
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		plt.xlim(left = 0)
		plt.legend(loc="upper left", frameon=False)
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		plt.close()




	if num == 1 or num == 3 or num == 16:

		ratio = 0.7

		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())

		if num == 16:
			agents = [agents[2], agents[0], agents[1]]
		
		max_trial = len(data["action"][agents[0]][0, 0, :])
		t = np.arange(max_trial)


		
		delta = [ np.abs(data["task"][x][i, 0, -1, 0] - data["task"][x][i, 0, 0, 0]) for i, x in enumerate(data["task"].keys())]
		task_num = len(delta)
	
		label = np.linspace(0.5, 0.2, len(delta))
		label = ["{0:.2f}".format(x) for x in label]
		label = ["{0:.2f}".format(x) for x in delta]
		print("There are four stationary A-AFC tasks tested in this experiment with A=[3, 3, 3, 3], Delta = {} respectively".format(label))
		print("Here is the models tested: {}".format(agents))
		if num == 1:
			print("Distributional RPE model is the d-CS model")
		x = np.arange(len(delta))
		
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.25
		multiplier = 0

		for i, a in enumerate(agents):

			error_bar = np.std(data["regret"][a][:, :, -1], axis = 1) / np.sqrt(data["regret"][a][:, :, -1].shape[1])
			# offset  = width * multiplier
			offset  = width * multiplier
			# rects = ax.bar(x + offset, np.mean(data["regret"][a][idx, :, -1], axis = 1), width, label = a, yerr = error_bar)
			if a == "Distributional RPE Model":
				
				rects = ax.bar(x + offset, np.mean(data["regret"][a][:, :, -1], axis = 1), width, label = "Our Model", yerr = error_bar)
			else:
				rects = ax.bar(x + offset, np.mean(data["regret"][a][:, :, -1], axis = 1), width, label = a, yerr = error_bar)

			# for j in range(len(delta)):
			#  	ax.boxplot(data["regret"][a][j, :, -1], positions = [x[j]+offset])
			
			multiplier += 1
			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		
		for i in range(len(delta)):
		# 	y_max = np.max([np.mean(data["regret"][a][i, :, -1]) +  np.std(data["regret"][a][i, :, -1]) / np.sqrt(len(data["regret"][a][i, :, -1]))  for a in agents[1:]])
		# 	y_min = np.min([data["regret"][a][i, :, -1] for a in agents[1:]])
			
		# 	ax.annotate("", xy=(j+0.25, y_max + 2), xycoords='data',
	       # xytext=(j+0.5, y_max + 2), textcoords='data',
	       # arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	       #                 connectionstyle="bar,fraction=0.2"))
			# z, p = scipy.stats.mannwhitneyu(data["regret"][agents[1]][i, :, -1], data["regret"][agents[2]][i, :, -1])

			y_max = np.max([np.mean(data["regret"][a][i, :, -1]) + np.std(data["regret"][a][i, :, -1]) / np.sqrt(len(data["regret"][a][i, :, -1]))  for a in agents])
			y_min = np.min([data["regret"][a][i, :, -1] for a in agents])
			
			
			res = scipy.stats.shapiro(data["regret"][agents[0]][i, :, -1])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = 3, Delta = {} is {}".format(agents[0],label[i],  res.pvalue))
			res = scipy.stats.shapiro(data["regret"][agents[1]][i, :, -1])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = 3, Delta = {} is {}".format(agents[1],label[i],  res.pvalue))
			res = scipy.stats.shapiro(data["regret"][agents[2]][i, :, -1])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = 3, Delta = {} is {}".format(agents[2],label[i],  res.pvalue))

			regret_data = [data["regret"][a][i, :, -1]  for a in agents]

			z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
			print("The p value of Kruskal Wallis test on the accumulated regret per run in environment A = 3, Delta = {} is {}".format(label[i], p))
		
			p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the accumulated regret is detailed in the following matrix:")
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			print(p)


		# plt.legend(loc="upper right", frameon = False)
		plt.legend(loc="upper left", frameon = False)
		ax.set_xticks(x + width, labels = label)
		plt.xlabel("\u0394")
		plt.ylabel("Regret")
		plt.ylim(bottom = 0)
		
		# plt.title("Averaged regret over different \u0394 between two alternatives".format(max_trial))
		sns.despine()
		plt.tight_layout()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()




		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.25
		multiplier = 0

		accuracy = {}
		for i, a in enumerate(agents):
			
			# offset  = width * multiplier
			offset  = width * multiplier
			# rects = ax.bar(x + offset, np.mean(data["regret"][a][idx, :, -1], axis = 1), width, label = a, yerr = error_bar)
			accuracy[a] = np.copy(data["action"][a])
			accuracy[a][accuracy[a] < 2] = 0
			accuracy[a][accuracy[a] == 2] = 1
			
			accuracy[a] = np.mean(accuracy[a], axis = 2) 
			error_bar = np.std(accuracy[a], axis = 1) / np.sqrt(accuracy[a].shape[1])

			if a == "Distributional RPE Model":
				
				rects = ax.bar(x + offset, np.mean(accuracy[a], axis = 1), width, label = "Our Model", yerr = error_bar)
			else:
				rects = ax.bar(x + offset, np.mean(accuracy[a], axis = 1), width, label = a, yerr = error_bar)

			# for j in range(len(delta)):
			#   	ax.boxplot(accuracy[a][j], positions = [x[j]+offset])

			
			
			multiplier += 1
			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

	
		for i in range(len(delta)):
		# 	y_max = np.max([np.mean(data["regret"][a][i, :, -1]) +  np.std(data["regret"][a][i, :, -1]) / np.sqrt(len(data["regret"][a][i, :, -1]))  for a in agents[1:]])
		# 	y_min = np.min([data["regret"][a][i, :, -1] for a in agents[1:]])
			
		# 	ax.annotate("", xy=(j+0.25, y_max + 2), xycoords='data',
	       # xytext=(j+0.5, y_max + 2), textcoords='data',
	       # arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	       #                 connectionstyle="bar,fraction=0.2"))
			# z, p = scipy.stats.mannwhitneyu(data["regret"][agents[1]][i, :, -1], data["regret"][agents[2]][i, :, -1])

			y_max = np.max([np.mean(accuracy[a][i])  for a in agents])
		
			
			res = scipy.stats.shapiro(accuracy[agents[0]][i])
			print("The p value of shapiro test on the accuracy of {} in environment A = 3, Delta = {} is {}".format(agents[0],label[i],  res.pvalue))
			res = scipy.stats.shapiro(accuracy[agents[1]][i])
			print("The p value of shapiro test on the accuracy of {} in environment A = 3, Delta = {} is {}".format(agents[1],label[i],  res.pvalue))
			res = scipy.stats.shapiro(accuracy[agents[2]][i])
			print("The p value of shapiro test on the accuracy of {} in environment A = 3, Delta = {} is {}".format(agents[2],label[i],  res.pvalue))

			accuracy_data = [accuracy[a][i]  for a in agents]

			z, p = scipy.stats.kruskal(accuracy_data[0], accuracy_data[1], accuracy_data[2])
			print("The p value of Kruskal Wallis test on the accuracy in environment A = 3, Delta = {} is {}".format(label[i], p))
		
			p = sp.posthoc_dunn(accuracy_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the accumulated regret is detailed in the following matrix:")
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			print(p)
		# plt.legend(loc="upper right", frameon = False)
		plt.legend(loc="upper left", frameon = False)
		ax.set_xticks(x + width, labels = label)
		plt.xlabel("\u0394")
		plt.ylabel("Accuracy")
		if num == 1:
			ylim = 0.25
		if num == 3:
			ylim = 0.45
		else:
			ylim = 0.25
		plt.ylim(bottom = ylim)
		
		# plt.title("Averaged regret over different \u0394 between two alternatives".format(max_trial))
		sns.despine()
		plt.tight_layout()
		plt.savefig("fig/experiment{}_accuracy.pdf".format(num), transparent = True)
		plt.close()

		for i in range(task_num):
			
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			for j, a in enumerate(agents):
				
				accuracy = np.copy(data["action"][a][i])
				accuracy[accuracy < 2] = 0
				accuracy[accuracy == 2] = 1
				choice_prob = np.mean(accuracy, axis = 0)
				ci = np.std(accuracy, axis = 0) / np.sqrt(accuracy.shape[0])
				
				plt.plot(t + 1, choice_prob, label = a, c = palette[j])
				plt.fill_between(t + 1, choice_prob + ci, choice_prob - ci, color = palette[j], alpha = 0.1)
			#plt.xscale('log')
			plt.legend(loc="lower right", frameon=False)
			plt.xlabel("Trial")
			plt.ylim(bottom = 0.15, top = 1)
			plt.ylabel("Accurate choice probability")
			plt.xscale("log")
			ax.set_xlim(right = max_trial)
			plt.title("Choice probability over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_task_{}action.pdf".format(num, i), transparent = True)
			plt.close()



			for j, a in enumerate(agents):
				
				if a == "Distributional RPE Model":
					fig, ax = plt.subplots()
					fig.set_figwidth(4.8 * ratio)
					fig.set_figheight(4.8 * ratio)
					quantile_data = np.mean(data["quantile_data"][a][i], axis = 0)[0, :]
					
					class_num, quantile_num, max_trial = quantile_data.shape
					quantile_data = quantile_data.reshape( class_num * quantile_num, max_trial)
					#quantile_data[quantile_data > 1] = 1

					im = ax.imshow(quantile_data, interpolation='nearest', aspect='auto', cmap = "YlGn_r")
					plt.axvline(10, c = "grey", linewidth = 1, linestyle = "dashed")
					plt.axvline(50, c = "grey", linewidth = 1, linestyle = "dashed")
					im.set_clim(0, 1)
					cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal", pad = 0.2)
					cbar.ax.set_xlabel("Activities")

					task_prob =[ "{0:.2f}".format(x) for x in data["task"]["gap {}".format(i)][i, 0, :, 0]]

					# ax.set_xticks(np.arange(0, 2001, 200))
					ax.set_yticks([50, 150, 250])
					
					ax.set_yticklabels(task_prob, rotation = 90)

					
					# plt.legend(loc="upper left")
					plt.xlabel("Trial")
					plt.ylabel("Action")
					plt.title("Striatal activities over {} trials".format(max_trial))
					plt.xscale('log')
					#plt.tight_layout()

					plt.savefig("fig/experiment{}_{}_task{}_quantilemap.pdf".format(num, a, i))
					plt.close()
					


					quantile_data = data["quantile_data"][a][i]

					bins = np.linspace(0, 1, 100)

					timepoints = [10, 50]

					# fig, ax = plt.subplots(len(timepoints),  sharey=True)
					# fig.set_figwidth(4.8 * ratio)
					# fig.set_figheight(4.8 * 2 * ratio)




					for k, ti in enumerate(timepoints):
						opt = {}
						opt["stimuli_num"] = 1
						opt["class_num"] = 3
						opt["gamma"] = 1
						opt["quantile_num"] = 100
						opt["N"] = 50
						opt["K"] = 3
						opt["a"] = 1
						opt["b"] = 1
						opt["a1"] = 0.75
						opt["b1"] = 1
						opt["a2"] = 2
						opt["b2"] = 1
						opt["tau"] = 10
						opt["eta"] = 0.5
						opt["threshold"] = 0.8
						opt["d_interval"] = 1000
						opt["uniform"] = False
						opt["uniform_init"] = False
						opt["epsilon"] = 1
						opt["lr"] = 0.1
						opt["hidden_size"] = 10
						opt["dt"] = 0.005

						agent = NeuralQuantileAgent(opt)
						agent.prob[0, :, :] = np.mean(quantile_data, axis = 0)[0, :, :, ti-1]

						result = []
						for _ in range(100):
							action = agent.forward(0)
							result.append(action)

						result = np.array(result)

						fig1, ax1 = plt.subplots()
						fig1.set_figwidth(4.8 * ratio)
						fig1.set_figheight(4.8 * ratio)

						
						result_data = [np.mean(result == 2), np.mean(result == 1), np.mean(result == 0)]
						print(result_data)

						ax1.bar(0, result_data[0], 0.5)
						ax1.bar(1, result_data[1], 0.5)
						ax1.bar(2, result_data[2], 0.5)
						plt.ylim(top = 1.1)

						
						plt.ylabel("Choice probability")
						ax1.set_xticks(np.arange(3))
						ax1.set_xticklabels(["1", "2", "3"])
						sns.despine()
						plt.savefig("fig/experiment{}_exploration_{}_at_trial_{}.pdf".format(num, i, ti), transparent = True)

						









						# for l in range(3):
						# 	quantile_num = quantile_data.shape[3]

						# 	if ti == 0:
						# 		df = np.arange(quantile_num) * 1.0 / quantile_num
						# 	else:

						# 		df = np.mean(quantile_data, axis = 0)[0, l, :, ti-1]

						# 	# df = (np.expand_dims(df, 0) + np.expand_dims(df, 1))/2
						# 	# df = df.flatten()

						# 	df = (np.expand_dims(df, (0, 1)) + np.expand_dims(df, (1, 2)) + np.expand_dims(df, (0, 2)))/3
						# 	df = df.flatten()

						# 	diff = np.expand_dims(bins, 1) - np.expand_dims(df, 0)
						# 	std = 0.03
						# 	q_data = np.sum(np.exp(-0.5 * diff ** 2 / std ** 2) / (np.sqrt(2 * np.pi) * std), axis = 1) / len(df)
						# 	if l == 0:
						# 		ax[k].plot(bins, q_data, label = "{}".format(l+1), color = palette_alt[l])
						# 	if l == 1:
						# 		ax[k].plot(bins, q_data, label = "{}".format(l+1), linestyle = "dashed", color = palette_alt[l])
						# 	if l == 2:
						# 		ax[k].plot(bins, q_data, label = "{}".format(l+1), linestyle = "dashed", color = palette_alt[l])


					# 	ax[k].title.set_text("Posterior at trial {}".format(ti))
						
					# handles, labels = ax[0].get_legend_handles_labels()
					
					# fig.legend(handles, labels,  bbox_to_anchor=(1, 0.95), loc = "upper right", frameon=False)
					# sns.despine()
					# fig.supxlabel("Values", x= 0.625, y = 0.06,  fontsize = "small")
					# plt.tight_layout()
					
					
					#plt.title("Averaged accumulated regret over {} trials".format(max_trial))
					
					#plt.show()
					# plt.savefig("fig/experiment{}_posterior_task{}.pdf".format(num, i), transparent = True)
					# plt.close()

					fig, ax = plt.subplots()
					fig.set_figwidth(4.8)
					fig.set_figheight(4.8)

					quantile_df = np.mean(quantile_data, axis = 0)[0, 2, :, 9]
					

					opt = {}
					opt["stimuli_num"] = 1
					opt["class_num"] = 3
					opt["gamma"] = 1
					opt["quantile_num"] = 100
					opt["N"] = 50
					opt["K"] = 3
					opt["a"] = 1
					opt["b"] = 1
					opt["a1"] = 0.75
					opt["b1"] = 1
					opt["a2"] = 2
					opt["b2"] = 1
					opt["tau"] = 10
					opt["eta"] = 0.5
					opt["threshold"] = 0.8
					opt["d_interval"] = 1000
					opt["uniform"] = False
					opt["uniform_init"] = False
					opt["epsilon"] = 1
					opt["lr"] = 0.1
					opt["hidden_size"] = 10
					opt["dt"] = 0.005

					agent = NeuralQuantileAgent(opt)
					agent.prob[0, 0, :] = quantile_df

					df = quantile_df
					result = []
					for _ in range(100):
						agent.forward(0)
						result.append(1.0 / 3 * np.sum(agent.value_neurons[:, 0]))
					result = np.array(result)
					

					df = (np.expand_dims(df, (0, 1)) + np.expand_dims(df, (1, 2)) + np.expand_dims(df, (0, 2)))/3
					df = df.flatten()

					diff = np.expand_dims(bins, 1) - np.expand_dims(df, 0)
					print(diff.shape)
					std = 0.03
					q_data = np.sum(np.exp(-0.5 * diff ** 2 / std ** 2) / (np.sqrt(2 * np.pi) * std), axis = 1) / len(df)

					diff = np.expand_dims(bins, 1) - np.expand_dims(result, 0)
					print(diff.shape)
					std = 0.03
					q_data_1 = np.sum(np.exp(-0.5 * diff ** 2 / std ** 2) / (np.sqrt(2 * np.pi) * std), axis = 1) / len(result)

					ax.plot(bins, q_data, label = "Actual distribution", color = palette_alt[0])
					ax.plot(bins, q_data_1, label = "Empirical distribution", color = palette_alt[1])
					plt.xlabel("Values")
					plt.ylabel("Probability density")
					plt.legend(loc="upper left", frameon = False)
					sns.despine()
					#plt.xscale('log')
					plt.tight_layout()

					plt.savefig("fig/experiment{}_sampling_mechanisms_at_trial_10_task_{}.pdf".format(num, i))
					plt.close()


					if i == 1:

						K = [1, 3, 9, 27, 81]

						fig, ax = plt.subplots()
						fig.set_figwidth(4.8)
						fig.set_figheight(4.8)

						for j, k in enumerate(K):
							print(k)
							opt["K"] = k
							agent = NeuralQuantileAgent(opt)
							agent.prob[0, 0, :] = quantile_df
							result = []
							for _ in range(500):
								agent.forward(0)
								result.append(1.0 / k * np.sum(agent.value_neurons[:, 0]))
							result = np.array(result)
							diff = np.expand_dims(bins, 1) - np.expand_dims(result, 0)
							print(diff.shape)
							std = 0.03
							q_data = np.sum(np.exp(-0.5 * diff ** 2 / std ** 2) / (np.sqrt(2 * np.pi) * std), axis = 1) / len(result)

							ax.plot(bins, q_data, label = "K = {}".format(k), color = palette_alt[j])

						plt.xlabel("Values")
						plt.ylabel("Probability density")
						plt.legend(loc="upper left", frameon = False)
						sns.despine()
						#plt.xscale('log')
						plt.tight_layout()

						plt.savefig("fig/experiment{}_empirical_posterior_for_varying_K.pdf".format(num))
						plt.close()

















					# fig, ax = plt.subplots(len(timepoints),  sharey=True)
					# fig.set_figwidth(4.8)
					# fig.set_figheight(4.8 * 2)

					# for k, ti in enumerate(timepoints):
					# 	for l in range(3):
					# 		quantile_num = quantile_data.shape[3]

					# 		if ti == 0:
					# 			df = np.arange(quantile_num) * 1.0 / quantile_num
					# 		else:

					# 			df = quantile_data[0, 0, l, :, ti-1]

					# 		# df = (np.expand_dims(df, 0) + np.expand_dims(df, 1))/2
					# 		# df = df.flatten()

					# 		df = (np.expand_dims(df, (0, 1)) + np.expand_dims(df, (1, 2)) + np.expand_dims(df, (0, 2)))/3
					# 		df = df.flatten()

					# 		diff = np.expand_dims(bins, 1) - np.expand_dims(df, 0)
					# 		std = 0.03
					# 		q_data = np.sum(np.exp(-0.5 * diff ** 2 / std ** 2) / (np.sqrt(2 * np.pi) * std), axis = 1) / len(df)
					# 		if l == 0:
					# 			ax[k].plot(bins, q_data, label = "{}".format(l+1), color = palette_alt[l])
					# 		if l == 1:
					# 			ax[k].plot(bins, q_data, label = "{}".format(l+1), linestyle = "dashed", color = palette_alt[l])
					# 		if l == 2:
					# 			ax[k].plot(bins, q_data, label = "{}".format(l+1), linestyle = "dashed", color = palette_alt[l])


					# 	ax[k].title.set_text("Posterior at trial {}".format(ti))
						
					# handles, labels = ax[0].get_legend_handles_labels()
					
					# fig.legend(handles, labels,  bbox_to_anchor=(1, 0.95), loc = "upper right", frameon=False)
					# sns.despine()
					# fig.supxlabel("Values", x= 0.625, y = 0.06,  fontsize = "small")
					# fig.supylabel("Probability density", x = 0.12, fontsize = "small")
					# plt.tight_layout()
					
					
					# #plt.title("Averaged accumulated regret over {} trials".format(max_trial))
					
					# #plt.show()
					# plt.savefig("fig/experiment{}_posterior_task{}_sample_{}.pdf".format(num, i, a), transparent = True)
					# plt.close()




		for i, a in enumerate(agents):

			for j in range(3):
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)


				action_data = data["action"][a][2, j, :]
			
				plt.scatter((t+1), action_data + 0.05 * np.random.normal(0, 1, 500), marker = 'x') 
				plt.xlabel("Trials")
				plt.ylabel("Action")
				ax.set_yticks([0, 1])
				sns.despine()
				plt.tight_layout()
				plt.savefig("fig/experiment{}_sample_action_{}_{}.pdf".format(num,a, j), transparent = True)
				plt.close()

				ratio = 0.8
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)

				if a == "Distributional RPE Model":
					quantile_data = data["quantile_data"][a][0][j, 0,  :]
					class_num, quantile_num, max_trial = quantile_data.shape
					quantile_data = quantile_data.reshape( class_num * quantile_num, max_trial)
					#quantile_data[quantile_data > 1] = 1
					plt.xscale('log')

					im = ax.imshow(quantile_data, interpolation='nearest', aspect='auto', cmap = "YlGn_r")
					plt.axvline(5, c = "grey", linewidth = 1, linestyle = "dashed")
					plt.axvline(10, c = "grey", linewidth = 1, linestyle = "dashed")
					im.set_clim(0, 1)
					cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal", pad = 0.2)
					cbar.ax.set_xlabel("Activities")


					# ax.set_xticks(np.arange(0, 2001, 200))
					ax.set_yticks([50, 150])
					ax.set_yticklabels(["Left", "Right"], rotation = 90)
					
					# plt.legend(loc="upper left")
					plt.xlabel("Trial")
					plt.ylabel("Action")
					plt.title("Striatal activities over {} trials".format(max_trial))
					#plt.xscale('log')
					#plt.tight_layout()

					plt.savefig("fig/experiment{}_{}_sample_quantilemap_{}.pdf".format(num, a, j))
					plt.close()




	if num == 2 or num == 4:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		# agents = [agents[1], agents[0], agents[2]]
		
		max_trial = len(data["action"][agents[0]][0, 0])
		
		t = np.arange(max_trial)

		delta = [ i+3 for i, x in enumerate(data["task"].keys())]
		task_num = len(delta)
		idx = np.arange(len(delta))
		ax = plt.figure().gca()
		ax.xaxis.get_major_locator().set_params(integer=True)

		print("There are four stationary A-AFC tasks tested in this experiment with A= {}, Delta = [0.4, 0.4, 0.4, 0.4] respectively".format(delta))
		print("Here is the models tested: {}".format(agents))
		if num == 2:
			print("Distributional RPE model is the d-CS model")
		
		ratio = 0.7
		fig, ax = plt.subplots(layout='constrained')
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.33
		multiplier = 0
		x = np.arange(len(delta))
		
		for i, a in enumerate(agents):
			error_bar = np.std(data["regret"][a][:, :, -1], axis = 1) / np.sqrt(data["regret"][a][:, :, -1].shape[1])

			# offset  = width * multiplier
			offset  = width * multiplier
			# rects = ax.bar(x + offset, np.mean(data["regret"][a][idx, :, -1], axis = 1), width, label = a, yerr = error_bar)
			if a == "Distributional RPE Model":
				
				rects = ax.bar(x + offset, np.mean(data["regret"][a][idx, :, -1], axis = 1), width, label = "Our Model", yerr = error_bar)
			else:
				rects = ax.bar(x + offset, np.mean(data["regret"][a][idx, :, -1], axis = 1), width, label = a, yerr = error_bar)
			# for j in range(len(delta)):
			#  	ax.boxplot(data["regret"][a][j, :, -1], positions = [x[j]+offset])

			multiplier += 1


			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a, alpha = 0.8)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		for i in x:

			# 	y_max = np.max([np.mean(data["regret"][a][i, :, -1]) +  np.std(data["regret"][a][i, :, -1]) / np.sqrt(len(data["regret"][a][i, :, -1]))  for a in agents[1:]])
		# 	y_min = np.min([data["regret"][a][i, :, -1] for a in agents[1:]])
			
		# 	ax.annotate("", xy=(j+0.25, y_max + 2), xycoords='data',
	       # xytext=(j+0.5, y_max + 2), textcoords='data',
	       # arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	       #                 connectionstyle="bar,fraction=0.2"))
			# z, p = scipy.stats.mannwhitneyu(data["regret"][agents[1]][i, :, -1], data["regret"][agents[2]][i, :, -1])
			
			res = scipy.stats.shapiro(data["regret"][agents[0]][i, :, -1])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = {}, Delta = 0.4 is {}".format(agents[0],delta[i],  res.pvalue))
	
			res = scipy.stats.shapiro(data["regret"][agents[1]][i, :, -1])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = {}, Delta = 0.4 is {}".format(agents[1],delta[i],  res.pvalue))
	
			res = scipy.stats.shapiro(data["regret"][agents[2]][i, :, -1])
			print("The p value of shapiro test on the accumulated regret of {} in environment A = 3, Delta = {} is {}".format(agents[2],delta[i],  res.pvalue))

			regret_data = [data["regret"][a][i, :, -1]  for a in agents]

			z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
			print("The p value of Kruskal Wallis test on the accumulated regret per run in environment A = 3, Delta = {} is {}".format(delta[i], p))
		
			p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the accumulated regret is detailed in the following matrix:")
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			print(p)

		
		plt.legend(loc="upper left", frameon = False)
		plt.xlabel("A")
		plt.ylabel("Regret")
		plt.ylim(bottom = 0)
		ax.set_xticks(x + width, delta)
		#plt.title("Averaged regret over different number of alternatives".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		print("num: {}".format(num))
		plt.close()

		for i in range(len(delta)):
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)

			

			for j, a in enumerate(agents):
				ax.scatter(j + 0.8 * (np.random.random(data["regret"][a][i, :, -1].shape) -0.5), data["regret"][a][i, :, -1], label = a, color = palette[j])

			plt.legend(loc="upper left", frameon = False)
			sns.despine()
			plt.tight_layout()
			plt.savefig("fig/experiment{}_regret_task_{}.pdf".format(num, delta[i]), transparent = True)
			print("hi")
			plt.close()


			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)

			regret_data = []

			for j, a in enumerate(agents):
				regret_data.append(data["regret"][a][i, :, -1])
				
			ax.boxplot(regret_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=agents)  #

			plt.legend(loc="upper left", frameon = False)
			sns.despine()
			plt.tight_layout()
			plt.savefig("fig/experiment{}_regret_box_task_{}.pdf".format(num, delta[i]), transparent = True)
			print("hi")
			plt.close()


			


		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.33
		multiplier = 0

		accuracy = {}
		for i, a in enumerate(agents):
			
			# offset  = width * multiplier
			offset  = width * multiplier 
			# rects = ax.bar(x + offset, np.mean(data["regret"][a][idx, :, -1], axis = 1), width, label = a, yerr = error_bar)
			accuracy[a] = np.copy(data["action"][a])
			for j in range(task_num):
				accuracy[a][j][accuracy[a][j] != j+2] = 0
				accuracy[a][j][accuracy[a][j] == j+2] = 1
			
			accuracy[a] = np.mean(accuracy[a], axis = 2) 
			error_bar = np.std(accuracy[a], axis = 1) / np.sqrt(accuracy[a].shape[1])

			if a == "Distributional RPE Model":
				
				rects = ax.bar(x + offset, np.mean(accuracy[a], axis = 1), width, label = "Our Model", yerr = error_bar)
			else:
				rects = ax.bar(x + offset, np.mean(accuracy[a], axis = 1), width, label = a, yerr = error_bar)

			
			
			multiplier += 1
			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		for i in range(len(delta)):
		# 	y_max = np.max([np.mean(data["regret"][a][i, :, -1]) +  np.std(data["regret"][a][i, :, -1]) / np.sqrt(len(data["regret"][a][i, :, -1]))  for a in agents[1:]])
		# 	y_min = np.min([data["regret"][a][i, :, -1] for a in agents[1:]])
			
		# 	ax.annotate("", xy=(j+0.25, y_max + 2), xycoords='data',
	       # xytext=(j+0.5, y_max + 2), textcoords='data',
	       # arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	       #                 connectionstyle="bar,fraction=0.2"))
			# z, p = scipy.stats.mannwhitneyu(data["regret"][agents[1]][i, :, -1], data["regret"][agents[2]][i, :, -1])

			
			res = scipy.stats.shapiro(accuracy[agents[0]][i])
			print("The p value of shapiro test on the accuracy per run of {} in environment A = {}, Delta = 0.4 is {}".format(agents[0],delta[i],  res.pvalue))
	
			res = scipy.stats.shapiro(accuracy[agents[1]][i])
			print("The p value of shapiro test on the accuracy per run of {} in environment A = {}, Delta = 0.4 is {}".format(agents[1],delta[i],  res.pvalue))
	
			res = scipy.stats.shapiro(accuracy[agents[2]][i])
			print("The p value of shapiro test on the accuracy per run of {} in environment A = 3, Delta = {} is {}".format(agents[2],delta[i],  res.pvalue))

			accuracy_data = [accuracy[a][i]  for a in agents]

			z, p = scipy.stats.kruskal(accuracy_data[0], accuracy_data[1], accuracy_data[2])
			print("The p value of Kruskal Wallis test on the accuracy in environment A = 3, Delta = {} is {}".format(delta[i], p))
		
			p = sp.posthoc_dunn(accuracy_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the accuracy is detailed in the following matrix:")
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			print(p)

		# plt.legend(loc="upper right", frameon = False)
		plt.legend(loc="upper left", frameon = False)
		ax.set_xticks(x + width, labels = delta)
		plt.xlabel("\u0394")
		plt.ylabel("Accuracy")
		if num == 2:
			ylim = 0.575
		else:
			ylim = 0.375
		plt.ylim(bottom = ylim)
		
		
		# plt.title("Averaged regret over different \u0394 between two alternatives".format(max_trial))
		sns.despine()
		plt.tight_layout()
		plt.savefig("fig/experiment{}_accuracy.pdf".format(num), transparent = True)
		plt.close()

		for i, a in enumerate(agents):
			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			
			trial_num =500
			for i, a in enumerate(agents):
				idx = 2
				action_accuracy = np.copy(data["action"][a][idx])
				action_accuracy[action_accuracy != (idx+2)] = 0
				action_accuracy[action_accuracy == (idx+2)] = 1
				choice_prob = np.mean(action_accuracy, axis = 0)[:trial_num]
				ci =  np.std(action_accuracy, axis = 0)[:trial_num] / np.sqrt(action_accuracy.shape[0])
				
				plt.plot(np.arange(trial_num) + 1, choice_prob, label = a, c = palette[i])
				plt.fill_between(np.arange(trial_num) + 1, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
			plt.xscale('log')
			plt.legend(loc="lower right", frameon=False)
			plt.xlabel("Trial")
			plt.ylabel("Accurate choice probability")
			ax.set_xlim(right = trial_num)
			plt.title("Choice probability over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
			plt.close()

		for i in range(task_num):
			
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			for j, a in enumerate(agents):
				
				accuracy = np.copy(data["action"][a][i])
				accuracy[accuracy < delta[i]-1] = 0
				accuracy[accuracy == delta[i]-1] = 1
				choice_prob = np.mean(accuracy, axis = 0)
				ci = np.std(accuracy, axis = 0) / np.sqrt(accuracy.shape[0])
				
				plt.plot(t + 1, choice_prob, label = a, c = palette[j])
				plt.fill_between(t + 1, choice_prob + ci, choice_prob - ci, color = palette[j], alpha = 0.1)
			#plt.xscale('log')
			plt.legend(loc="lower right", frameon=False)
			plt.xlabel("Trial")
			plt.ylabel("Accurate choice probability")
			plt.xscale("log")
			plt.ylim(bottom = 0, top = 1)
			ax.set_xlim(right = max_trial)
			plt.title("Choice probability over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_task_{}action.pdf".format(num, i), transparent = True)
			plt.close()



			for j, a in enumerate(agents):
				
				if a == "Distributional RPE Model":
					fig, ax = plt.subplots()
					fig.set_figwidth(4.8 * ratio)
					fig.set_figheight(4.8 * ratio)
					quantile_data = np.mean(data["quantile_data"][a][i], axis = 0)[0, :]
					
					class_num, quantile_num, max_trial = quantile_data.shape
					quantile_data = quantile_data.reshape( class_num * quantile_num, max_trial)
					#quantile_data[quantile_data > 1] = 1

					im = ax.imshow(quantile_data, interpolation='nearest', aspect='auto', cmap = "YlGn_r")
					plt.axvline(10, c = "grey", linewidth = 1, linestyle = "dashed")
					plt.axvline(50, c = "grey", linewidth = 1, linestyle = "dashed")
					im.set_clim(0, 1)
					cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal", pad = 0.2)
					cbar.ax.set_xlabel("Activities")

					task_prob =[ "{0:.2f}".format(x) for x in data["task"]["action number {}".format(i)][i, 0, :, 0]]

					# ax.set_xticks(np.arange(0, 2001, 200))
					ax.set_yticks( np.linspace(50, 50 + (delta[i]-1) * 100, delta[i]))
					
					ax.set_yticklabels(task_prob, rotation = 90)

					
					# plt.legend(loc="upper left")
					plt.xlabel("Trial")
					plt.ylabel("Action")
					plt.title("Striatal activities over {} trials".format(max_trial))
					plt.xscale('log')
					#plt.tight_layout()

					plt.savefig("fig/experiment{}_{}_task{}_quantilemap.pdf".format(num, a, i))
					plt.close()



		for i, a in enumerate(agents):

			for j in range(3):
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)


				action_data = data["action"][a][2, j, :]
			
				plt.scatter((t+1), action_data + 0.05 * np.random.normal(0, 1, 500), marker = 'x') 
				plt.xlabel("Trials")
				plt.ylabel("Action")
				ax.set_yticks([0, 1])
				sns.despine()
				plt.tight_layout()
				plt.savefig("fig/experiment{}_sample_action_{}_{}.pdf".format(num,a, j), transparent = True)
				plt.close()

				ratio = 0.8
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)

				if a == "Distributional RPE Model":
					quantile_data = data["quantile_data"][a][0][j, 0,  :]
					class_num, quantile_num, max_trial = quantile_data.shape
					quantile_data = quantile_data.reshape( class_num * quantile_num, max_trial)
					#quantile_data[quantile_data > 1] = 1
					plt.xscale('log')

					im = ax.imshow(quantile_data, interpolation='nearest', aspect='auto', cmap = "YlGn_r")
					plt.axvline(5, c = "grey", linewidth = 1, linestyle = "dashed")
					plt.axvline(10, c = "grey", linewidth = 1, linestyle = "dashed")
					im.set_clim(0, 1)
					cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal", pad = 0.2)
					cbar.ax.set_xlabel("Activities")


					# ax.set_xticks(np.arange(0, 2001, 200))
					ax.set_yticks([50, 150])
					ax.set_yticklabels(["Left", "Right"], rotation = 90)
					
					# plt.legend(loc="upper left")
					plt.xlabel("Trial")
					plt.ylabel("Action")
					plt.title("Striatal activities over {} trials".format(max_trial))
					#plt.xscale('log')
					#plt.tight_layout()

					plt.savefig("fig/experiment{}_{}_sample_quantilemap_{}.pdf".format(num, a, j))
					plt.close()
	
		

	
	
	if num == 5:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		agents = [agents[0], agents[2],agents[1]]
		max_trial = len(data["action"]["Discounted Thompson Sampling"][0])
		max_trial = 1000
		max_episode = data["action"]["Discounted Thompson Sampling"].shape[0]
		switch = len(data["switch"]["Discounted Thompson Sampling"][0])
		switch = 4
		t = np.arange(max_trial)
		block_size = 200

		print("The following models are tested in a probability reversal task: {}".format(agents))
		

		
		
		bt = np.arange(0, 1000, 200)


		smooth_switch_data = {}
		for a in agents:
			smooth_switch_data[a] = np.zeros((max_episode, switch))
			for i in range(max_episode):
				for j in range(switch):
					if j == 0:
						smooth_switch_data[a][i][j] = data["switch"][a][i][j]
					smooth_switch_data[a][i][j] = (data["switch"][a][i][j] + data["switch"][a][i][j-1]) / 2.0

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		for i, a in enumerate(agents):
				
			choice_prob = np.mean(data["action"][a], axis = 0)
			ci =  np.std(data["action"][a], axis = 0) / np.sqrt(data["action"][a].shape[0])
			
			plt.plot(t + 1, choice_prob[:max_trial], label = a, c = palette[i])
			plt.fill_between(t + 1, choice_prob[:max_trial] + ci[:max_trial], choice_prob[:max_trial] - ci[:max_trial], color = palette[i], alpha = 0.1)
		
		for v in bt:
			plt.axvline(v, c = "grey", linewidth = 1, linestyle = "dashed")
		#ax.legend(loc = "upper left")
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		plt.xlim(left = 0)
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		plt.close()

		for i, a in enumerate(agents):

			for j in range(3):
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)


				action_data = data["action"][a][j, :]
			
				plt.scatter((t+1), action_data + 0.05 * np.random.normal(0, 1, 1000), marker = 'x') 
				for v in bt:
					plt.axvline(v, c = "grey", linewidth = 1, linestyle = "dashed")
				plt.xlabel("Trials")
				plt.ylabel("Action")
				ax.set_yticks([0, 1])
				sns.despine()
				plt.tight_layout()
				plt.savefig("fig/experiment{}_sample_action_{}_{}.pdf".format(num,a, j), transparent = True)
				plt.close()


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		
		block_t = np.arange(block_size)
		for i, a in enumerate(agents):
		

			action_block_data = np.zeros((max_episode * ((max_trial // block_size)-1), block_size))
			for j in range(max_trial // block_size):
				if j == 0:
					continue
				if j %2 == 0:
					action_block_data[(j-1) * max_episode:(j) * max_episode ] =   np.concatenate([data["action"][a][:, j * block_size- block_size // 2 :j * block_size], 1-data["action"][a][:, j * block_size :j * block_size + block_size//2]], axis = 1)
				
				else:
					action_block_data[(j-1) * max_episode:(j) * max_episode ] =  np.concatenate([1-data["action"][a][:, j * block_size- block_size // 2 :j * block_size], data["action"][a][:, j * block_size :j * block_size + block_size//2]], axis = 1)
				

			choice_prob = np.mean(action_block_data, axis = 0)
			ci =  np.std(action_block_data, axis = 0) / np.sqrt(action_block_data.shape[0])
			
			plt.plot(block_t - block_size//2, choice_prob, label = a, c = palette[i])
			plt.fill_between(block_t - block_size//2, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
		
		
		ax.legend(loc = "lower right", frameon = False)
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action_block.pdf".format(num), transparent = True)
		plt.close()


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		accuracy_data = []
		label_data = []
		for i, a in enumerate(agents):
			accuracy = np.zeros((max_episode, max_trial))
			for j in range(max_trial // block_size):
				if j % 2 == 0:
					accuracy[:, j * block_size: (j+1) * block_size] = 1-data["action"][a][:, j * block_size: (j+1) * block_size]
				else:
					accuracy[:, j * block_size: (j+1) * block_size] = data["action"][a][:, j * block_size: (j+1) * block_size]


			accuracy = np.mean(accuracy, axis = 1)

			accuracy_data.append(accuracy)
			label_data.append(a)
		ax.boxplot(accuracy_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks
		res = scipy.stats.shapiro(accuracy_data[0])
		print("The p value of shapiro test on the accuracy per run of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
		res = scipy.stats.shapiro(accuracy_data[1])
		print("The p value of shapiro test on the accuracy per run of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
		res = scipy.stats.shapiro(accuracy_data[2])
		print("The p value of shapiro test on the accuracy per run of {} in probability reversal task is {}".format(agents[2], res.pvalue))
	
		z, p = scipy.stats.kruskal(accuracy_data[0], accuracy_data[1], accuracy_data[2])
		print("The p value of Kruskal Wallis test on the accuracy per run in a probability reversal task is {}".format(p))
		
		p = sp.posthoc_dunn(accuracy_data, p_adjust='bonferroni')
		print("The p value of posthoc dunn test with bonferroni correction on the accuracy per run is detailed in the following matrix:")
		print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
		print(p)

			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accuracy")
		plt.title("Accuracy over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_accuracy_box.pdf".format(num), transparent = True)
		plt.close()


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		for i, a in enumerate(agents):
		

			regret = np.mean(data["regret"][a], axis = 0)
			ci =  np.std(data["regret"][a], axis = 0) / np.sqrt(data["regret"][a].shape[0])
			
			plt.plot(t, regret[:max_trial], label = a, c = palette[i])
			plt.fill_between(t, regret[:max_trial] + ci[:max_trial], regret[:max_trial] - ci[:max_trial], color = palette[i], alpha = 0.1)
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		regret_data = []
		label_data = []
		for i, a in enumerate(agents):
			regret_data.append(data["regret"][a][:, -1])
			label_data.append(a)
		ax.boxplot(regret_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks
		res = scipy.stats.shapiro(regret_data[0])
		print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
		res = scipy.stats.shapiro(regret_data[1])
		print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
		res = scipy.stats.shapiro(regret_data[2])
		print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[2], res.pvalue))
	
		z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
		print("The p value of Kruskal Wallis test on the accumulated regret in a probability reversal task is {}".format(p))
		
		p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
		print("The p value of posthoc dunn test with bonferroni correction on the accumulated regret is detailed in the following matrix:")
		print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
		print(p)

			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret_box.pdf".format(num), transparent = True)
		plt.close()



		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		t = np.arange(len(smooth_switch_data[a][0]))+1
		
		for i, a in enumerate(agents):
			error_bar =  np.std(smooth_switch_data[a], axis = 0) / np.sqrt(smooth_switch_data[a].shape[0])
			
			plt.errorbar(t,np.mean(smooth_switch_data[a], axis = 0), yerr = error_bar, c = palette[i], label = a)
			plt.plot(t, np.mean(smooth_switch_data[a], axis = 0), label = a, c = palette[i])
			x = np.vstack([np.arange(4) for _ in range(max_episode)])
			
			res = scipy.stats.shapiro(smooth_switch_data[a].flatten())
			print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[i], res.pvalue))
	
			res = scipy.stats.permutation_test((x.flatten(), smooth_switch_data[a].flatten()), lambda x, y: scipy.stats.spearmanr(x, y).statistic, n_resamples=100000)
			print("The spearman r coefficient of {}'s switch time is {} with p value {} in 10^6 resamples permutation test".format(agents[i], res.statistic, res.pvalue))
			
		
		#plt.legend(loc="upper left")
		plt.xlabel("Block")
		plt.ylabel("Time to switch")
		plt.xticks(t)
		plt.title("Averaged time to switch over blocks")
		sns.despine()
		plt.savefig("fig/experiment{}_switch_time.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		switch_data = []
		label_data = []
		for i, a in enumerate(agents):
			switch_data.append(smooth_switch_data[a].flatten())
			label_data.append(a)
		ax.boxplot(switch_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks
		res = scipy.stats.shapiro(switch_data[0])
		print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
		res = scipy.stats.shapiro(switch_data[1])
		print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
		res = scipy.stats.shapiro(switch_data[2])
		print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[2], res.pvalue))
	
		z, p = scipy.stats.kruskal(switch_data[0], switch_data[1], switch_data[2])
		print("The p value of Kruskal Wallis test on the switch time in a probability reversal task is {}".format(p))
		
	
		p = sp.posthoc_dunn(switch_data, p_adjust='bonferroni')
		print("The p value of posthoc dunn test with bonferroni correction on the switch time is detailed in the following matrix:")
		print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
		print(p)

			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_switch_box.pdf".format(num), transparent = True)
		plt.close()


		t = np.arange(max_trial)
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		for i in range(2):
			
			task = data["task"]
			line_style = ["solid", "dashed"]
			action = ["Left", "Right"]
			
			plt.plot(t, task[0][i][:max_trial], label = action[i], c = palette[-4], linestyle = line_style[i] )
		plt.legend(loc="upper left")
		plt.ylim(0, 1)
		plt.xlabel("Trial")
		plt.ylabel("Probability of reward")
		plt.title("Task reward structure")
		sns.despine()
		plt.savefig("fig/experiment{}_task.pdf".format(num), transparent = True)
		plt.close()


		
		fig, ax = plt.subplots(2, sharex = True, gridspec_kw={'height_ratios': [1, 3]})
		#plt.subplots_adjust()

		fig.set_figwidth(4.8 )
		fig.set_figheight(4.8)
		plt.tight_layout(pad = -3.5)
		for i, a in enumerate(agents):
			

			if a == "Distributional RPE Model":
				
				quantile_data = data["quantile_data"][a]
				
				quantile_data = quantile_data[0, 0].reshape( 2 * 100, 1000)
				quantile_data[quantile_data > 1] = 1

				im = ax[0].imshow(quantile_data, cmap = "YlGn_r")
				# cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal")
				# cbar.ax.set_xlabel("Activities")

				ax[0].set_xticks(np.arange(0, max_trial+1, 200))
				ax[0].set_yticks([50, 150])
				ax[0].set_yticklabels(["L", "R"], rotation = 90)
				for v in bt:
					ax[0].axvline(v, c = "grey", linewidth = 1, linestyle= "dashed")
				# plt.legend(loc="upper left")
				
				ax[0].set_ylabel("[Action]")
				ax[0].set_title("Striatal activities in context oblivious model over {} trials".format(max_trial))
				

			if a == "Known-Context Distributional RPE Model":
				
				quantile_data = data["quantile_data"][a]
				
				quantile_data = quantile_data[0, :, 0, :, :, :].reshape(2 * 2 * 100, 1000)
				quantile_data[quantile_data > 1] = 1

				im = ax[1].imshow(quantile_data, cmap = "YlGn_r")

				axins = inset_axes(ax[1],
                    width="100%",  
                    height="15%",
                    loc='lower center',
                    borderpad=-5.5
                   )
				cbar = ax[1].figure.colorbar(im, ax = ax[1], cax=axins, orientation="horizontal")
				cbar.ax.set_xlabel("Activities")

				ax[1].set_xticks(np.arange(0, max_trial+1, 200))
				ax[1].set_yticks([50, 150, 250, 350])
				ax[1].set_yticklabels(["1/L", "1/R", "2/L", "2/R"], rotation = 90)
				for v in bt:
					ax[1].axvline(v, c = "grey", linewidth = 1, linestyle = "dashed")
				# plt.legend(loc="upper left")
				ax[1].set_xlabel("Trial")
				ax[1].set_ylabel("[Context/Action]")
				ax[1].set_title("Striatal activities in known-context model over {} trials".format(max_trial))
		plt.tight_layout()
		plt.savefig("fig/experiment{}_quantilemap.pdf".format(num), transparent = True)
		plt.close()



		fig, ax = plt.subplots(1, 3)
		fig.set_figwidth(4.8 * 3)
		fig.set_figheight(4.8)
		for i, a in enumerate(agents):
			value = data["value"][a]

			if i == 0:
				width = [1, 3]
				width_patch = [mlines.Line2D([], [], color = "grey",linewidth=width[i], label="Context {}".format(i+1)) for i in range(2)]
				legend_width = fig.legend(handles = width_patch, frameon = False, bbox_to_anchor=(0.1, 1), loc="upper left")
		
			
			
			for s in range(1):
				for j in range(2):
					
			
					
					if a == "Known-Context Distributional RPE Model":
						for k in range(2):
							width = [1, 3]
							if j == 0:
								value_data = np.mean(value, axis = 0)[k][s][j]
								ci =  np.std(value[:, k, s, j], axis = 0) / np.sqrt(value[:, k, s, j].shape[0])
								ax[i].plot(t+1, np.mean(value, axis = 0)[k][s][j][:max_trial],  label = "Known-Context Model Context {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k])
								ax[i].fill_between(t + 1, value_data[:max_trial] + ci[:max_trial], value_data[:max_trial] - ci[:max_trial], color = palette[i], alpha = 0.1)
							else:
								value_data = np.mean(value, axis = 0)[k][s][j]
								ci =  np.std(value[:, k, s, j], axis = 0) / np.sqrt(value[:, k, s, j].shape[0])
								ax[i].plot(t+1,  np.mean(value, axis = 0)[k][s][j][:max_trial], label = "Known-Context Model Context {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k], linestyle = "dashed")
								ax[i].fill_between(t + 1, value_data[:max_trial] + ci[:max_trial], value_data[:max_trial] - ci[:max_trial], color = palette[i], alpha = 0.1)
					else:
						if j == 0:
							value_data = np.mean(value, axis = 0)[s][j]
						
							ci =  np.std(value[:, s, j], axis = 0) / np.sqrt(value[:, s, j].shape[0])
							ax[i].plot(t+1, np.mean(value, axis = 0)[s][j][:max_trial],  label = "{} Action {}".format(a, j+1), c = palette[i])
							ax[i].fill_between(t + 1, value_data[:max_trial] + ci[:max_trial], value_data[:max_trial] - ci[:max_trial], color = palette[i], alpha = 0.1)
						else:
							value_data = np.mean(value, axis = 0)[s][j]
							ci =  np.std(value[:, s, j], axis = 0) / np.sqrt(value[:, s, j].shape[0])
							ax[i].plot(t+1,  np.mean(value, axis = 0)[s][j][:max_trial], label = "{} Action {}".format(a, j+1), c = palette[i], linestyle = "dashed")
							ax[i].fill_between(t + 1, value_data[:max_trial] + ci[:max_trial], value_data[:max_trial] - ci[:max_trial], color = palette[i], alpha = 0.1)
				for v in bt:
					ax[i].axvline(v, c = "grey", linewidth = 1, linestyle = "dashed")



		width_patch = [mlines.Line2D([], [], color = "grey",linewidth=width[i], label="Context {}".format(i+1)) for i in range(2)]
		line_patch = [mlines.Line2D([], [],color="grey", label=i, linestyle = t) for i, t in [("Left", "solid"), ("Right", "dashed")]]

		
		#legend_line = plt.legend(handles = line_patch,  title = "Action", frameon = False, bbox_to_anchor=(0.95, 0.5), loc="upper left")
		fig.legend(handles = line_patch, bbox_to_anchor=(0.05, 1), loc = "upper left", frameon = False)
		#ax[0].add_artist(legend_width)
		fig.supxlabel("Trial", y = 0.04, fontsize = "medium")
		fig.supylabel("Estimated value", x = 0.01, fontsize = "medium")
		plt.suptitle("Estimated value over {} trials".format(max_trial), fontsize = "medium")
		sns.despine()
		plt.tight_layout()
		fig.savefig("fig/experiment{}_value.pdf".format(num), transparent = True)
		plt.close()

		# for i, a in enumerate(agents):
			
			
		# 	for s in range(1):
		# 		for j in range(2):
					
		# 			value = np.mean(data["value"][a], axis = 0)
		# 			value[value > 1] = 1

		# 			if a == "Known-Context Distributional RPE Model":
		# 				for k in range(2):
							
		# 					if k == 0:
		# 						c = "orange"
		# 						c1 = "moccasin"
		# 					else:
		# 						c = "red"
		# 						c1 = "lightcoral"

		# 					if j == 0:
		# 						plt.plot(t, value[k][s][j], label = "Known-Context Model Context {}".format(k+1), c = c)
		# 					else:
		# 						pass
		# 						#plt.plot(t, value[k][s][j], label = "Known-Context Model Context {} Action {}".format(k+1, j+1), c = c1)


		# 			else:

		# 				if j == 0:
		# 					plt.plot(t, value[s][j],  label = a, c = c)
		# 				else:
		# 					pass
		# 					#plt.plot(t, value[s][j], label = "{} Action {}".format(a, j+1), c = c1)

		# for v in bt:
		# 	plt.axvline(v, c = "grey", linewidth = 1)
		# plt.legend(loc="upper right")
		# plt.xlabel("Trial")
		# plt.ylabel("Estimated value")
		# plt.title("Estimated value of Action 1 over {} trials".format(max_trial))
		# plt.savefig("fig/experiment{}_value.pdf".format(num), transparent = True)
		# plt.close()



	
	if num == 11 or num == 15:
		data11 = load_dict("experiment{}_data".format(11))
		data15 = load_dict("experiment{}_data".format(15))

		agents = list(data11["action"].keys())

		agents = [agents[2], agents[0], agents[1]]
		max_trial = data11["action"][agents[0]].shape[1]
		max_episode = data11["action"][agents[0]].shape[0]

		# t = np.arange(max_trial)
		# ratio = 0.8
		# fig, ax = plt.subplots()
		# fig.set_figwidth(4.8 * ratio)
		# fig.set_figheight(4.8 * ratio)

		regret_data11 = {}
		regret_data15 = {}

		for i, a in enumerate(agents):
			regret_data11[a] = np.zeros((max_episode, max_trial))
			regret_data11[a][:, 0] = data11["regret"][a][:, 0]
			regret_data15[a] = np.zeros((max_episode, max_trial))
			regret_data15[a][:, 0] = data15["regret"][a][:, 0]
			for j in range(max_trial-1):
				regret_data11[a][:, j+1] = regret_data11[a][:, j] + data11["regret"][a][:, j+1]
				regret_data15[a][:, j+1] = regret_data15[a][:, j] + data15["regret"][a][:, j+1]
		

		# 	regret = np.mean(regret_data, axis = 0)
		# 	ci =  np.std(regret_data, axis = 0) / np.sqrt(regret_data.shape[0])
			
		# 	plt.plot(t, regret, label = a, c = palette[i])
		# 	plt.fill_between(t, regret + ci, regret - ci, color = palette[i], alpha = 0.1)
		# plt.legend(loc="upper left", frameon=False)
		# plt.xlabel("Trial")
		# plt.ylabel("Accumulated regret")
		# plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		# sns.despine()
		# plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		# plt.close()

		# ratio = 0.8
		# fig, ax = plt.subplots()
		# fig.set_figwidth(4.8 * ratio)
		# fig.set_figheight(4.8 * ratio)
		
		# accuracy = {}
		# accuracy_data = []
		# for i, a in enumerate(agents):
			
		# 	accuracy[a] = np.copy(data["action"][a])
		# 	accuracy[a][accuracy[a] < 1] = 0
		# 	accuracy[a][accuracy[a] == 1] = 1
		# 	accuracy[a] = 1- accuracy[a]

		# 	choice_prob = np.mean(accuracy[a], axis = 0)
		# 	ci =  np.std(accuracy[a], axis = 0) / np.sqrt(accuracy[a].shape[0])
			
		# 	plt.plot(t + 1, choice_prob, label = a, c = palette[i])
		# 	plt.fill_between(t + 1, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
	
		# plt.xlabel("Trial")
		# plt.ylabel("Accurate choice probability")
		# plt.xlim(left = 0)
		# plt.title("Choice probability over {} trials".format(max_trial))
		# sns.despine()
		# plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		# plt.close()


		label = ["1/3", "2/3"]

		x = np.arange(len(label))
		ratio = 0.8
		
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.25
		multiplier = 0

		for i, a in enumerate(agents):

			regret_err = [np.std(regret_data11[a][:, -1], axis = 0) / np.sqrt(regret_data11[a][:, -1].shape[0]), np.std(regret_data15[a][:, -1], axis = 0) / np.sqrt(regret_data15[a][:, -1].shape[0])]
			regret_data = [np.mean(regret_data11[a][:, -1], axis = 0), np.mean(regret_data15[a][:, -1], axis = 0)]

			
			offset  = width * multiplier
			# rects = ax.bar(x + offset, np.mean(data["regret"][a][idx, :, -1], axis = 1), width, label = a, yerr = error_bar)
			if a == "Distributional RPE Model":
				
				rects = ax.bar(x + offset, regret_data, width, label = "CogLink", yerr = regret_err)
			else:
				rects = ax.bar(x + offset, regret_data, width, label = a, yerr = regret_err)

			# for j in range(len(delta)):
			#  	ax.boxplot(data["regret"][a][j, :, -1], positions = [x[j]+offset])
			
			multiplier += 1
			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		
	
			
		res = scipy.stats.shapiro(regret_data11[agents[0]][:, -1])
		print("The p value of shapiro test on the accumulated regret of {} in environment a = {} is {}".format(agents[0],label[0],  res.pvalue))
		res = scipy.stats.shapiro(regret_data11[agents[1]][:, -1])
		print("The p value of shapiro test on the accumulated regret of {} in environment a = {} is {}".format(agents[1],label[0],  res.pvalue))
		res = scipy.stats.shapiro(regret_data11[agents[2]][:, -1])
		print("The p value of shapiro test on the accumulated regret of {} in environment a = {} is {}".format(agents[2],label[0],  res.pvalue))

		regret_data = [regret_data11[a][:, -1]  for a in agents]

		z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
		print("The p value of Kruskal Wallis test on the accumulated regret per run in environment a = {} is {}".format(label[0], p))
	
		p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
		print("The p value of posthoc dunn test with bonferroni correction on the accumulated regret is detailed in the following matrix:")
		print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
		print(p)

		res = scipy.stats.shapiro(regret_data15[agents[0]][:, -1])
		print("The p value of shapiro test on the accumulated regret of {} in environment a = {} is {}".format(agents[0],label[1],  res.pvalue))
		res = scipy.stats.shapiro(regret_data15[agents[1]][:, -1])
		print("The p value of shapiro test on the accumulated regret of {} in environment a = {} is {}".format(agents[1],label[1],  res.pvalue))
		res = scipy.stats.shapiro(regret_data15[agents[2]][:, -1])
		print("The p value of shapiro test on the accumulated regret of {} in environment a = {} is {}".format(agents[2],label[1],  res.pvalue))

		regret_data = [regret_data15[a][:, -1]  for a in agents]

		z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
		print("The p value of Kruskal Wallis test on the accumulated regret per run in environment a = {} is {}".format(label[1], p))
	
		p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
		print("The p value of posthoc dunn test with bonferroni correction on the accumulated regret is detailed in the following matrix:")
		print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
		print(p)


		# plt.legend(loc="upper right", frameon = False)
		plt.legend(loc="upper left", frameon = False)
		ax.set_xticks(x + width, labels = label)
		plt.xlabel("a")
		plt.ylabel("Regret")
		plt.ylim(bottom = 0)
		
		# plt.title("Averaged regret over different \u0394 between two alternatives".format(max_trial))
		sns.despine()
		plt.tight_layout()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

	if num == 17:
		data17 = load_dict("experiment{}_data".format(num))
		data14 = load_dict("experiment{}_data".format(14))
		agents = list(data17["action"].keys())

		
		switch = len(data17["switch"][agents[0]][0])
		max_trial = len(data17["action"][agents[0]][0])
		max_episode = len(data17["action"][agents[0]])
		
		block_size = 100

		

		switch_data17 = {}
		switch_bool = {}
		
		for a in agents:
			switch_data17[a] = np.zeros((max_episode, switch))
			switch_bool[a] = [[False] * switch  for _ in range(max_episode)] 
			for i in range(max_episode):
				for j, action in enumerate(data17["action"][a][i]): 
					
					block_size = 200
					idx = int(j / block_size) - 1

					if idx > 1:
						if not switch_bool[a][i][idx-2]:
							
							switch_data17[a][i, idx-2] = block_size
							switch_bool[a][i][idx - 2] = True
						
					if idx > -1 and not switch_bool[a][i][idx]:
						new_a = (idx+1) % 2


						if new_a == 1:
							if data17["choice_prob"][a][i][j] >= 0.8:
								switch_data17[a][i][idx] += j+1  - (idx + 1) * block_size
								switch_bool[a][i][idx] = True
								#print(self.switch_time[agent.name][idx])

						else:
							if data17["choice_prob"][a][i][j] <= 0.2:
								switch_data17[a][i][idx]  += j+1  - (idx + 1) * block_size
								switch_bool[a][i][idx] = True

		data14 = load_dict("experiment{}_data".format(14))

		
		switch = len(data14["switch"][agents[0]][0])
		max_trial = len(data14["action"][agents[0]][0])
		max_episode = len(data14["action"][agents[0]])
		
		block_size = 100

		agents = list(data14["action"].keys())

		switch_data14 = {}
		switch_bool = {}
		
		for a in agents:
			switch_data14[a] = np.zeros((max_episode, switch))
			switch_bool[a] = [[False] * switch  for _ in range(max_episode)] 
			for i in range(max_episode):
				for j, action in enumerate(data14["action"][a][i]): 
					
					block_size = 200
					idx = int(j / block_size) - 1

					if idx > 1:
						if not switch_bool[a][i][idx-2]:
							
							switch_data14[a][i, idx-2] = block_size
							switch_bool[a][i][idx - 2] = True
						
					if idx > -1 and not switch_bool[a][i][idx]:
						new_a = (idx+1) % 2


						if new_a == 1:
							if data14["choice_prob"][a][i][j] >= 0.8:
								switch_data14[a][i][idx] += j+1  - (idx + 1) * block_size
								switch_bool[a][i][idx] = True
								#print(self.switch_time[agent.name][idx])

						else:
							if data14["choice_prob"][a][i][j] <= 0.2:
								switch_data14[a][i][idx]  += j+1  - (idx + 1) * block_size
								switch_bool[a][i][idx] = True

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		a = "Thalamocortical Model"
		switch_result = [switch_data14[a].flatten(), switch_data17[a].flatten()]
		label_data = ["High uncertainty", "Low uncertainty"]
			
		ax.boxplot(switch_result, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks
		res = scipy.stats.shapiro(switch_result[0])
		print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(label_data[0], res.pvalue))
	
		res = scipy.stats.shapiro(switch_result[1])
		print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(label_data[1], res.pvalue))
	
		z, p = scipy.stats.mannwhitneyu(switch_result[0], switch_result[1])
		print("The p value of two-way rank sum test on the switch time in a probability reversal task is {}".format(p))
		
		print("switch for {} = {}, sem = {}".format(label_data[0], np.mean(switch_result[0]), np.std(switch_result[0]) / np.sqrt(len(switch_result[0]))))
		print("switch for {} = {}, sem = {}".format(label_data[1],  np.mean(switch_result[1]), np.std(switch_result[1]) / np.sqrt(len(switch_result[1]))))


			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Switching time")
		plt.title("Switching time")
		sns.despine()
		plt.savefig("fig/experiment{}_switch_box.pdf".format(num), transparent = True)
		plt.close()





	if num == 6 or num == 10 or num == 14:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())

		
		print(data["switch"]["HMM"])
		print(data["switch"][agents[0]])

	
		
		
		
		switch = len(data["switch"][agents[0]][0])
		max_trial = len(data["action"][agents[0]][0])
		max_episode = len(data["action"][agents[0]])
		
		block_size = 100

		agents = list(data["action"].keys())
		if num == 14:
			agents = [agents[0], agents[2]]

		
		mask = [ np.logical_and((data["switch"][a][:, -1] < 200) , (data["switch"][a][:, -2] < 200)) for a in agents]
		

		# agents = [agents[1], agents[0]]

		print("The following models are tested in a probability reversal task: {}".format(agents))
		

		switch_data = {}
		switch_bool = {}
		
		for a in agents:
			switch_data[a] = np.zeros((max_episode, switch))
			switch_bool[a] = [[False] * switch  for _ in range(max_episode)] 
			for i in range(max_episode):
				for j, action in enumerate(data["action"][a][i]): 
					
					block_size = 200
					idx = int(j / block_size) - 1

					if idx > 1:
						if not switch_bool[a][i][idx-2]:
							
							switch_data[a][i, idx-2] = block_size
							switch_bool[a][i][idx - 2] = True
						
					if idx > -1 and not switch_bool[a][i][idx]:
						new_a = (idx+1) % 2


						if new_a == 1:
							if data["choice_prob"][a][i][j] >= 0.8:
								switch_data[a][i][idx] += j+1  - (idx + 1) * block_size
								switch_bool[a][i][idx] = True
								#print(self.switch_time[agent.name][idx])

						else:
							if data["choice_prob"][a][i][j] <= 0.2:
								switch_data[a][i][idx]  += j+1  - (idx + 1) * block_size
								switch_bool[a][i][idx] = True
		smooth_switch_data = {}
		for a in agents:
			smooth_switch_data[a] = np.zeros((max_episode, switch))
			for i in range(max_episode):
				for j in range(switch):
					if j == 0:
						smooth_switch_data[a][i][j] = switch_data[a][i][j]
					smooth_switch_data[a][i][j] = (switch_data[a][i][j] + switch_data[a][i][j-1]) / 2.0

	


		t = np.arange(max_trial)
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		for i, a in enumerate(agents):
			

			regret = np.mean(data["regret"][a][mask[i], :], axis = 0)
			ci =  np.std(data["regret"][a][mask[i], :], axis = 0) / np.sqrt(data["regret"][a][mask[i], :].shape[0])
			
			plt.plot(t, regret, label = a, c = palette[i])
			plt.fill_between(t, regret + ci, regret - ci, color = palette[i], alpha = 0.1)
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		regret_data = []
		label_data = []
		for i, a in enumerate(agents):
			regret_data.append(data["regret"][a][:, -1])
			label_data.append(a)
		ax.boxplot(regret_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks
		res = scipy.stats.shapiro(regret_data[0])
		print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
		res = scipy.stats.shapiro(regret_data[1])
		print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
		z, p = scipy.stats.mannwhitneyu(regret_data[0], regret_data[1])
		print("The p value of two-way rank sum test on the accumulated regret in a probability reversal task is {}".format(p))
		print("regret for {} = {}, sem = {}".format(agents[0], np.mean(regret_data[0]), np.std(regret_data[0]) / np.sqrt(len(regret_data[0]))))
		print("regret for {} = {}, sem = {}".format(agents[1],  np.mean(regret_data[1]), np.std(regret_data[1]) / np.sqrt(len(regret_data[1]))))

			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret_box.pdf".format(num), transparent = True)
		plt.close()









		
	

		t = np.arange(max_trial)
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		for i in range(2):
			
			task = data["task"]
			line_style = ["solid", "dashed"]
			action = ["Left", "Right"]
			
			plt.plot(t, task[0][i], label = action[i], c = palette[-4], linestyle = line_style[i] )
		plt.legend(loc="upper left")
		plt.ylim(0, 1)
		plt.xlabel("Trial")
		plt.ylabel("Probability of reward")
		plt.title("Task reward structure")
		sns.despine()
		plt.savefig("fig/experiment{}_task.pdf".format(num), transparent = True)
		plt.close()


		
			
		
		bt = np.arange(0, 1001, 200)
	
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		

		for i, a in enumerate(agents):
				
			choice_prob = np.mean(data["action"][a][mask[i], :], axis = 0)
			ci =  np.std(data["action"][a][mask[i], :], axis = 0) / np.sqrt(data["action"][a][mask[i], :].shape[0])
			
			plt.plot(t + 1, choice_prob, label = a, c = palette[i])
			plt.fill_between(t + 1, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
		
		for v in bt:
			plt.axvline(v, c = "grey", linewidth = 1, linestyle = "dashed")
		#ax.legend(loc = "upper left")
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		plt.xlim(left = 0)
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		plt.close()

		for i, a in enumerate(agents):
			if a == "Thalamocortical Model":
			

				md0 = np.vstack([ x["MD"][0] for x in data["histogram"][a]]).reshape(max_episode, max_trial)
				md1 = np.vstack([ x["MD"][1] for x in data["histogram"][a]]).reshape(max_episode, max_trial)

				md0_data = np.mean(md0, axis = 0)
				md1_data = np.mean(md1, axis = 0)
				ci0 =  np.std(md0, axis = 0) / np.sqrt(md0.shape[0])
				ci1 =  np.std(md1, axis = 0) / np.sqrt(md1.shape[0])

				ratio = 0.8
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)
				plt.plot(t+1, md0_data,  c = palette[0])
				plt.fill_between(t + 1, md0_data + ci0, md0_data - ci0, color = palette[0], alpha = 0.1)
				plt.plot(t+1, md1_data,  c = palette[1])
				plt.fill_between(t + 1, md1_data + ci1, md1_data - ci1, color = palette[1], alpha = 0.1)
				for v in bt:
					plt.axvline(v, c = "grey", linestyle = "dashed")
				plt.legend(loc="upper left")
				plt.ylim(bottom = 3.2)
				plt.xlabel("Trial")
				plt.ylabel("Normalized firing rate")
				plt.title("MD activities in {} over {} trials".format(a, max_trial))
				sns.despine()
				plt.savefig("fig/experiment{}_md_activities_{}.pdf".format(num,a))
				plt.close()

				if num == 14:
					ratio = 0.8
					fig, ax = plt.subplots()
					fig.set_figwidth(4.8 * ratio)
					fig.set_figheight(4.8 * ratio)

					time = np.linspace(-20, 20, 40)
					
					scatter = plt.scatter(md0[1, 580:620], md1[1, 580:620], c=time, cmap='viridis',  vmin = -20, vmax = 20)

					colorbar = plt.colorbar(scatter)
					colorbar.set_label('Trials after switch')
					plt.xlabel('MD (context 1)')
					plt.ylabel('MD (context 2)')

					sns.despine()
					plt.savefig("fig/experiment{}_low_dimensional_manifold.pdf".format(num), transparent = True)
					plt.close()

			idx = np.random.randint(0, max_episode, 3)
			for j in range(3):
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)


				action_data = data["action"][a][idx[j], :]

			
				plt.scatter((t+1), action_data + 0.05 * np.random.normal(0, 1, max_trial), marker = 'x') 
				for v in bt:
					plt.axvline(v, c = "grey", linewidth = 1, linestyle = "dashed")
				plt.xlabel("Trials")
				plt.ylabel("Action")
				ax.set_yticks([0, 1])
				sns.despine()
				plt.tight_layout()
				plt.savefig("fig/experiment{}_sample_action_{}_{}.pdf".format(num,a, j), transparent = True)
				plt.close()

				

				if a == "Thalamocortical Model":
					ratio = 0.8
					fig, ax = plt.subplots()
					fig.set_figwidth(4.8 * ratio)
					fig.set_figheight(4.8 * ratio)
				
					quantile_data = data["quantile_data"][a][idx[j], :, 0, :].reshape(2 * 2 * 100, max_trial)

					im = ax.imshow(quantile_data, cmap = "YlGn_r")
					cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal")
					cbar.ax.set_xlabel("Activities")

					ax.set_xticks(np.arange(0, 1001, 200))
					ax.set_yticks([50, 150, 250, 350])
					ax.set_yticklabels(["1/L", "1/R", "2/L", "2/R"])
					for v in bt:
						plt.axvline(v, c = "grey", linewidth = 1)
					# plt.legend(loc="upper left")
					plt.xlabel("Trial")
					plt.ylabel("[Context/Action]")
					plt.title("Striatal activities in {} model over {} trials".format(a, max_trial))
					plt.tight_layout()
					plt.savefig("fig/experiment{}_{}_sample_quantilemap_{}.pdf".format(num, a, j))
					plt.close()

				if a == "Thalamocortical Model" or a == "HMM Model":
					if i == 0:
						c = "lawngreen"
					elif i == 1:
						c = "purple"
					elif i == 2:
						c = "green"
					else:
						c = "black"

					evidence = data["evidence"][a][idx[j], :]
					
					ratio = 0.8
					fig, ax = plt.subplots()
					fig.set_figwidth(4.8 * ratio)
					fig.set_figheight(4.8 * ratio)
					plt.plot(t, evidence,  c = c)
					for v in bt:
						plt.axvline(v, c = "grey", linestyle = "dashed")
					plt.legend(loc="upper left")
					plt.xlabel("Trial")
					plt.ylabel("Normalized firing rate")
					plt.title("Difference between two contextual populations\nfor {} over {} trials".format(a, max_trial))
					sns.despine()
					plt.savefig("fig/experiment{}_sample_evidence_{}_{}.pdf".format(num,a, j))
					plt.close()



		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		
		block_t = np.arange(block_size)
		for i, a in enumerate(agents):
		

			action_block_data = np.zeros((max_episode * ((max_trial // block_size)-1), block_size))
			for j in range(max_trial // block_size):
				if j == 0:
					continue
				if j %2 == 0:
					action_block_data[(j-1) * max_episode:(j) * max_episode ] =   np.concatenate([data["action"][a][:, j * block_size- block_size // 2 :j * block_size], 1-data["action"][a][:, j * block_size :j * block_size + block_size//2]], axis = 1)
				
				else:
					action_block_data[(j-1) * max_episode:(j) * max_episode ] =  np.concatenate([1-data["action"][a][:, j * block_size- block_size // 2 :j * block_size], data["action"][a][:, j * block_size :j * block_size + block_size//2]], axis = 1)
				

			choice_prob = np.mean(action_block_data, axis = 0)
			ci =  np.std(action_block_data, axis = 0) / np.sqrt(action_block_data.shape[0])
			
			plt.plot(block_t - block_size//2, choice_prob, label = a, c = palette[i])
			plt.fill_between(block_t - block_size//2, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)


		
		
		ax.legend(loc = "lower right", frameon = False)
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		plt.xlim(left = -block_size//2, right = block_size//2)
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action_block.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		accuracy_data = []
		label_data = []
		for i, a in enumerate(agents):
			accuracy = np.zeros((max_episode, max_trial))
			for j in range(max_trial // block_size):
				if j % 2 == 0:
					accuracy[:, j * block_size: (j+1) * block_size] = 1-data["action"][a][:, j * block_size: (j+1) * block_size]
				else:
					accuracy[:, j * block_size: (j+1) * block_size] = data["action"][a][:, j * block_size: (j+1) * block_size]


			accuracy = np.mean(accuracy, axis = 1)

			accuracy_data.append(accuracy)
			label_data.append(a)
		ax.boxplot(accuracy_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks
		res = scipy.stats.shapiro(accuracy_data[0])
		print("The p value of shapiro test on the accuracy per run of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
		res = scipy.stats.shapiro(accuracy_data[1])
		print("The p value of shapiro test on the accuracy per run of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
		z, p = scipy.stats.mannwhitneyu(accuracy_data[0], accuracy_data[1])
		print("The p value of two-way rank sum test on the accuracy per run in a probability reversal task is {}".format(p))
		
		print("accuracy for {} = {}, sem = {}".format(agents[0], np.mean(accuracy_data[0]), np.std(accuracy_data[0]) / np.sqrt(len(accuracy_data[0]))))
		print("accuracy for {} = {}, sem = {}".format(agents[1],  np.mean(accuracy_data[1]), np.std(accuracy_data[1]) / np.sqrt(len(accuracy_data[1]))))
		
			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accuracy")
		plt.title("Accuracy over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_accuracy_box.pdf".format(num), transparent = True)
		plt.close()






		st = np.arange(switch) + 1


		# fig, ax = plt.subplots(layout='constrained')
		# width = 0.25
		# multiplier = 0

		# x = np.arange(switch)
			
		# for i, a in enumerate(agents):
		# 	error_bar =  np.std(smooth_switch_data[a][mask, :] - 8, axis = 0) / np.sqrt(smooth_switch_data[a][mask, :].shape[0])
		# 	offset  = width * multiplier
		# 	rects = ax.bar(x + offset, np.mean(smooth_switch_data[a][mask, :] - 8, axis = 0), width, label = a, yerr = error_bar, color = palette[i])
		# 	multiplier += 1
		# 	# plt.errorbar(st,np.mean(smooth_switch_data[a][mask, :] - 8, axis = 0), yerr = error_bar, c = palette[i], label = a)
		# 	# plt.plot(st, np.mean(smooth_switch_data[a][mask, :] - 8, axis = 0), label = a, c = palette[i])
		
		# #plt.legend(loc="upper left")
		# plt.xlabel("Block")
		# plt.ylabel("Time to switch")
		# ax.set_xticks(x + width, st)
		# plt.ylim(bottom = 0)
		# plt.title("Averaged time to switch over blocks")
		# sns.despine()
		# plt.savefig("fig/experiment{}_switch_time.pdf".format(num), transparent = True)
		# plt.close()




		

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		ax.xaxis.get_major_locator().set_params(integer=True)
		ax.yaxis.get_major_locator().set_params(integer=True)
		
		
		for i, a in enumerate(agents):


			if num == 14:
				if a != "Thalamocortical Model":
					continue
			
			switch_result = np.mean(switch_data[a][mask[i], :], axis = 0) - 8

			error_bar =  np.std(switch_data[a][mask[i], :], axis = 0) / np.sqrt(switch_data[a][mask[i], :].shape[0])
			
			plt.errorbar(st, switch_result, yerr = error_bar, c = palette[i], label = a)
			plt.plot(st, switch_result, label = a, c = palette[i])
			x = np.vstack([np.arange(4) for _ in range(max_episode)])
			
			res = scipy.stats.shapiro(switch_data[a].flatten())
			print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[i], res.pvalue))
	
			res = scipy.stats.permutation_test((x.flatten(), switch_data[a].flatten()), lambda x, y: scipy.stats.spearmanr(x, y).statistic, n_resamples=100000)
			print("The spearman r coefficient of {}'s switch time is {} with p value {} in 10^6 resamples permutation test".format(agents[i], res.statistic, res.pvalue))
			

		
		
		# plt.legend(loc="upper right",  frameon = False)
		plt.xlabel("Block")
		plt.ylabel("Trials to switch")
		plt.title("Averaged trials to switch over blocks")
		sns.despine()
		plt.savefig("fig/experiment{}_switch_time.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		switch_result = []
		label_data = []
		for i, a in enumerate(agents):
			switch_result.append(smooth_switch_data[a].flatten())
			label_data.append(a)
		ax.boxplot(switch_result, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks
		res = scipy.stats.shapiro(switch_result[0])
		print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
		res = scipy.stats.shapiro(switch_result[1])
		print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
		z, p = scipy.stats.mannwhitneyu(switch_result[0], switch_result[1])
		print("The p value of two-way rank sum test on the switch time in a probability reversal task is {}".format(p))
		
		print("switch for {} = {}, sem = {}".format(agents[0], np.mean(switch_result[0]), np.std(switch_result[0]) / np.sqrt(len(switch_result[0]))))
		print("switch for {} = {}, sem = {}".format(agents[1],  np.mean(switch_result[1]), np.std(switch_result[1]) / np.sqrt(len(switch_result[1]))))


			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Switching time")
		plt.title("Switching time")
		sns.despine()
		plt.savefig("fig/experiment{}_switch_box.pdf".format(num), transparent = True)
		plt.close()





		for i, a in enumerate(agents):


			if a == "Thalamocortical Model":

				ratio = 0.8
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)

				switch_result = [switch_data[a][:, 0], switch_data[a][:, 1:].flatten()]
				label_data = ["High asso-uncertainty", "Low asso-uncertainty"]
				
				ax.boxplot(switch_result, sym = '', widths = 0.7, showcaps = False, 
		                     vert=True,  # vertical box alignment
		                     labels=label_data)  # will be used to label x-ticks
				res = scipy.stats.shapiro(switch_result[0])
				print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(label_data[0], res.pvalue))
			
				res = scipy.stats.shapiro(switch_result[1])
				print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(label_data[1], res.pvalue))
			
				z, p = scipy.stats.mannwhitneyu(switch_result[0], switch_result[1])
				print("The p value of two-way rank sum test on the switch time in a probability reversal task is {}".format(p))
				
				print("switch for {} = {}, sem = {}, n = {}".format(label_data[0], np.mean(switch_result[0]), np.std(switch_result[0]) / np.sqrt(len(switch_result[0])), len(switch_result[0])))
				print("switch for {} = {}, sem = {}, n = {}".format(label_data[1],  np.mean(switch_result[1]), np.std(switch_result[1]) / np.sqrt(len(switch_result[1])), len(switch_result[1])))


					
				plt.legend(loc="upper left", frameon=False)
				plt.xlabel("Trial")
				plt.ylabel("Switching time")
				plt.title("Switching time")
				sns.despine()
				plt.savefig("fig/experiment{}_switch_associative_uncertainty.pdf".format(num), transparent = True)
				plt.close()



				if i == 0:
					c = "lawngreen"
				elif i == 1:
					c = "purple"
				elif i == 2:
					c = "green"
				else:
					c = "black"

				evidence = np.mean(data["evidence"][a][mask[i], :], axis = 0)
				ci =  np.std(data["evidence"][a][mask[i], :], axis = 0) / np.sqrt(data["evidence"][a][mask[i], :].shape[0])
				
				
				ratio = 0.8
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)
				plt.plot(t, evidence,  c = c)
				plt.fill_between(t, evidence + ci, evidence - ci, color = palette[i], alpha = 0.1)
				for v in bt:
					plt.axvline(v, c = "grey", linestyle = "dashed")
				plt.legend(loc="upper left")
				plt.xlabel("Trial")
				plt.ylabel("Normalized firing rate")
				plt.title("Difference between two contextual populations\nfor {} over {} trials".format(a, max_trial))
				sns.despine()
				plt.savefig("fig/experiment{}_evidence_{}.pdf".format(num,a))
				plt.close()

			if a == "Thalamocortical Model":
				if i == 0:
					c = "purple"
				elif i == 1:
					c = "purple"
				elif i == 2:
					c = "green"
				else:
					c = "black"


				evidence = np.vstack([ x["MD"][0] - x["MD"][1] for x in data["histogram"][a]]).reshape(max_episode, max_trial)
				
				evidence[evidence <= 0] = -evidence[evidence <= 0]
				evidence = 2- 2 / (1 + np.exp(-evidence )) 
				evidence_data = np.mean(evidence, axis = 0)
				ci =  np.std(evidence, axis = 0) / np.sqrt(evidence.shape[0])
				


				ratio = 0.8
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)


				plt.plot(t+1, evidence_data,  c = palette[i])
				plt.fill_between(t + 1, evidence_data + ci, evidence_data - ci, color = palette[i], alpha = 0.1)
				

				for v in bt:
					plt.axvline(v, c = "grey", linestyle = "dashed")
				plt.legend(loc="upper left")
				plt.xlabel("Trial")
				plt.ylabel("Probability density")
				plt.title("Contextual uncertainty for {} over {} trials".format(a, max_trial))
				sns.despine()
				plt.savefig("fig/experiment{}_uncertainty_{}.pdf".format(num,a))
				plt.close()

			if a == "Thalamocortical Model":
				md = np.vstack([ x["VIP"] - x["PV"] for x in data["histogram"][a]]).reshape(max_episode, max_trial, 2)
				

				ratio = 0.8
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)
				for j in range(2):
					md_lr = relu(2 /  (1 + np.exp( - 2*(md +0.25)))-1)
					md_lr_data = np.mean(md_lr, axis = 0)

					ci =  np.std(md_lr, axis = 0) / np.sqrt(md_lr.shape[0])
					
					plt.plot(t + 1, md_lr_data[:, j], label = "Context {}".format(j+1), c = palette[j])
					plt.fill_between(t + 1, md_lr_data[:, j] + ci[:, j], md_lr_data[:, j] - ci[:, j], color = palette[j], alpha = 0.1)
				
				for v in bt:
					plt.axvline(v, c = "grey", linestyle = "dashed")
				plt.legend(loc="upper left")
				plt.xlabel("Trial")
				plt.ylabel("Learning rate")
				plt.title("Learning rate modulation from cortical interneurons over {} trials".format(max_trial))
				sns.despine()
				plt.savefig("fig/experiment{}_lr_in_{}.pdf".format(num,a))
				plt.close()

			if a == "Thalamocortical Model":
				md = np.vstack([ x["MD"] for x in data["histogram"][a]]).reshape(max_episode, max_trial, 2)
			

				
				ratio = 0.8
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)

				for j in range(2):

					md_lr = relu(2 /  (1 + np.exp(8 - 4*(md)))- 1)
					md_lr_data = np.mean(md_lr, axis = 0)

					ci =  np.std(md_lr, axis = 0) / np.sqrt(md_lr.shape[0])
					
					plt.plot(t + 1, md_lr_data[:, j], label = "Context {}".format(j+1), c = palette[j])
					plt.fill_between(t + 1, md_lr_data[:, j] + ci[:, j], md_lr_data[:, j] - ci[:, j], color = palette[j], alpha = 0.1)
				


				for v in bt:
					plt.axvline(v, c = "grey", linestyle = "dashed")
				plt.legend(loc="upper left", frameon = False)
				plt.xlabel("Trial")
				plt.ylabel("Learning rate")
				plt.title("Learning rate of PFC-MD plasticity over {} trials".format(max_trial))
				sns.despine()
				plt.savefig("fig/experiment{}_lr_bcm_{}.pdf".format(num,a))
				plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		for i, a in enumerate(agents):
			if a == "Thalamocortical Model" or a == "HMM" or a == "learned HMM":

				width = [1, 3]
				width_patch = [mlines.Line2D([], [], color = "grey",linewidth=width[i], label="Context {}".format(i+1)) for i in range(2)]
				legend_width = fig.legend(handles = width_patch, frameon = False, bbox_to_anchor=(0.18, 1), loc="upper left")
		
				
				for k in range(2):
					for s in range(1):
						for j in range(2):
							value = data["value"][a][mask[i], :]
							
							if j == 0:
								value_data = np.mean(value, axis = 0)[k][s][j]
								ci =  np.std(value[:, k, s, j], axis = 0) / np.sqrt(value[:, k, s, j].shape[0])
								ax.plot(t+1, np.mean(value, axis = 0)[k][s][j],  label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k])
								ax.fill_between(t + 1, value_data + ci, value_data - ci, color = palette[i], alpha = 0.1)
							else:
								value_data = np.mean(value, axis = 0)[k][s][j]
								ci =  np.std(value[:, k, s, j], axis = 0) / np.sqrt(value[:, k, s, j].shape[0])
								ax.plot(t+1,  np.mean(value, axis = 0)[k][s][j], label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k], linestyle = "dashed")
								ax.fill_between(t + 1, value_data + ci, value_data - ci, color = palette[i], alpha = 0.1)
					
				line_patch = [mlines.Line2D([], [],color="grey", label=i, linestyle = t) for i, t in [("Left", "solid"), ("Right", "dashed")]]

		
				#legend_line = plt.legend(handles = line_patch,  title = "Action", frameon = False, bbox_to_anchor=(0.95, 0.5), loc="upper left")
				fig.legend(handles = line_patch, bbox_to_anchor=(0.05, 1), loc = "upper left", frameon = False)
				plt.xlabel("Trial")
				plt.ylabel("Estimated value")
				plt.title("Estimated contextual value over {} trials".format(max_trial))
				sns.despine()
				plt.savefig("fig/experiment{}_value_{}.pdf".format(num,a))
				plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		for i, a in enumerate(agents):
			if a == "Thalamocortical Model" or a == "HMM" or a == "learned HMM":
				width = [1, 3]
				width_patch = [mlines.Line2D([], [], color = "grey",linewidth=width[i], label="Context {}".format(i+1)) for i in range(2)]
				legend_width = fig.legend(handles = width_patch, frameon = False, bbox_to_anchor=(0.18, 1), loc="upper left")
		
				for k in range(2):
					for s in range(1):
						for j in range(2):
							model = data["model"][a][mask[i], :][:, :, :, :, 1]
							if j == 0:
								model_data = np.mean(model, axis = 0)[k][s][j]
								ci =  np.std(model[:, k, s, j], axis = 0) / np.sqrt(model[:, k, s, j].shape[0])
								ax.plot(t+1, np.mean(model, axis = 0)[k][s][j],  label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k])
								ax.fill_between(t + 1, model_data + ci, model_data - ci, color = palette[i], alpha = 0.1)
							else:
								model_data = np.mean(model, axis = 0)[k][s][j]
								ci =  np.std(model[:, k, s, j], axis = 0) / np.sqrt(model[:, k, s, j].shape[0])
								ax.plot(t+1,  np.mean(model, axis = 0)[k][s][j], label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k], linestyle = "dashed")
								ax.fill_between(t + 1, model_data + ci, model_data - ci, color = palette[i], alpha = 0.1)
					

				fig.legend(handles = line_patch, bbox_to_anchor=(0.05, 1), loc = "upper left", frameon = False)
				plt.xlabel("Trial")
				plt.ylabel("Estimated Probability")
				plt.title("Generative model of receiving reward over {} trials".format(max_trial))
				sns.despine()
				plt.savefig("fig/experiment{}_model_{}.pdf".format(num, a))
				plt.close()
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		for i, a in enumerate(agents):
			if a == "Thalamocortical Model":
			
				quantile_data = data["quantile_data"][a]

				
				quantile_data = np.mean(quantile_data, axis = 0)[:, 0, :, :, :].reshape(2 * 2 * 100, max_trial)

				im = ax.imshow(quantile_data, cmap = "YlGn_r")
				cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal")
				cbar.ax.set_xlabel("Activities")

				ax.set_xticks(np.arange(0, 1001, 200))
				ax.set_yticks([50, 150, 250, 350])
				ax.set_yticklabels(["1/L", "1/R", "2/L", "2/R"])
				for v in bt:
					plt.axvline(v, c = "grey", linewidth = 1)
				# plt.legend(loc="upper left")
				plt.xlabel("Trial")
				plt.ylabel("[Context/Action]")
				plt.title("Striatal activities in {} model over {} trials".format(a, max_trial))
				plt.tight_layout()
				plt.savefig("fig/experiment{}_{}_quantilemap.pdf".format(num, a))
				plt.close()



			






	if num == 7 or num == 9 or num == 13:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		
	
		switch = len(data["switch"][agents[0]][0])

		max_trial = len(data["action"][agents[0]][0])
		max_episode = len(data["action"][agents[0]])
		
		block_size = 200

		bt = np.arange(0, 1000, 200)

		print("The following models are tested in a probability reversal task: {}".format(agents))
		

		
	

		switch_data = {}
		switch_bool = {}
		
		for a in agents:
			switch_data[a] = np.zeros((max_episode, switch))
			switch_bool[a] = [[False] * switch  for _ in range(max_episode)] 
			for i in range(max_episode):
				for j, action in enumerate(data["action"][a][i]): 
					
					idx = int(j / 200) - 1

					if idx > 1:
						if not switch_bool[a][i][idx-2]:
							
							switch_data[a][i, idx-2] = 200
							switch_bool[a][i][idx - 2] = True
							
							#print(task.trial,self.switch_time[agent.name][idx-2])
					
					if idx > -1 and not switch_bool[a][i][idx]:
						new_a = (idx+1) % 2


						# if agent.get_ev()[0][new_a] - agent.get_ev()[0][1-new_a] >= 0.1:
						# 	self.switch_time[agent.name][idx] += task.trial  - (idx + 1) * self.opt["block_size"] 
						# 	switch_bool[agent.name][idx] = True
						# 	print(task.trial, idx, self.switch_time[agent.name][idx])

						if new_a == 1:
							if data["choice_prob"][a][i][j] >= 0.8:
								switch_data[a][i][idx] += j+1  - (idx + 1) * 200
								switch_bool[a][i][idx] = True
								#print(self.switch_time[agent.name][idx])

						else:
							if data["choice_prob"][a][i][j] <= 0.2:
								switch_data[a][i][idx]  += j+1  - (idx + 1) * 200
								switch_bool[a][i][idx] = True
		smooth_switch_data = {}
		for a in agents:
			smooth_switch_data[a] = np.zeros((max_episode, switch))
			for i in range(max_episode):
				for j in range(switch):
					if j == 0:
						smooth_switch_data[a][i][j] = switch_data[a][i][j]
					smooth_switch_data[a][i][j] = (switch_data[a][i][j] + switch_data[a][i][j-1]) / 2.0

	



		t = np.arange(max_trial)
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		for i, a in enumerate(agents):
		

			regret = np.mean(data["regret"][a], axis = 0)
			ci =  np.std(data["regret"][a], axis = 0) / np.sqrt(data["regret"][a].shape[0])
			
			plt.plot(t, regret, label = a, c = palette[i])
			plt.fill_between(t, regret + ci, regret - ci, color = palette[i], alpha = 0.1)
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		regret_data = []
		label_data = []
		for i, a in enumerate(agents):
			regret_data.append(data["regret"][a][:, -1])
			label_data.append(a)
		ax.boxplot(regret_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks
		if num == 7:
			res = scipy.stats.shapiro(regret_data[0])
			print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(regret_data[1])
			print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			z, p = scipy.stats.mannwhitneyu(regret_data[0], regret_data[1])
			print("The p value of two-way rank sum test on the accumulated regret in a probability reversal task is {}".format(p))
		
			print("regret for {} = {}, sem = {}".format(agents[0], np.mean(regret_data[0]), np.std(regret_data[0]) / np.sqrt(len(regret_data[0]))))
			print("regret for {} = {}, sem = {}".format(agents[1],  np.mean(regret_data[1]), np.std(regret_data[1]) / np.sqrt(len(regret_data[1]))))


		if num == 9:
			res = scipy.stats.shapiro(regret_data[0])
			print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(regret_data[1])
			print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			res = scipy.stats.shapiro(regret_data[2])
			print("The p value of shapiro test on the accumulated regret of {} in probability reversal task is {}".format(agents[2], res.pvalue))
	
			z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
			print("The p value of Kruskal Wallis test on the accumulated regret in a probability reversal task is {}".format(p))
		
			p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the accumulated regret is detailed in the following matrix:")
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
		
			print(p)
			
			print("regret for {} = {}, sem = {}".format(agents[0], np.mean(regret_data[0]), np.std(regret_data[0]) / np.sqrt(len(regret_data[0]))))
			print("regret for {} = {}, sem = {}".format(agents[1],  np.mean(regret_data[1]), np.std(regret_data[1]) / np.sqrt(len(regret_data[1]))))
			print("regret for {} = {}, sem = {}".format(agents[2], np.mean(regret_data[2]), np.std(regret_data[2]) / np.sqrt(len(regret_data[2]))))
				


			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret_box.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		for i, a in enumerate(agents):
		

			regret = np.mean(data["regret"][a], axis = 0)
			ci =  np.std(data["regret"][a], axis = 0) / np.sqrt(data["regret"][a].shape[0])
			
			plt.plot(t[600:], regret[600:] - regret[600], label = a, c = palette[i])
			plt.fill_between(t[600:], regret[600:] - regret[600] + ci[600:], regret[600:] - regret[600] - ci[600:], color = palette[i], alpha = 0.1)
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials after three switches".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret_after_three_switches.pdf".format(num), transparent = True)
		plt.close()

		for i, a in enumerate(agents):
			idx = np.random.randint(0, max_episode, 3)
			for j in range(3):
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)


				action_data = data["action"][a][idx[j], :]
			
				plt.scatter((t+1), action_data + 0.05 * np.random.normal(0, 1, 1000), marker = 'x') 
				for v in bt:
					plt.axvline(v, c = "grey", linewidth = 1, linestyle = "dashed")
				plt.xlabel("Trials")
				plt.ylabel("Action")
				ax.set_yticks([0, 1])
				sns.despine()
				plt.tight_layout()
				plt.savefig("fig/experiment{}_sample_action_{}_{}.pdf".format(num,a, j), transparent = True)
				plt.close()

				

			
				ratio = 0.8
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)
			
				quantile_data = data["quantile_data"][a][idx[j], :, 0, :].reshape(2 * 2 * 100, max_trial)

				im = ax.imshow(quantile_data, cmap = "YlGn_r")
				cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal")
				cbar.ax.set_xlabel("Activities")

				ax.set_xticks(np.arange(0, 1001, 200))
				ax.set_yticks([50, 150, 250, 350])
				ax.set_yticklabels(["1/L", "1/R", "2/L", "2/R"])
				for v in bt:
					plt.axvline(v, c = "grey", linewidth = 1)
				# plt.legend(loc="upper left")
				plt.xlabel("Trial")
				plt.ylabel("[Context/Action]")
				plt.title("Striatal activities in {} model over {} trials".format(a, max_trial))
				plt.tight_layout()
				plt.savefig("fig/experiment{}_{}_sample_quantilemap_{}.pdf".format(num, a, j))
				plt.close()

			
				if i == 0:
					c = "lawngreen"
				elif i == 1:
					c = "purple"
				elif i == 2:
					c = "green"
				else:
					c = "black"

				evidence = np.vstack([ x["MD"][0]- x["MD"][1] for x in data["histogram"][a]]).reshape(max_episode, max_trial)[idx[j], :]

				
				ratio = 0.8
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)
				plt.plot(t, evidence,  c = c)
				for v in bt:
					plt.axvline(v, c = "grey", linestyle = "dashed")
				plt.legend(loc="upper left")
				plt.xlabel("Trial")
				plt.ylabel("Normalized firing rate")
				plt.title("Difference between two contextual populations\nfor {} over {} trials".format(a, max_trial))
				sns.despine()
				plt.savefig("fig/experiment{}_sample_evidence_{}_{}.pdf".format(num,a, j))
				plt.close()

		

	

		t = np.arange(max_trial)
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		for i in range(2):
			
			task = data["task"]
			line_style = ["solid", "dashed"]
			action = ["Left", "Right"]
			
			plt.plot(t, task[0][i], label = action[i], c = palette[-4], linestyle = line_style[i] )
		plt.legend(loc="upper left")
		plt.ylim(0, 1)
		plt.xlabel("Trial")
		plt.ylabel("Probability of reward")
		plt.title("Task reward structure")
		sns.despine()
		plt.savefig("fig/experiment{}_task.pdf".format(num), transparent = True)
		plt.close()


		
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		

		for i, a in enumerate(agents):
				
			choice_prob = np.mean(data["action"][a], axis = 0)
			ci =  np.std(data["action"][a], axis = 0) / np.sqrt(data["action"][a].shape[0])
			
			plt.plot(t + 1, choice_prob, label = a, c = palette[i])
			plt.fill_between(t + 1, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
		
		for v in bt:
			plt.axvline(v, c = "grey", linewidth = 1, linestyle = "dashed")
		#ax.legend(loc = "upper left")
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		plt.xlim(left = 0)
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		accuracy_data = []
		label_data = []
		for i, a in enumerate(agents):
			accuracy = np.zeros((max_episode, max_trial))
			for j in range(max_trial // block_size):
				if j % 2 == 0:
					accuracy[:, j * block_size: (j+1) * block_size] = 1-data["action"][a][:, j * block_size: (j+1) * block_size]
				else:
					accuracy[:, j * block_size: (j+1) * block_size] = data["action"][a][:, j * block_size: (j+1) * block_size]


			accuracy = np.mean(accuracy, axis = 1)

			accuracy_data.append(accuracy)
			label_data.append(a)
		ax.boxplot(accuracy_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks
		if num == 7:
			res = scipy.stats.shapiro(accuracy_data[0])
			print("The p value of shapiro test on the accuracy per run of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(accuracy_data[1])
			print("The p value of shapiro test on the accuracy per run of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			z, p = scipy.stats.mannwhitneyu(accuracy_data[0], accuracy_data[1])
			print("The p value of two-way rank sum test on the accuracy per run in a probability reversal task is {}".format(p))
			print("accuracy for {} = {}, sem = {}".format(agents[0], np.mean(accuracy_data[0]), np.std(accuracy_data[0]) / np.sqrt(len(accuracy_data[0]))))
			print("accuracy for {} = {}, sem = {}".format(agents[1], np.mean(accuracy_data[1]), np.std(accuracy_data[1]) / np.sqrt(len(accuracy_data[1]))))
		

		if num == 9:
			res = scipy.stats.shapiro(accuracy_data[0])
			print("The p value of shapiro test on the accuracy per run of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(accuracy_data[1])
			print("The p value of shapiro test on the accuracy per run of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			res = scipy.stats.shapiro(accuracy_data[2])
			print("The p value of shapiro test on the accuracy per run of {} in probability reversal task is {}".format(agents[2], res.pvalue))
	
			z, p = scipy.stats.kruskal(accuracy_data[0], accuracy_data[1], accuracy_data[2])
			print("The p value of Kruskal Wallis test on the accuracy per run in a probability reversal task is {}".format(p))
		
		
			p = sp.posthoc_dunn(accuracy_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the accuracy per run is detailed in the following matrix:")
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
		
			print(p)

			print("accuracy for {} = {}, sem = {}".format(agents[0], np.mean(accuracy_data[0]), np.std(accuracy_data[0]) / np.sqrt(len(accuracy_data[0]))))
			print("accuracy for {} = {}, sem = {}".format(agents[1], np.mean(accuracy_data[1]), np.std(accuracy_data[1]) / np.sqrt(len(accuracy_data[1]))))
			print("accuracy for {} = {}, sem = {}".format(agents[2], np.mean(accuracy_data[2]), np.std(accuracy_data[2]) / np.sqrt(len(accuracy_data[2]))))
		

	

		
			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accuracy")
		plt.title("Accuracy over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_accuracy_box.pdf".format(num), transparent = True)
		plt.close()



		win_switch_data = []
		win_stay_data = []
		lose_switch_data = []
		lose_stay_data = []

		for i, a in enumerate(agents):
			action_data = data["action"][a][:, block_size-1:]
			reward_data = data["reward"][a]
			reward_data = reward_data[:, 1:] - reward_data[:, :-1]
			reward_data = reward_data[:,block_size-1:-1]
			
			
			win_switch = np.zeros(max_episode)
			win_stay = np.zeros(max_episode)
			lose_switch = np.zeros(max_episode)
			lose_stay = np.zeros(max_episode)
			winning = np.zeros(max_episode)
			losing = np.zeros(max_episode)



			for s in range(reward_data.shape[0]):
				for j, r in enumerate(reward_data[s]):
									
					if int(r) == 1:
						winning[s] += 1 
						
						if action_data[s][j] == action_data[s][j+1]:
							win_stay[s]+= 1
						else:
							win_switch[s] +=1
					else:
						losing[s] +=1 
						if action_data[s][j] == action_data[s][j+1]:
							lose_stay[s]+= 1
						else:
							lose_switch[s] +=1


			win_switch = win_switch / winning
			win_stay = win_stay / winning
			lose_switch = lose_switch / winning
			lose_stay = lose_stay / winning

			win_switch_data.append(win_switch)
			win_stay_data.append(win_stay)
			lose_switch_data.append(lose_switch)
			lose_stay_data.append(lose_stay)

		if num == 7:
			res = scipy.stats.shapiro(win_switch_data[0])
			print("The p value of shapiro test on the win switch rate of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(win_switch_data[1])
			print("The p value of shapiro test on the win switch rate of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			z, p = scipy.stats.mannwhitneyu(win_switch_data[0], win_switch_data[1])
			print("The p value of two-way rank sum test on the win switch rate in a probability reversal task is {}".format(p))
			print("win switch rate for {} = {}, sem = {}".format(agents[0], np.mean(win_switch_data[0]), np.std(win_switch_data[0]) / np.sqrt(len(win_switch_data[0]))))
			print("win switch rate for {} = {}, sem = {}".format(agents[1], np.mean(win_switch_data[1]), np.std(win_switch_data[1]) / np.sqrt(len(win_switch_data[1]))))
		

		if num == 9:
			res = scipy.stats.shapiro(win_switch_data[0])
			print("The p value of shapiro test on the win switch rate of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(win_switch_data[1])
			print("The p value of shapiro test on the win switch rate of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			res = scipy.stats.shapiro(win_switch_data[2])
			print("The p value of shapiro test on the win switch rate of {} in probability reversal task is {}".format(agents[2], res.pvalue))
	
			z, p = scipy.stats.kruskal(win_switch_data[0], win_switch_data[1], win_switch_data[2])
			print("The p value of Kruskal Wallis test on the win switch rate in a probability reversal task is {}".format(p))
		
		
			p = sp.posthoc_dunn(win_switch_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the win switch rate is detailed in the following matrix:")
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			print(p)

			print("win switch rate for {} = {}, sem = {}".format(agents[0], np.mean(win_switch_data[0]), np.std(win_switch_data[0]) / np.sqrt(len(win_switch_data[0]))))
			print("win switch rate for {} = {}, sem = {}".format(agents[1], np.mean(win_switch_data[1]), np.std(win_switch_data[1]) / np.sqrt(len(win_switch_data[1]))))
			print("win switch rate for {} = {}, sem = {}".format(agents[2], np.mean(win_switch_data[2]), np.std(win_switch_data[2]) / np.sqrt(len(win_switch_data[2]))))
		

		if num == 7:
			res = scipy.stats.shapiro(win_stay_data[0])
			print("The p value of shapiro test on the win stay rate of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(win_stay_data[1])
			print("The p value of shapiro test on the win stay rate of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			z, p = scipy.stats.mannwhitneyu(win_stay_data[0], win_stay_data[1])
			print("The p value of two-way rank sum test on the win stay rate in a probability reversal task is {}".format(p))
			print("win stay rate for {} = {}, sem = {}".format(agents[0], np.mean(win_stay_data[0]), np.std(win_stay_data[0]) / np.sqrt(len(win_stay_data[0]))))
			print("win stay rate for {} = {}, sem = {}".format(agents[1], np.mean(win_stay_data[1]), np.std(win_stay_data[1]) / np.sqrt(len(win_stay_data[1]))))
		

		if num == 9:
			res = scipy.stats.shapiro(win_stay_data[0])
			print("The p value of shapiro test on the win stay rate of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(win_stay_data[1])
			print("The p value of shapiro test on the win stay rate of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			res = scipy.stats.shapiro(win_stay_data[2])
			print("The p value of shapiro test on the win stay rate of {} in probability reversal task is {}".format(agents[2], res.pvalue))
	
			z, p = scipy.stats.kruskal(win_stay_data[0], win_stay_data[1], win_stay_data[2])
			print("The p value of Kruskal Wallis test on the win stay rate in a probability reversal task is {}".format(p))
		 
		
			p = sp.posthoc_dunn(win_stay_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the win switch rate is detailed in the following matrix:")
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			print(p)

			print("win stay rate for {} = {}, sem = {}".format(agents[0], np.mean(win_stay_data[0]), np.std(win_stay_data[0]) / np.sqrt(len(win_stay_data[0]))))
			print("win stay rate for {} = {}, sem = {}".format(agents[1], np.mean(win_stay_data[1]), np.std(win_stay_data[1]) / np.sqrt(len(win_stay_data[1]))))
			print("win stay rate for {} = {}, sem = {}".format(agents[2], np.mean(win_stay_data[2]), np.std(win_stay_data[2]) / np.sqrt(len(win_stay_data[2]))))
		
		if num == 7:
			res = scipy.stats.shapiro(lose_switch_data[0])
			print("The p value of shapiro test on the lose switch rate of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(lose_switch_data[1])
			print("The p value of shapiro test on the lose switch rate of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			z, p = scipy.stats.mannwhitneyu(lose_switch_data[0], lose_switch_data[1])
			print("The p value of two-way rank sum test on the lose switch rate in a probability reversal task is {}".format(p))
			print("lose switch rate for {} = {}, sem = {}".format(agents[0], np.mean(lose_switch_data[0]), np.std(lose_switch_data[0]) / np.sqrt(len(lose_switch_data[0]))))
			print("lose switch rate for {} = {}, sem = {}".format(agents[1], np.mean(lose_switch_data[1]), np.std(lose_switch_data[1]) / np.sqrt(len(lose_switch_data[1]))))
		

		if num == 9:
			res = scipy.stats.shapiro(lose_switch_data[0])
			print("The p value of shapiro test on the lose switch rate of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(lose_switch_data[1])
			print("The p value of shapiro test on the lose switch rate of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			res = scipy.stats.shapiro(lose_switch_data[2])
			print("The p value of shapiro test on the lose switch rate of {} in probability reversal task is {}".format(agents[2], res.pvalue))
	
			z, p = scipy.stats.kruskal(lose_switch_data[0], lose_switch_data[1], lose_switch_data[2])
			print("The p value of Kruskal Wallis test on the lose switch rate in a probability reversal task is {}".format(p))
		 
		
			p = sp.posthoc_dunn(lose_switch_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the lose switch rate is detailed in the following matrix:")
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			print(p)

			print("lose switch rate for {} = {}, sem = {}".format(agents[0], np.mean(lose_switch_data[0]), np.std(lose_switch_data[0]) / np.sqrt(len(lose_switch_data[0]))))
			print("lose switch rate for {} = {}, sem = {}".format(agents[1], np.mean(lose_switch_data[1]), np.std(lose_switch_data[1]) / np.sqrt(len(lose_switch_data[1]))))
			print("lose switch rate for {} = {}, sem = {}".format(agents[2], np.mean(lose_switch_data[2]), np.std(lose_switch_data[2]) / np.sqrt(len(lose_switch_data[2]))))
		
		if num == 7:
			res = scipy.stats.shapiro(lose_stay_data[0])
			print("The p value of shapiro test on the lose stay rate of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(lose_stay_data[1])
			print("The p value of shapiro test on the lose stay rate of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			z, p = scipy.stats.mannwhitneyu(lose_stay_data[0], lose_stay_data[1])
			print("The p value of two-way rank sum test on the lose stay rate in a probability reversal task is {}".format(p))
			print("lose stay rate for {} = {}, sem = {}".format(agents[0], np.mean(lose_stay_data[0]), np.std(lose_stay_data[0]) / np.sqrt(len(lose_stay_data[0]))))
			print("lose stay rate for {} = {}, sem = {}".format(agents[1], np.mean(lose_stay_data[1]), np.std(lose_stay_data[1]) / np.sqrt(len(lose_stay_data[1]))))
		

		if num == 9:
			res = scipy.stats.shapiro(lose_stay_data[0])
			print("The p value of shapiro test on the lose stay rate of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(lose_stay_data[1])
			print("The p value of shapiro test on the lose stay rate of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			res = scipy.stats.shapiro(lose_stay_data[2])
			print("The p value of shapiro test on the lose stay rate of {} in probability reversal task is {}".format(agents[2], res.pvalue))
	
			z, p = scipy.stats.kruskal(lose_stay_data[0], lose_stay_data[1], lose_stay_data[2])
			print("The p value of Kruskal Wallis test on the lose stay rate in a probability reversal task is {}".format(p))
		 
		
			p = sp.posthoc_dunn(lose_stay_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the lose stay rate is detailed in the following matrix:")
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			print(p)

			print("lose stay rate for {} = {}, sem = {}".format(agents[0], np.mean(lose_stay_data[0]), np.std(lose_stay_data[0]) / np.sqrt(len(lose_stay_data[0]))))
			print("lose stay rate for {} = {}, sem = {}".format(agents[1], np.mean(lose_stay_data[1]), np.std(lose_stay_data[1]) / np.sqrt(len(lose_stay_data[1]))))
			print("lose stay rate for {} = {}, sem = {}".format(agents[2], np.mean(lose_stay_data[2]), np.std(lose_stay_data[2]) / np.sqrt(len(lose_stay_data[2]))))
		
		

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		bp = ax.boxplot(win_switch_data,
                     sym = '', widths = 0.7, showcaps = False, 
                     vert=True,
                     labels=agents)  # will be used to label x-ticks


		plt.ylabel("Win-switch rate")
		sns.despine()
		plt.savefig("fig/experiment{}_win_switch.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		bp = ax.boxplot(win_stay_data,
                     sym = '', widths = 0.7, showcaps = False, 
                     vert=True,
                     labels=agents)  # will be used to label x-ticks


		plt.ylabel("Win-stay rate")
		sns.despine()
		plt.savefig("fig/experiment{}_win_stay.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		bp = ax.boxplot(lose_switch_data,
                     sym = '', widths = 0.7, showcaps = False, 
                     vert=True,
                     labels=agents)  # will be used to label x-ticks


		plt.ylabel("Lose-switch rate")
		sns.despine()
		plt.savefig("fig/experiment{}_lose_switch.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		bp = ax.boxplot(lose_stay_data,
                     sym = '', widths = 0.7, showcaps = False, 
                     vert=True,
                     labels=agents)  # will be used to label x-ticks


		plt.ylabel("Lose-stay rate")
		sns.despine()
		plt.savefig("fig/experiment{}_lose_stay.pdf".format(num), transparent = True)
		plt.close()


			


			




		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		
		block_t = np.arange(block_size)
		for i, a in enumerate(agents):
		

			action_block_data = np.zeros((max_episode * ((max_trial // block_size)-1), block_size))
			for j in range(max_trial // block_size):
				if j == 0:
					continue
				if j %2 == 0:
					action_block_data[(j-1) * max_episode:(j) * max_episode ] =   np.concatenate([data["action"][a][:, j * block_size- block_size // 2 :j * block_size], 1-data["action"][a][:, j * block_size :j * block_size + block_size//2]], axis = 1)
				
				else:
					action_block_data[(j-1) * max_episode:(j) * max_episode ] =  np.concatenate([1-data["action"][a][:, j * block_size- block_size // 2 :j * block_size], data["action"][a][:, j * block_size :j * block_size + block_size//2]], axis = 1)
				

			choice_prob = np.mean(action_block_data, axis = 0)
			ci =  np.std(action_block_data, axis = 0) / np.sqrt(action_block_data.shape[0])
			
			plt.plot(block_t - block_size//2, choice_prob, label = a, c = palette[i])
			plt.fill_between(block_t - block_size//2, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
		
		
		ax.legend(loc = "lower right", frameon = False)
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		plt.xlim(left = -block_size//2, right = block_size//2)
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action_block.pdf".format(num), transparent = True)
		plt.close()



		st = np.arange(switch) + 1


		# fig, ax = plt.subplots(layout='constrained')
		# width = 0.25
		# multiplier = 0

		# x = np.arange(switch)
			
		# for i, a in enumerate(agents):
		# 	error_bar =  np.std(smooth_switch_data[a][mask, :] - 8, axis = 0) / np.sqrt(smooth_switch_data[a][mask, :].shape[0])
		# 	offset  = width * multiplier
		# 	rects = ax.bar(x + offset, np.mean(smooth_switch_data[a][mask, :] - 8, axis = 0), width, label = a, yerr = error_bar, color = palette[i])
		# 	multiplier += 1
		# 	# plt.errorbar(st,np.mean(smooth_switch_data[a][mask, :] - 8, axis = 0), yerr = error_bar, c = palette[i], label = a)
		# 	# plt.plot(st, np.mean(smooth_switch_data[a][mask, :] - 8, axis = 0), label = a, c = palette[i])
		
		# #plt.legend(loc="upper left")
		# plt.xlabel("Block")
		# plt.ylabel("Time to switch")
		# ax.set_xticks(x + width, st)
		# plt.ylim(bottom = 0)
		# plt.title("Averaged time to switch over blocks")
		# sns.despine()
		# plt.savefig("fig/experiment{}_switch_time.pdf".format(num), transparent = True)
		# plt.close()




		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		ax.yaxis.get_major_locator().set_params(integer=True)
		ax.xaxis.get_major_locator().set_params(integer=True)
		
		for i, a in enumerate(agents):
			
			switch_data = np.mean(smooth_switch_data[a], axis = 0) - 8

			error_bar =  np.std(smooth_switch_data[a], axis = 0) / np.sqrt(smooth_switch_data[a].shape[0])
			
			plt.errorbar(st + (i-1) * 0.1, switch_data, yerr = error_bar, c = palette[i], label = a)
			plt.plot(st+ (i-1) * 0.1, switch_data, label = a, c = palette[i])
			x = np.vstack([np.arange(4) for _ in range(max_episode)])
			print(x.shape)
			print(smooth_switch_data[a].shape)
			res = scipy.stats.shapiro(smooth_switch_data[a].flatten())
			print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[i], res.pvalue))
	
			res = scipy.stats.permutation_test((x.flatten(), smooth_switch_data[a].flatten()), lambda x, y: scipy.stats.spearmanr(x, y).statistic, n_resamples=100000)
			print("The spearman r coefficient of {}'s switch time is {} with p value {} in 10^6 resamples permutation test".format(agents[i], res.statistic, res.pvalue))
			
		
		# plt.legend(loc="upper right",  frameon = False)
		plt.xlabel("Block")
		plt.ylabel("Trials to switch")

		plt.title("Averaged trials to switch over blocks")
		sns.despine()
		plt.savefig("fig/experiment{}_switch_time.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		switch_data = []
		label_data = []
		for i, a in enumerate(agents):
			switch_data.append(smooth_switch_data[a].flatten())
			label_data.append(a)
		ax.boxplot(switch_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks
		
		

		if num == 7:
			res = scipy.stats.shapiro(switch_data[0])
			print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(switch_data[1])
			print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			z, p = scipy.stats.mannwhitneyu(switch_data[0], switch_data[1])
			print("The p value of two-way rank sum test on the switch time in a probability reversal task is {}".format(p))
			print("switch time for {} = {}, sem = {}".format(agents[0], np.mean(switch_data[0]), np.std(switch_data[0]) / np.sqrt(len(switch_data[0]))))
			print("switch time for {} = {}, sem = {}".format(agents[1], np.mean(switch_data[1]), np.std(switch_data[1]) / np.sqrt(len(switch_data[1]))))
		

		if num == 9:
			res = scipy.stats.shapiro(switch_data[0])
			print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[0], res.pvalue))
	
			res = scipy.stats.shapiro(switch_data[1])
			print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[1], res.pvalue))
	
			res = scipy.stats.shapiro(switch_data[2])
			print("The p value of shapiro test on the switch time of {} in probability reversal task is {}".format(agents[2], res.pvalue))
	
			z, p = scipy.stats.kruskal(switch_data[0], switch_data[1], switch_data[2])
			print("The p value of Kruskal Wallis test on the switch time in a probability reversal task is {}".format(p))
		 
		
			p = sp.posthoc_dunn(switch_data, p_adjust='bonferroni')
			print("The p value of posthoc dunn test with bonferroni correction on the switch time is detailed in the following matrix:")
			print("1 represents {}, 2 represents {}, 3 represents".format(agents[0], agents[1], agents[2]))
			print(p)

			print("switch time for {} = {}, sem = {}".format(agents[0], np.mean(switch_data[0]), np.std(switch_data[0]) / np.sqrt(len(switch_data[0]))))
			print("switch time for {} = {}, sem = {}".format(agents[1], np.mean(switch_data[1]), np.std(switch_data[1]) / np.sqrt(len(switch_data[1]))))
			print("switch time for {} = {}, sem = {}".format(agents[2], np.mean(switch_data[2]), np.std(switch_data[2]) / np.sqrt(len(switch_data[2]))))
		
			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Switching time")
		plt.title("Switching time")
		sns.despine()
		plt.savefig("fig/experiment{}_switch_box.pdf".format(num), transparent = True)
		plt.close()




		if num == 9:
			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			for v in bt:
				plt.axvline(v, c = "grey", linestyle = "dashed")

			for i, a in enumerate(agents):
				
				evidence = np.mean(data["evidence"][a], axis = 0)
				ci =  np.std(data["evidence"][a], axis = 0) / np.sqrt(data["evidence"][a].shape[0])
				plt.plot(t+1, evidence, label = a,  c = palette[i])
				plt.fill_between(t + 1, evidence + ci, evidence - ci, color = palette[i], alpha = 0.1)

			plt.legend(loc="upper left")
			plt.xlabel("Trial")
			plt.ylabel("Normalized firing rate")
			plt.title("Difference between two contextual populations".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_evidence.pdf".format(num))
			plt.close()





		for i, a in enumerate(agents):
		
			if i == 0:
				c = "lawngreen"
			elif i == 1:
				c = "purple"
			elif i == 2:
				c = "green"
			else:
				c = "black"

			evidence = np.mean(data["evidence"][a], axis = 0)
			ci =  np.std(data["evidence"][a], axis = 0) / np.sqrt(data["evidence"][a].shape[0])
			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			plt.plot(t+1, evidence,  c = c)
			plt.fill_between(t + 1, evidence + ci, evidence - ci, color = palette[i], alpha = 0.1)
			for v in bt:
				plt.axvline(v, c = "grey", linestyle = "dashed")
			plt.legend(loc="upper left")
			plt.xlabel("Trial")
			plt.ylabel("Normalized firing rate")
			plt.title("Difference between two contextual populations\nfor {} over {} trials".format(a, max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_evidence_{}.pdf".format(num,a))
			plt.close()


			md0 = np.vstack([ x["MD"][0] for x in data["histogram"][a]]).reshape(max_episode, max_trial)
			md1 = np.vstack([ x["MD"][1] for x in data["histogram"][a]]).reshape(max_episode, max_trial)

			md0_data = np.mean(md0, axis = 0)
			md1_data = np.mean(md1, axis = 0)
			ci0 =  np.std(md0, axis = 0) / np.sqrt(md0.shape[0])
			ci1 =  np.std(md1, axis = 0) / np.sqrt(md1.shape[0])

			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			plt.plot(t+1, md0_data,  c = c)
			plt.fill_between(t + 1, md0_data + ci0, md0_data - ci0, color = palette[i], alpha = 0.1)
			plt.plot(t+1, md1_data,  c = c)
			plt.fill_between(t + 1, md1_data + ci1, md1_data - ci1, color = palette[i], alpha = 0.1)
			for v in bt:
				plt.axvline(v, c = "grey", linestyle = "dashed")
			plt.legend(loc="upper left")
			plt.ylim(bottom = 3.2)
			plt.xlabel("Trial")
			plt.ylabel("Normalized firing rate")
			plt.title("MD activities in {} over {} trials".format(a, max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_md_activities_{}.pdf".format(num,a))
			plt.close()


		
			if i == 0:
				c = "purple"
			elif i == 1:
				c = "purple"
			elif i == 2:
				c = "green"
			else:
				c = "black"

			md = np.vstack([ x["MD"]  for x in data["histogram"][a]]).reshape(max_episode, max_trial, 2)
			md_lr = relu(2 /  (1 + np.exp(8 - 4*(md-4)))- 1)
			evidence = 1-np.sum(md_lr, axis = 2)
			# evidence = 2 - 2 / (1 + np.exp(-md))
		
			evidence_data = np.mean(evidence, axis = 0)
			ci =  np.std(evidence, axis = 0) / np.sqrt(evidence.shape[0])




			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)


			plt.plot(t+1, evidence_data,  c = palette[i])
			plt.fill_between(t + 1, evidence_data + ci, evidence_data - ci, color = palette[i], alpha = 0.1)
				

			for v in bt:
				plt.axvline(v, c = "grey", linestyle = "dashed")
			
			plt.legend(loc="upper left")
			plt.xlabel("Trial")
			plt.ylabel("Probability density")
			plt.title("Contextual uncertainty for {} over {} trials".format(a, max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_uncertainty_{}.pdf".format(num,a))
			plt.close()

			
		
			md = np.vstack([ x["VIP"] - x["PV"] for x in data["histogram"][a]]).reshape(max_episode, max_trial, 2)
			

			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			for j in range(2):
			
				md_lr = 1 /  (1 + np.exp( - 2*(md - 0.5)))
				md_lr_data = np.mean(md_lr, axis = 0)

				ci =  np.std(md_lr, axis = 0) / np.sqrt(md_lr.shape[0])
				
				plt.plot(t + 1, md_lr_data[:, j], label = "Context {}".format(j+1), c = palette[j])
				plt.fill_between(t + 1, md_lr_data[:, j] + ci[:, j], md_lr_data[:, j] - ci[:, j], color = palette[j], alpha = 0.1)
			
			for v in bt:
				plt.axvline(v, c = "grey", linestyle = "dashed")
			plt.legend(loc="upper left")
			plt.xlabel("Trial")
			plt.ylabel("Learning rate")
			plt.title("Learning rate modulation from cortical interneurons over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_lr_in_{}.pdf".format(num,a))
			plt.close()

			
			md = np.vstack([ x["MD"] for x in data["histogram"][a]]).reshape(max_episode, max_trial, 2)
		

			
			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)

			for j in range(2):

				md_lr = relu(2 /  (1 + np.exp(8 - 4*(md-4)))- 1)
				md_lr_data = np.mean(md_lr, axis = 0)

				ci =  np.std(md_lr, axis = 0) / np.sqrt(md_lr.shape[0])
				
				plt.plot(t + 1, md_lr_data[:, j], label = "Context {}".format(j+1), c = palette[j])
				plt.fill_between(t + 1, md_lr_data[:, j] + ci[:, j], md_lr_data[:, j] - ci[:, j], color = palette[j], alpha = 0.1)
			


			for v in bt:
				plt.axvline(v, c = "grey", linestyle = "dashed")
			plt.legend(loc="upper left", frameon = False)
			plt.xlabel("Trial")
			plt.ylabel("Learning rate")
			plt.title("Learning rate of PFC-MD plasticity over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_lr_bcm_{}.pdf".format(num,a))
			plt.close()
	

		divergence = {}
		for i, a in enumerate(agents):

			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
				
			width = [1, 3]
			width_patch = [mlines.Line2D([], [], color = "grey",linewidth=width[i], label="Context {}".format(i+1)) for i in range(2)]
			legend_width = fig.legend(handles = width_patch, frameon = False, bbox_to_anchor=(0.18, 1), loc="upper left")
	
			
			for k in range(2):
				for s in range(1):
					for j in range(2):
						value = data["value"][a]
						
						if j == 0:
							value_data = np.mean(value, axis = 0)[k][s][j]
							ci =  np.std(value[:, k, s, j], axis = 0) / np.sqrt(value[:, k, s, j].shape[0])
							ax.plot(t+1, np.mean(value, axis = 0)[k][s][j],  label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k])
							ax.fill_between(t + 1, value_data + ci, value_data - ci, color = palette[i], alpha = 0.1)
						else:
							value_data = np.mean(value, axis = 0)[k][s][j]
							ci =  np.std(value[:, k, s, j], axis = 0) / np.sqrt(value[:, k, s, j].shape[0])
							ax.plot(t+1,  np.mean(value, axis = 0)[k][s][j], label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k], linestyle = "dashed")
							ax.fill_between(t + 1, value_data + ci, value_data - ci, color = palette[i], alpha = 0.1)
				
			line_patch = [mlines.Line2D([], [],color="grey", label=i, linestyle = t) for i, t in [("Left", "solid"), ("Right", "dashed")]]

	
			#legend_line = plt.legend(handles = line_patch,  title = "Action", frameon = False, bbox_to_anchor=(0.95, 0.5), loc="upper left")
			fig.legend(handles = line_patch, bbox_to_anchor=(0.05, 1), loc = "upper left", frameon = False)
			plt.xlabel("Trial")
			plt.ylabel("Estimated value")
			plt.title("Estimated contextual value over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_value_{}.pdf".format(num,a))
			plt.close()

			
			if a == "Full model" or a == "IN KO":
				value = data["value"][a]
				divergence[a] = np.sqrt((0.7 - value[:, 0, 0, 0, -1])**2 + (0.3 - value[:, 0, 0, 1, -1])**2 + (0.3 - value[:, 1, 0, 0, -1])**2 + (0.7 - value[:, 1, 0, 1, -1])**2)


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		
		label_data = list(divergence.keys())
		divergence_data = [ divergence[a] for a in label_data]
		
		ax.boxplot(divergence_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks

		plt.ylabel("KL divergence")
		sns.despine()
		plt.savefig("fig/experiment{}_value_kl_divergence.pdf".format(num), transparent = True)
		plt.close()









		divergence = {}
		
		for i, a in enumerate(agents):
			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
		
			width = [1, 3]
			width_patch = [mlines.Line2D([], [], color = "grey",linewidth=width[i], label="Context {}".format(i+1)) for i in range(2)]
			legend_width = fig.legend(handles = width_patch, frameon = False, bbox_to_anchor=(0.18, 1), loc="upper left")
	
			for k in range(2):
				for s in range(1):
					for j in range(2):
						model = data["model"][a][:, :, :, :, 1]
						if j == 0:
							model_data = np.mean(model, axis = 0)[k][s][j]
							ci =  np.std(model[:, k, s, j], axis = 0) / np.sqrt(model[:, k, s, j].shape[0])
							ax.plot(t+1, np.mean(model, axis = 0)[k][s][j],  label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k])
							ax.fill_between(t + 1, model_data + ci, model_data - ci, color = palette[i], alpha = 0.1)
						else:
							model_data = np.mean(model, axis = 0)[k][s][j]
							ci =  np.std(model[:, k, s, j], axis = 0) / np.sqrt(model[:, k, s, j].shape[0])
							ax.plot(t+1,  np.mean(model, axis = 0)[k][s][j], label = "Thalamocortical Model {} Action {}".format(k+1, j+1), c = palette[i], linewidth = width[k], linestyle = "dashed")
							ax.fill_between(t + 1, model_data + ci, model_data - ci, color = palette[i], alpha = 0.1)
				
						print("context {}, action {}, model {}, probability {}".format(k, j, a, np.mean(model, axis = 0)[k][s][j][-1]))
			fig.legend(handles = line_patch, bbox_to_anchor=(0.05, 1), loc = "upper left", frameon = False)
			plt.xlabel("Trial")
			plt.ylabel("Estimated Probability")
			plt.title("Generative model of receiving reward over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_model_{}.pdf".format(num, a))
			plt.close()

			if a == "Full model" or a == "MD KO":
				model = data["model"][a][:, :, :, :, 1]
				divergence[a] = np.sqrt((0.7 - model[:, 0, 0, 0, -1])**2 + (0.3 - model[:, 0, 0, 1, -1])**2 + (0.3 - model[:, 1, 0, 0, -1])**2 + (0.7 - model[:, 1, 0, 1, -1])**2)

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		
		label_data = list(divergence.keys())
		divergence_data = [ divergence[a] for a in label_data]
		
		ax.boxplot(divergence_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks

		plt.ylabel("KL divergence")
		sns.despine()
		plt.savefig("fig/experiment{}_model_kl_divergence.pdf".format(num), transparent = True)
		plt.close()
		
		for i, a in enumerate(agents):
			ratio = 0.8
			fig, ax = plt.subplots()
			fig.set_figwidth(4.8 * ratio)
			fig.set_figheight(4.8 * ratio)
			
			
			quantile_data = data["quantile_data"][a]
			
			quantile_data = np.mean(quantile_data, axis = 0)[:, 0, :, :, :].reshape(2 * 2 * 100, max_trial)
			
			im = ax.imshow(quantile_data, cmap = "YlGn_r")
			cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal")
			cbar.ax.set_xlabel("Activities")

			ax.set_xticks(np.arange(0, 1001, 200))
			ax.set_yticks([50, 150, 250, 350])
			ax.set_yticklabels(["1/L", "1/R", "2/L", "2/R"])
			for v in bt:
				plt.axvline(v, c = "grey", linewidth = 1)
			# plt.legend(loc="upper left")
			plt.xlabel("Trial")
			plt.ylabel("[Context/Action]")
			plt.title("Striatal activities in {} model over {} trials".format(a, max_trial))
			plt.tight_layout()
			plt.savefig("fig/experiment{}_{}_quantilemap.pdf".format(num, a))
			plt.close()

			if a == "Full model":
				print(data["histogram"][a][0]["PFC/MD"].shape)
				ct = np.vstack([ x["PFC/MD"] for x in data["histogram"][a]]).reshape(max_episode, max_trial, 2, 1, 2, 2)[1, 599, :]
				prob = np.vstack([x["ALM/BG"] for x in data["histogram"][a]]).reshape(max_episode, max_trial, 2, 1, 2, 100)[1, 599, :]
				
				inverse_uncertainty = lambda x: 0.25 * (24 - np.log(2 / (x+1) - 1))

				md_activity = inverse_uncertainty(np.linspace(0.99, 0, 6))


				result_arr = []


				for m in md_activity:

					thalamus = [m, 12 - m]


					new_opt = {}
					new_opt["stimuli_num"] = 1
					new_opt["dt"] = 0.005
					new_opt["context_num"] = 2
					new_opt["class_num"] = 2
					new_opt["gamma1"] = 0.99
					new_opt["gamma2"] = 0.8
					new_opt["lr"] = 0.1
					new_opt["temperature"] = 30
					new_opt["tau"] = 2
					new_opt["a"] =  1
					new_opt["gamma"] = 1
					new_opt["iter"] = 40
					new_opt["md_learning"] = True
					new_opt["in_learning"] = True
					new_opt["fixmd"] = True

					new_opt["inhibit"] = False
					new_opt["d2"] = False
					new_opt["rescue"] = False
					new_opt["quantile_num"] = 100
					new_opt["N"] = 3 
					new_opt["K"] = 3
					new_opt["a"] = 1
					new_opt["b"] = 1
					new_opt["a1"] = 0.75
					new_opt["b1"] = 1
					new_opt["a2"] = 1.2
					new_opt["b2"] = 1
					new_opt["tau1"] = 10
					new_opt["eta"] = 0.5
					new_opt["threshold"] = 2
					new_opt["d_interval"] = 1000

					agent = TwoTimeScaleNeuralAgent(new_opt)
					agent.prob = prob
					agent.ct = ct
					agent.thalamus = thalamus

					result = []
					for _ in range(100):
						action = agent.forward(0)
						result.append(action)
	
					result_arr.append(result)

				result_arr = np.array(result_arr)

				fig1, ax1 = plt.subplots()
				fig1.set_figwidth(4.8 * ratio)
				fig1.set_figheight(4.8 * ratio)

				result_data = np.mean(result_arr, axis = 1)
				ci =  np.std(result_arr, axis = 1) / np.sqrt(md_lr.shape[1])

				uncertainty = np.linspace(0, 1, 6)

				plt.plot(uncertainty, result_data, c = palette[0])
				plt.fill_between(uncertainty, result_data + ci, result_data - ci, color = palette[0], alpha = 0.1)
				plt.xlabel("Contextual uncertainty")
				plt.ylabel("Probability to choose action 1")
				sns.despine()
				plt.savefig("fig/experiment{}_uncertainty_exploration.pdf".format(num))
				plt.close()

			





				# ax1.bar(0, result_data[0], 0.5)
				# ax1.bar(1, result_data[1], 0.5)
				# ax1.bar(2, result_data[2], 0.5)
				# plt.ylim(top = 1.1)

				
				# plt.ylabel("Choice probability")
				# ax1.set_xticks(np.arange(3))
				# ax1.set_xticklabels(["1", "2", "3"])
				# sns.despine()
				# plt.savefig("fig/experiment{}_exploration_{}_at_trial_{}.pdf".format(num, i, ti), transparent = True)




				
						




		
print(num)
plot(num)

# data = {}

# data_1 = load_dict("experiment33_data_2")
# data_2 = load_dict("experiment33_data_1")

# # #data_3 = load_dict("experiment11_data_4")
# keys = data_2.keys()
# agents = data_1["action"].keys()

# for k in keys:


# 	if k == "task":

# 		data[k] = np.vstack([data_1[k], data_2[k]])

		
# 	else:
# 		data[k] = {}
# 		for a in agents:

			
# 			if (k == "evidence" or k == "value" or k== "model") and (a == "Bayesian RL" or a == "Discounted Thompson Sampling"):
# 				continue

# 			elif (k == "quantile_data") and (a == "Discounted Thompson Sampling" or a == "HMM Model"):
# 				continue
# 			elif k== "scalars" or k=="histogram":
# 				data_1[k][a].extend(data_2[k][a])
# 				data[k][a] = data_1[k][a]
# 				print(len(data_1[k][a]))
# 			else:
# 				data[k][a] = np.vstack([data_1[k][a], data_2[k][a]])
# 				#data[k][a] =  data_2[k][a]


# save_dict(data, "experiment33_data")
			








