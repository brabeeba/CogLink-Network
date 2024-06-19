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

from sklearn.svm import LinearSVC 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


num =36

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
	# if num == 32:
	# 	data = load_dict("experiment{}_data".format(num))
	# 	agents = list(data["action"].keys())

	# 	pfc = np.array([x ["PFC"] for x in data["histogram"][agents[0]]]).reshape(100000, 4)
	# 	md = np.array([x ["MD"] for x in data["histogram"][agents[0]]]).reshape(100000, 2)

	# 	def f(x):
	# 		if x % 400 < 200:
	# 			return 0
	# 		else:
	# 			return 1
	# 	context = np.expand_dims(np.array([f(x) for x in range(2000)]),0) * np.ones((50, 1))
	# 	context = context.flatten()
	# 	clf = LinearSVC()
	# 	clf.fit(pfc[:80000], context[:80000])

	# 	print(md[205:220])
		
	# 	print(clf.score(pfc[-20000:], context[-20000:]))

	# 	clf = LinearSVC()
	# 	clf.fit(md[:80000], context[:80000])
		
	# 	print(clf.score(md[-20000:], context[-20000:]))
		

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
	
	if num == 36:
		ratio = 0.7

		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())

		
		print(agents)
		
		max_trial = len(data["action"][agents[0]][0, 0, :])
		t = np.arange(max_trial)


		
		delta = [ data["task"][x][i, 0, 1, 0] - data["task"][x][i, 0, 0, 0] for i, x in enumerate(data["task"].keys())]
		task_num = len(delta)
	
		label = np.linspace(0.5, 0.2, len(delta))
		label = ["{0:.2f}".format(x) for x in label]
		print(label)
		x = np.arange(len(delta))
		print(delta)
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.25
		multiplier = 0
		regret_data = []

		print("regret")
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
			z, p = scipy.stats.kruskal(regret_data[0][i], regret_data[1][i], regret_data[2][i])
			print("kruskal wallis p value for regret is {}".format(p))
			
			p = sp.posthoc_dunn([x[i] for x in regret_data], p_adjust='bonferroni')
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

		print("accuracy")
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
			z, p = scipy.stats.kruskal(accuracy_data[0][i], accuracy_data[1][i], accuracy_data[2][i])
			print("kruskal wallis p value for accuracy is {}".format(p))
			
			p = sp.posthoc_dunn([x[i] for x in accuracy_data], p_adjust='bonferroni')
			print(p)
			
		# plt.legend(loc="upper right", frameon = False)
		plt.legend(loc="upper left", frameon = False)
		ax.set_xticks(x + width, labels = label)
		plt.xlabel("\u0394")
		plt.ylabel("Regret")
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
			
			z, p = scipy.stats.kruskal(accuracy_data[0], accuracy_data[1], accuracy_data[2])
			print("kruskal wallis p value for accuracy is {}".format(p))
			
			p = sp.posthoc_dunn(accuracy_data, p_adjust='bonferroni')
			print(p)


			print("accuracy for {} = {}, sem = {}".format(agents[0], np.mean(accuracy_data[0]), np.std(accuracy_data[0]) / np.sqrt(len(accuracy_data[0]))))
			print("accuracy for {} = {}, sem = {}".format(agents[1],  np.mean(accuracy_data[1]), np.std(accuracy_data[1]) / np.sqrt(len(accuracy_data[1]))))
			print("accuracy for {} = {}, sem = {}".format(agents[2],  np.mean(accuracy_data[2]), np.std(accuracy_data[2]) / np.sqrt(len(accuracy_data[2]))))

			
				
			plt.legend(loc="upper left", frameon=False)
			plt.xlabel("Trial")
			plt.ylabel("Accumulated regret")
			plt.title("Averaged accumulated regret over {} trials".format(max_trial))
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
			
			z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
			print("kruskal wallis p value for regret is {}".format(p))
			
			p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
			print(p)


			print("accuracy for {} = {}, sem = {}".format(agents[0], np.mean(regret_data[0]), np.std(regret_data[0]) / np.sqrt(len(regret_data[0]))))
			print("accuracy for {} = {}, sem = {}".format(agents[1],  np.mean(regret_data[1]), np.std(regret_data[1]) / np.sqrt(len(regret_data[1]))))
			print("accuracy for {} = {}, sem = {}".format(agents[2],  np.mean(regret_data[2]), np.std(regret_data[2]) / np.sqrt(len(regret_data[2]))))

			
				
			plt.legend(loc="upper left", frameon=False)
			plt.xlabel("Trial")
			plt.ylabel("Accumulated regret")
			plt.title("Averaged accumulated regret over {} trials".format(max_trial))
			sns.despine()
			plt.savefig("fig/experiment{}_task_{}_regret_box.pdf".format(num, i), transparent = True)
			plt.close()






			





	if num == 15 or num == 25:
		ratio = 0.7

		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())

		# agents = [agents[1], agents[0], agents[2]]
		
		max_trial = len(data["action"][agents[0]][0, 0, :])
		t = np.arange(max_trial)


		
		delta = [ data["task"][x][i, 0, 2, 0] - data["task"][x][i, 0, 0, 0] for i, x in enumerate(data["task"].keys())]
		task_num = len(delta)
	
		label = np.linspace(0.5, 0.2, len(delta))
		label = ["{0:.2f}".format(x) for x in label]
		print(label)
		x = np.arange(len(delta))
		print(delta)
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.33
		multiplier = 0

		print("regret")
		for i, a in enumerate(agents):
			error_bar = np.std(data["regret"][a][:, :, -1], axis = 1) / np.sqrt(data["regret"][a][:, :, -1].shape[1])
			# offset  = width * multiplier
			offset  = width * multiplier + width / 2
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
			
			ax.annotate("", xy=(i+width / 2, y_max + 1), xycoords='data',
	       xytext=(i+width * 3 /2, y_max + 1), textcoords='data',
	       arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	                       connectionstyle="bar,fraction=0.2"))
			z, p = scipy.stats.mannwhitneyu(data["regret"][agents[0]][i, :, -1], data["regret"][agents[1]][i, :, -1])
			print(delta[i], p)
			ax.text(i + width, y_max + 2, stars(p),horizontalalignment='center',verticalalignment='center')
		
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
		width = 0.33
		multiplier = 0

		accuracy = {}
		for i, a in enumerate(agents):
			
			# offset  = width * multiplier
			offset  = width * multiplier + width / 2
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

		print("accuracy")
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
			z, p = scipy.stats.mannwhitneyu(accuracy[agents[0]][i], accuracy[agents[1]][i])
			print(delta[i], p)
			ax.text(i + width, y_max, stars(p),horizontalalignment='center',verticalalignment='center')
		
		# plt.legend(loc="upper right", frameon = False)
		plt.legend(loc="upper left", frameon = False)
		ax.set_xticks(x + width, labels = label)
		plt.xlabel("\u0394")
		plt.ylabel("Regret")
		if num == 15:
			ylim = 0.675
		if num == 25:
			ylim = 0.45
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
				
				if a != "Thompson Sampling":
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


					# quantile_data = data["quantile_data"][a][i]

					# bins = np.linspace(0, 1, 100)

					# timepoints = [10, 50]

					# fig, ax = plt.subplots(len(timepoints),  sharey=True)
					# fig.set_figwidth(4.8 * ratio)
					# fig.set_figheight(4.8 * 2 * ratio)


					# for k, ti in enumerate(timepoints):
					# 	for l in range(3):
					# 		quantile_num = quantile_data.shape[3]

					# 		if ti == 0:
					# 			df = np.arange(quantile_num) * 1.0 / quantile_num
					# 		else:

					# 			df = np.mean(quantile_data, axis = 0)[0, l, :, ti-1]

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
					# plt.tight_layout()
					
					
					# #plt.title("Averaged accumulated regret over {} trials".format(max_trial))
					
					# #plt.show()
					# plt.savefig("fig/experiment{}_posterior_task{}.pdf".format(num, i), transparent = True)
					# plt.close()

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

				if a != "Thompson Sampling":
					print(data["quantile_data"][a][0].shape)
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



	if num == 35:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		max_trial = len(data["action"][agents[0]][0])
		t = np.arange(max_trial)

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		
		
		for i, a in enumerate(agents):
			
			choice_prob = np.mean(data["action"][a], axis = 0)
			ci = np.std(data["action"][a], axis = 0) / np.sqrt(data["action"][a].shape[0])
			
			plt.plot(t + 1, choice_prob, label = a, c = palette[i])
			plt.fill_between(t + 1, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
		#plt.xscale('log')
		plt.legend(loc="lower right", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		plt.xscale("log")
		ax.set_xlim(right = max_trial)
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		plt.close()


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.5
		multiplier = 0
		x = np.arange(len(agents))

		for i, a in enumerate(agents):
			error_bar = np.std(data["regret"][a][:,  -1], axis = 0) / np.sqrt(data["regret"][a][:,  -1].shape[0])
			ax.bar(x[i], np.mean(data["regret"][a][:, -1], axis = 0), width, label = a, yerr = error_bar)
		

			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a, alpha = 0.8)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		
		y_max = np.max([np.mean(data["regret"][a][:,  -1]) +  np.std(data["regret"][a][:,  -1], axis = 0) / np.sqrt(data["regret"][a][:,  -1].shape[0])  for a in agents])
		y_min = np.min([np.mean(data["regret"][a][:,  -1]) for a in agents])
			
		ax.annotate("", xy=(0, y_max ), xycoords='data',
	       xytext=(1, y_max ), textcoords='data',
	       arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	                       connectionstyle="bar,fraction=0.1"))
		z, p = scipy.stats.mannwhitneyu(data["regret"][agents[0]][:, -1], data["regret"][agents[1]][:, -1])
		print(p)
		ax.text(0.5, y_max , stars(p),horizontalalignment='center',verticalalignment='center')
		
		
		#plt.legend(loc="upper left", frameon = False)
	
		plt.ylabel("Regret")
		plt.ylim(bottom = 0)
		plt.xticks(x, agents)
		plt.title("Averaged regret in the environment with fixed volatility".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

	# if num == 25:
	# 	data = load_dict("experiment{}_data".format(num))
	# 	agents = list(data["action"].keys())
	# 	max_trial = len(data["action"][agents[0]][0])
	# 	t = np.arange(max_trial)



	# 	plt.figure(dpi=1200)

	# 	delta = [ data["task"][x][i, 0, 1, 0] - data["task"][x][i, 0, 0, 0] for i, x in enumerate(data["task"].keys())]
	# 	print(delta)
	# 	task_num = len(delta)
	# 	label = np.linspace(0.5, 0.2, len(delta))
	# 	label = ["{0:.2f}".format(x) for x in label]
	# 	print(label)
	# 	x = np.arange(len(delta))

	# 	fig, ax = plt.subplots(layout='constrained')
	# 	ratio = 0.7
	# 	fig.set_figwidth(4.8 * ratio)
	# 	fig.set_figheight(4.8 * ratio)
	# 	width = 0.33
	# 	multiplier = 0

	
	# 	for i, a in enumerate(agents):
	# 		error_bar = np.std(data["regret"][a][:, :, -1], axis = 1) / np.sqrt(data["regret"][a][:, :, -1].shape[1])
	# 		offset  =  width * (multiplier+0.5)
	# 		rects = ax.bar(x + offset, np.mean(data["regret"][a][:, :, -1], axis = 1), width, label = a, yerr = error_bar)
	# 		# for j in range(len(delta)):
	# 		# 	ax.violinplot(data["regret"][a][j, :, -1], positions = [x[j]+offset])
			
	# 		multiplier += 1
	# 		#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a)
	# 		#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		
	# 	for j, i in enumerate(range(len(delta))):
	# 		y_max = np.max([np.mean(data["regret"][a][i, :, -1]) +  np.std(data["regret"][a][i, :, -1]) / np.sqrt(len(data["regret"][a][i, :, -1]))  for a in agents[1:]])
	# 		y_min = np.min([data["regret"][a][i, :, -1] for a in agents[1:]])
			
	# 		ax.annotate("", xy=(j+0.165, y_max + 2), xycoords='data',
	#        xytext=(j+0.495, y_max + 2), textcoords='data',
	#        arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	#                        connectionstyle="bar,fraction=0.2"))
	# 		print(data["regret"][agents[0]].shape)
	# 		print(data["regret"][agents[0]][i, :, -1])
	# 		print(data["regret"][agents[1]][i, :, -1])
	# 		#z, p = scipy.stats.ttest_ind(data["regret"][agents[0]][i, :, -1], data["regret"][agents[1]][i, :, -1])
	# 		z, p = scipy.stats.kruskal(data["regret"][agents[0]][i, :, -1], data["regret"][agents[1]][i, :, -1])
	# 		print(p)
	# 		ax.text(j + 0.33, y_max + 4, stars(p),horizontalalignment='center',verticalalignment='center')
		
	# 	plt.legend(loc="upper right", frameon = False)
	# 	ax.set_xticks(x + width, label)
	# 	plt.xlabel("\u0394")
	# 	plt.ylabel("Regret")
	# 	plt.ylim(bottom = 0)
		
	# 	#plt.title("Averaged regret over different \u0394 between two alternatives".format(max_trial))
	# 	sns.despine()
	# 	plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
	# 	plt.close()

	# 	for i in range(task_num):
			
	# 		fig, ax = plt.subplots()
	# 		fig.set_figwidth(4.8 * ratio)
	# 		fig.set_figheight(4.8 * ratio)
	# 		for j, a in enumerate(agents):
				
	# 			accuracy = np.copy(data["action"][a][i])
	# 			accuracy[accuracy < 2] = 0
	# 			accuracy[accuracy == 2] = 1
	# 			choice_prob = np.mean(accuracy, axis = 0)
	# 			ci = np.std(accuracy, axis = 0) / np.sqrt(accuracy.shape[0])
				
	# 			plt.plot(t + 1, choice_prob, label = a, c = palette[j])
	# 			plt.fill_between(t + 1, choice_prob + ci, choice_prob - ci, color = palette[j], alpha = 0.1)
	# 		#plt.xscale('log')
	# 		plt.legend(loc="lower right", frameon=False)
	# 		plt.xlabel("Trial")
	# 		plt.ylim(bottom = 0.15, top = 1)
	# 		plt.ylabel("Accurate choice probability")
	# 		plt.xscale("log")
	# 		ax.set_xlim(right = max_trial)
	# 		plt.title("Choice probability over {} trials".format(max_trial))
	# 		sns.despine()
	# 		plt.savefig("fig/experiment{}_task_{}action.pdf".format(num, i), transparent = True)
	# 		plt.close()



	# 		for j, a in enumerate(agents):
				
	# 			if a != "Thompson Sampling":
	# 				fig, ax = plt.subplots()
	# 				fig.set_figwidth(4.8 * ratio)
	# 				fig.set_figheight(4.8 * ratio)
	# 				quantile_data = np.mean(data["quantile_data"][a][i], axis = 0)[0, :]
					
	# 				class_num, quantile_num, max_trial = quantile_data.shape
	# 				quantile_data = quantile_data.reshape( class_num * quantile_num, max_trial)
	# 				#quantile_data[quantile_data > 1] = 1

	# 				im = ax.imshow(quantile_data, interpolation='nearest', aspect='auto', cmap = "YlGn_r")
	# 				plt.axvline(10, c = "grey", linewidth = 1, linestyle = "dashed")
	# 				plt.axvline(50, c = "grey", linewidth = 1, linestyle = "dashed")
	# 				im.set_clim(0, 1)
	# 				cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal", pad = 0.2)
	# 				cbar.ax.set_xlabel("Activities")

	# 				task_prob =[ "{0:.2f}".format(x) for x in data["task"]["gap {}".format(i)][i, 0, :, 0]]

	# 				# ax.set_xticks(np.arange(0, 2001, 200))
	# 				ax.set_yticks([50, 150, 250])
					
	# 				ax.set_yticklabels(task_prob, rotation = 90)

					
	# 				# plt.legend(loc="upper left")
	# 				plt.xlabel("Trial")
	# 				plt.ylabel("Action")
	# 				plt.title("Striatal activities over {} trials".format(max_trial))
	# 				plt.xscale('log')
	# 				#plt.tight_layout()

	# 				plt.savefig("fig/experiment{}_{}_task{}_quantilemap.pdf".format(num, a, i))
	# 				plt.close()


	# 				quantile_data = data["quantile_data"][a][i]

	# 				bins = np.linspace(0, 1, 100)

	# 				timepoints = [10, 50]

	# 				fig, ax = plt.subplots(len(timepoints),  sharey=True)
	# 				fig.set_figwidth(4.8 * ratio)
	# 				fig.set_figheight(4.8 * 2 * ratio)


	# 				for k, ti in enumerate(timepoints):
	# 					for l in range(3):
	# 						quantile_num = quantile_data.shape[3]

	# 						if ti == 0:
	# 							df = np.arange(quantile_num) * 1.0 / quantile_num
	# 						else:

	# 							df = np.mean(quantile_data, axis = 0)[0, l, :, ti-1]

	# 						# df = (np.expand_dims(df, 0) + np.expand_dims(df, 1))/2
	# 						# df = df.flatten()

	# 						df = (np.expand_dims(df, (0, 1)) + np.expand_dims(df, (1, 2)) + np.expand_dims(df, (0, 2)))/3
	# 						df = df.flatten()

	# 						diff = np.expand_dims(bins, 1) - np.expand_dims(df, 0)
	# 						std = 0.03
	# 						q_data = np.sum(np.exp(-0.5 * diff ** 2 / std ** 2) / (np.sqrt(2 * np.pi) * std), axis = 1) / len(df)
	# 						if l == 0:
	# 							ax[k].plot(bins, q_data, label = "{}".format(l+1), color = palette_alt[l])
	# 						if l == 1:
	# 							ax[k].plot(bins, q_data, label = "{}".format(l+1), linestyle = "dashed", color = palette_alt[l])
	# 						if l == 2:
	# 							ax[k].plot(bins, q_data, label = "{}".format(l+1), linestyle = "dashed", color = palette_alt[l])


	# 					ax[k].title.set_text("Posterior at trial {}".format(ti))
						
	# 				handles, labels = ax[0].get_legend_handles_labels()
					
	# 				fig.legend(handles, labels,  bbox_to_anchor=(1, 0.95), loc = "upper right", frameon=False)
	# 				sns.despine()
	# 				fig.supxlabel("Values", x= 0.625, y = 0.06,  fontsize = "small")
	# 				plt.tight_layout()
					
					
	# 				#plt.title("Averaged accumulated regret over {} trials".format(max_trial))
					
	# 				#plt.show()
	# 				plt.savefig("fig/experiment{}_posterior_task{}.pdf".format(num, i), transparent = True)
	# 				plt.close()

	# 				fig, ax = plt.subplots(len(timepoints),  sharey=True)
	# 				fig.set_figwidth(4.8)
	# 				fig.set_figheight(4.8 * 2)

	# 				for k, ti in enumerate(timepoints):
	# 					for l in range(3):
	# 						quantile_num = quantile_data.shape[3]

	# 						if ti == 0:
	# 							df = np.arange(quantile_num) * 1.0 / quantile_num
	# 						else:

	# 							df = quantile_data[0, 0, l, :, ti-1]

	# 						# df = (np.expand_dims(df, 0) + np.expand_dims(df, 1))/2
	# 						# df = df.flatten()

	# 						df = (np.expand_dims(df, (0, 1)) + np.expand_dims(df, (1, 2)) + np.expand_dims(df, (0, 2)))/3
	# 						df = df.flatten()

	# 						diff = np.expand_dims(bins, 1) - np.expand_dims(df, 0)
	# 						std = 0.03
	# 						q_data = np.sum(np.exp(-0.5 * diff ** 2 / std ** 2) / (np.sqrt(2 * np.pi) * std), axis = 1) / len(df)
	# 						if l == 0:
	# 							ax[k].plot(bins, q_data, label = "{}".format(l+1), color = palette_alt[l])
	# 						if l == 1:
	# 							ax[k].plot(bins, q_data, label = "{}".format(l+1), linestyle = "dashed", color = palette_alt[l])
	# 						if l == 2:
	# 							ax[k].plot(bins, q_data, label = "{}".format(l+1), linestyle = "dashed", color = palette_alt[l])


	# 					ax[k].title.set_text("Posterior at trial {}".format(ti))
						
	# 				handles, labels = ax[0].get_legend_handles_labels()
					
	# 				fig.legend(handles, labels,  bbox_to_anchor=(1, 0.95), loc = "upper right", frameon=False)
	# 				sns.despine()
	# 				fig.supxlabel("Values", x= 0.625, y = 0.06,  fontsize = "small")
	# 				fig.supylabel("Probability density", x = 0.12, fontsize = "small")
	# 				plt.tight_layout()
					
					
	# 				#plt.title("Averaged accumulated regret over {} trials".format(max_trial))
					
	# 				#plt.show()
	# 				plt.savefig("fig/experiment{}_posterior_task{}_sample.pdf".format(num, i), transparent = True)
	# 				plt.close()




	# 	for i, a in enumerate(agents):

	# 		for j in range(3):
	# 			fig, ax = plt.subplots()
	# 			fig.set_figwidth(4.8 * ratio)
	# 			fig.set_figheight(4.8 * ratio)


	# 			action_data = data["action"][a][2, j, :]
			
	# 			plt.scatter((t+1), action_data + 0.05 * np.random.normal(0, 1, 500), marker = 'x') 
	# 			plt.xlabel("Trials")
	# 			plt.ylabel("Action")
	# 			ax.set_yticks([0, 1])
	# 			sns.despine()
	# 			plt.tight_layout()
	# 			plt.savefig("fig/experiment{}_sample_action_{}_{}.pdf".format(num,a, j), transparent = True)
	# 			plt.close()

	# 			ratio = 0.8
	# 			fig, ax = plt.subplots()
	# 			fig.set_figwidth(4.8 * ratio)
	# 			fig.set_figheight(4.8 * ratio)

	# 			if a != "Thompson Sampling":
	# 				print(data["quantile_data"][a][0].shape)
	# 				quantile_data = data["quantile_data"][a][0][j, 0,  :]
	# 				class_num, quantile_num, max_trial = quantile_data.shape
	# 				quantile_data = quantile_data.reshape( class_num * quantile_num, max_trial)
	# 				#quantile_data[quantile_data > 1] = 1
	# 				plt.xscale('log')

	# 				im = ax.imshow(quantile_data, interpolation='nearest', aspect='auto', cmap = "YlGn_r")
	# 				plt.axvline(5, c = "grey", linewidth = 1, linestyle = "dashed")
	# 				plt.axvline(10, c = "grey", linewidth = 1, linestyle = "dashed")
	# 				im.set_clim(0, 1)
	# 				cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal", pad = 0.2)
	# 				cbar.ax.set_xlabel("Activities")


	# 				# ax.set_xticks(np.arange(0, 2001, 200))
	# 				ax.set_yticks([50, 150])
	# 				ax.set_yticklabels(["Left", "Right"], rotation = 90)
					
	# 				# plt.legend(loc="upper left")
	# 				plt.xlabel("Trial")
	# 				plt.ylabel("Action")
	# 				plt.title("Striatal activities over {} trials".format(max_trial))
	# 				#plt.xscale('log')
	# 				#plt.tight_layout()

	# 				plt.savefig("fig/experiment{}_{}_sample_quantilemap_{}.pdf".format(num, a, j))
	# 				plt.close()

	# if num == 26:
	# 	data = load_dict("experiment{}_data".format(num))
	# 	agents = list(data["action"].keys())
	# 	max_trial = len(data["action"][agents[0]][0])
	# 	t = np.arange(max_trial)

		

	# 	delta = [ data["task"][x][i, 0, 1, 0] - data["task"][x][i, 0, 0, 0] for i, x in enumerate(data["task"].keys())]
	# 	task_num = len(delta)
	# 	label = np.linspace(3, 6, len(delta))
	# 	label = ["{:d}".format(int(x)) for x in label]
	# 	print(delta, label)
	# 	x = np.arange(len(delta))

	# 	ratio = 0.7
	# 	fig, ax = plt.subplots()
	# 	fig.set_figwidth(4.8 * ratio)
	# 	fig.set_figheight(4.8 * ratio)
	# 	width = 0.33
	# 	multiplier = 0

	
	# 	for i, a in enumerate(agents):
	# 		error_bar = np.std(data["regret"][a][:, :, -1], axis = 1) / np.sqrt(data["regret"][a][:, :, -1].shape[1])
	# 		# offset  = width * multiplier
	# 		offset  = width * (multiplier+0.5)
	# 		rects = ax.bar(x + offset, np.mean(data["regret"][a][:, :, -1], axis = 1), width, label = a, yerr = error_bar)
			
	# 		multiplier += 1
	# 		#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a)
	# 		#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		
	# 	for j, i in enumerate(range(len(delta))):
	# 		y_max = np.max([np.mean(data["regret"][a][i, :, -1]) +  np.std(data["regret"][a][i, :, -1]) / np.sqrt(len(data["regret"][a][i, :, -1]))  for a in agents[1:]])
	# 		y_min = np.min([data["regret"][a][i, :, -1] for a in agents[1:]])
			
	# 		ax.annotate("", xy=(j+0.33/2, y_max + 2), xycoords='data',
	#        xytext=(j+0.33 * 3 / 2, y_max + 2), textcoords='data',
	#        arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	#                        connectionstyle="bar,fraction=0.2"))
	# 		#z, p = scipy.stats.kruskal(data["regret"][agents[0]][i, :, -1], data["regret"][agents[1]][i, :, -1])

	# 		z, p = scipy.stats.kruskal(data["regret"][agents[0]][i, :, -1], data["regret"][agents[1]][i, :, -1])
	# 		print(p)
	# 		ax.text(j + 0.33 , y_max + 4, stars(p),horizontalalignment='center',verticalalignment='center', fontsize = 12)
		
	# 	plt.legend(loc="upper right", frameon = False)
	# 	print(x+width)
	# 	ax.set_xticks(x +width, label)
	# 	plt.xlabel("A")
	# 	plt.ylabel("Regret")
	# 	plt.ylim(bottom = 0)
		
	# 	#plt.title("Averaged regret over different numbers of alternatives".format(max_trial))
	# 	sns.despine()
	# 	plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
	# 	plt.close()

	# 	for i, a in enumerate(agents):
	# 		ratio = 0.8
	# 		fig, ax = plt.subplots()
	# 		fig.set_figwidth(4.8 * ratio)
	# 		fig.set_figheight(4.8 * ratio)
			
	# 		trial_num =500
	# 		for i, a in enumerate(agents):
	# 			print(data["action"][a].shape)
	# 			idx = 2
	# 			action_accuracy = np.copy(data["action"][a][idx])
	# 			action_accuracy[action_accuracy != (idx+1)] = 0
	# 			action_accuracy[action_accuracy == (idx+1)] = 1
	# 			choice_prob = np.mean(action_accuracy, axis = 0)[:trial_num]
	# 			ci = np.std(action_accuracy, axis = 0) / np.sqrt(action_accuracy.shape[0])
				
	# 			plt.plot(np.arange(trial_num) + 1, choice_prob, label = a, c = palette[i])
	# 			plt.fill_between(np.arange(trial_num) + 1, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
	# 		plt.xscale('log')
	# 		plt.legend(loc="lower right", frameon=False)
	# 		plt.xlabel("Trial")
	# 		plt.ylabel("Accurate choice probability")
	# 		ax.set_xlim(right = trial_num)
	# 		plt.title("Choice probability over {} trials".format(max_trial))
	# 		sns.despine()
	# 		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
	# 		plt.close()

	# 	for i, a in enumerate(agents):

	# 		idx = np.random.randint(0, data["action"][a].shape[1], 3)
	# 		for j in range(3):
	# 			fig, ax = plt.subplots()
	# 			fig.set_figwidth(4.8 * ratio)
	# 			fig.set_figheight(4.8 * ratio)


	# 			action_data = data["action"][a][1, idx[j], :]
	# 			print(action_data.shape)
	# 			plt.scatter(np.arange(trial_num) + 1, action_data + 1 + 0.05 * np.random.normal(0, 1, 500), marker = 'x') 
	# 			plt.xlabel("Trials")
	# 			plt.ylabel("Action")
	# 			ax.set_yticks([1, 2, 3])
	# 			sns.despine()
	# 			plt.tight_layout()
	# 			plt.savefig("fig/experiment{}_sample_action_{}_{}.pdf".format(num,a, j), transparent = True)
	# 			plt.close()

				

	if num == 34:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		max_trial = len(data["action"][agents[0]][0])
		t = np.arange(max_trial)

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		
		
		for i, a in enumerate(agents):
			accuracy =  np.copy(data["action"][a])

			accuracy[accuracy != 1] = 0
			accuracy[accuracy == 1] = 1
			
			accuracy_data = np.mean(accuracy, axis = 0)
			ci =  np.std(accuracy, axis = 0) / np.sqrt(accuracy.shape[0])
			
			plt.plot(t + 1, accuracy_data, label = a, c = palette[i])
			plt.fill_between(t + 1, accuracy_data + ci, accuracy_data - ci, color = palette[i], alpha = 0.1)
		#plt.xscale('log')
		plt.legend(loc="lower right", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		plt.xscale("log")
		ax.set_xlim(right = max_trial)
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		
		
		for i, a in enumerate(agents):
			
			regret = np.mean(data["regret"][a], axis = 0)
			ci =  np.std(data["regret"][a], axis = 0) / np.sqrt(data["action"][a].shape[0])
			
			plt.plot(t + 1, regret, label = a, c = palette[i])
			plt.fill_between(t + 1, regret + ci, regret - ci, color = palette[i], alpha = 0.1)
		#plt.xscale('log')
		plt.legend(loc="lower right", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("regret")
		plt.xscale("log")
		ax.set_xlim(right = max_trial)
		plt.title("regret over {} trials".format(max_trial))
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
		
		z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
		print("kruskal wallis p value for regret is {}".format(p))
		
		p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
		print(p)
		
		print("regret for {} = {}, sem = {}".format(agents[0], np.mean(regret_data[0]), np.std(regret_data[0]) / np.sqrt(len(regret_data[0]))))
		print("regret for {} = {}, sem = {}".format(agents[1],  np.mean(regret_data[1]), np.std(regret_data[1]) / np.sqrt(len(regret_data[1]))))
		print("regret for {} = {}, sem = {}".format(agents[2],  np.mean(regret_data[2]), np.std(regret_data[2]) / np.sqrt(len(regret_data[2]))))

			
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

		accuracy_data = []
		label_data = []
		for i, a in enumerate(agents):
			accuracy =  np.copy(data["action"][a])

			accuracy[accuracy != 1] = 0
			accuracy[accuracy == 1] = 1
			accuracy = np.mean(accuracy, axis = 1)

			accuracy_data.append(accuracy)
			label_data.append(a)
		ax.boxplot(accuracy_data, sym = '', widths = 0.7, showcaps = False, 
                     vert=True,  # vertical box alignment
                     labels=label_data)  # will be used to label x-ticks
		
		z, p = scipy.stats.kruskal(accuracy_data[0], accuracy_data[1], accuracy_data[2])
		print("kruskal wallis p value for accuracy is {}".format(p))
		
		p = sp.posthoc_dunn(accuracy_data, p_adjust='bonferroni')
		print(p)


		print("accuracy for {} = {}, sem = {}".format(agents[0], np.mean(accuracy_data[0]), np.std(accuracy_data[0]) / np.sqrt(len(accuracy_data[0]))))
		print("accuracy for {} = {}, sem = {}".format(agents[1],  np.mean(accuracy_data[1]), np.std(accuracy_data[1]) / np.sqrt(len(accuracy_data[1]))))
		print("accuracy for {} = {}, sem = {}".format(agents[2],  np.mean(accuracy_data[2]), np.std(accuracy_data[2]) / np.sqrt(len(accuracy_data[2]))))

		
			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_accuracy_box.pdf".format(num), transparent = True)
		plt.close()





	if num == 16 or num == 26:
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
		
		ratio = 0.7
		fig, ax = plt.subplots(layout='constrained')
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.33
		multiplier = 0
		x = np.arange(len(delta))
		print("regret")
		for i, a in enumerate(agents):
			error_bar = np.std(data["regret"][a][:, :, -1], axis = 1) / np.sqrt(data["regret"][a][:, :, -1].shape[1])

			# offset  = width * multiplier
			offset  = width * multiplier + width / 2
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
			y_max = np.max([np.mean(data["regret"][a][i, :, -1]) +  np.std(data["regret"][a][i, :, -1]) / np.sqrt(len(data["regret"][a][i, :, -1]))  for a in agents])
			y_min = np.min([data["regret"][a][i, :, -1] for a in agents])
			
			ax.annotate("", xy=(i+width / 2, y_max + 1), xycoords='data',
	       xytext=(i+width * 1.5, y_max + 1), textcoords='data',
	       arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	                       connectionstyle="bar,fraction=0.2"))
			z, p = scipy.stats.mannwhitneyu(data["regret"][agents[0]][i, :, -1], data["regret"][agents[1]][i, :, -1])
			print( delta[i],p)
			ax.text(i + width, y_max + 2, stars(p),horizontalalignment='center',verticalalignment='center')
			# ax.text(i + 0.375, y_max + 4, stars(p * 2),horizontalalignment='center',verticalalignment='center')
		
		
		plt.legend(loc="upper left", frameon = False)
		plt.xlabel("A")
		plt.ylabel("Regret")
		plt.ylim(bottom = 0)
		ax.set_xticks(x + width, delta)
		#plt.title("Averaged regret over different number of alternatives".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.33
		multiplier = 0

		accuracy = {}
		for i, a in enumerate(agents):
			
			# offset  = width * multiplier
			offset  = width * multiplier + width / 2
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

		print("accuracy")
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
			z, p = scipy.stats.mannwhitneyu(accuracy[agents[0]][i], accuracy[agents[1]][i])
			print(delta[i], p)
			ax.text(i + width, y_max, stars(p),horizontalalignment='center',verticalalignment='center')
		
		# plt.legend(loc="upper right", frameon = False)
		plt.legend(loc="upper left", frameon = False)
		ax.set_xticks(x + width, labels = delta)
		plt.xlabel("\u0394")
		plt.ylabel("Regret")
		if num == 16:
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
				print(data["action"][a].shape)
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
				
				if a != "Thompson Sampling":
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

				if a != "Thompson Sampling":
					print(data["quantile_data"][a][0].shape)
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
	
		
	if num == 12:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		
		max_trial = len(data["action"]["Thompson Sampling"][0])
		t = np.arange(max_trial)

		# ax = plt.figure().gca()
		# ax.yaxis.get_major_locator().set_params(integer=True)
		# ax1 = ax.twinx()

		# for i, a in enumerate(agents):
			
		# 	choice_prob = np.mean(data["choice_prob"][a], axis = 0)
		# 	ax.scatter(t, data["action"][a][1] + 1 + (i-1) * 0.03, label = a,  facecolors='none', edgecolors=c, s = 3)
		# 	ax1.plot(t, choice_prob, c = c1)
		
		# ax.legend(loc = "upper center", bbox_to_anchor=(0.5, -0.05))
		# plt.xlabel("Trial")
		# ax.set_ylabel("Action")
		# ax.set_yticks(np.arange(2)+1)
		# ax1.set_ylabel("Choice Probability")
		# ax1.set_yticks(np.arange(0, 1.1, 0.2))
		# ax1.set_ylim(bottom = -0.2, top = 1.2)
		# plt.title("Action over {} trials".format(max_trial))
		# plt.tight_layout()
		# plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		# plt.close()

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		
		trial_num =200
		for i, a in enumerate(agents):
			print(data["action"][a].shape)
			choice_prob = np.mean(data["action"][a], axis = 0)[:trial_num]
			ci =  np.std(data["action"][a], axis = 0)[:trial_num] / np.sqrt(data["action"][a].shape[0])
			
			plt.plot(np.arange(trial_num) + 1, choice_prob, label = a, c = palette[i])
			plt.fill_between(np.arange(trial_num) + 1, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
		#plt.xscale('log')
		plt.legend(loc="lower right", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		ax.set_xlim(right = trial_num)
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		plt.close()

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



	
		# for i in range(10):
		# 	c = (1 - i * 1.0 / (10-1), 0, i * 1.0 / (10-1) )
		# 	task = data["task"]
			
		# 	plt.plot(t, task[0][i], label = "Action {}".format(i+1), c = c)
		# plt.legend(loc="upper left", ncol=2)
		# plt.ylim(0.4, 0.7)
		# plt.xlabel("Trial")
		# plt.ylabel("Probability of reward")
		# plt.title("Task reward structure")
		# plt.savefig("fig/experiment{}_task.pdf".format(num), transparent = True)
		# plt.close()

		for i in range(2):
			if i == 0:
				c = "m"
			elif i == 1:
				c = "c"
			elif i == 2:
				c = "green"
			else:
				c = "red"
			task = data["task"]
			
			plt.plot(t, task[0][i], label = "Action {}".format(i+1), c = c)
		plt.legend(loc="upper left", frameon=False)
		plt.ylim(0, 1)
		plt.xlabel("Trial")
		plt.ylabel("Probability of reward")
		plt.title("Task reward structure")
		plt.savefig("fig/experiment{}_task.pdf".format(num), transparent = True)
		plt.close()

		for i, a in enumerate(agents):
			if a == "Distributional RPE Model":
				ratio = 0.8
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)
				quantile_data = data["quantile_data"][a]
				class_num, quantile_num, max_trial = quantile_data[0, 0].shape
				quantile_data = np.mean(quantile_data, axis = 0)[0].reshape( class_num * quantile_num, max_trial)
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

				plt.savefig("fig/experiment{}_{}_quantilemap.pdf".format(num, a))
				plt.close()


				quantile_data = data["quantile_data"][a]

				bins = np.linspace(0, 1, 100)

				timepoints = [0, 5, 10]

				fig, ax = plt.subplots(len(timepoints),  sharey=True)
				fig.set_figwidth(5)
				fig.set_figheight(12)


				for i, t in enumerate(timepoints):
					for j in range(2):
						quantile_num = quantile_data.shape[3]

						if t == 0:
							df = np.arange(quantile_num) * 1.0 / quantile_num
						else:

							df = np.mean(quantile_data, axis = 0)[0, j, :, t-1]

						
						# df = (np.expand_dims(df, 0) + np.expand_dims(df, 1))/2
						# df = df.flatten()

						# df = (np.expand_dims(df, (0, 1)) + np.expand_dims(df, (1, 2)) + np.expand_dims(df, (0, 2)))/3
						# df = df.flatten()

						diff = np.expand_dims(bins, 1) - np.expand_dims(df, 0)
						std = 0.03
						q_data = np.sum(np.exp(-0.5 * diff ** 2 / std ** 2) / (np.sqrt(2 * np.pi) * std), axis = 1) / len(df)
						if j == 0:
							ax[i].plot(bins, q_data, label = "Left".format(j+1), color = palette_alt[j])
						if j == 1:
							ax[i].plot(bins, q_data, label = "Right".format(j+1), linestyle = "dashed", color = palette_alt[j])

					ax[i].title.set_text("Posterior at trial {}".format(t))
					
				handles, labels = ax[0].get_legend_handles_labels()
				
				fig.legend(handles, labels,  bbox_to_anchor=(1, 0.95), loc = "upper right", frameon=False)
				sns.despine()
				fig.supxlabel("Values", x= 0.625, y = 0.06,  fontsize = "small")
				fig.supylabel("Probability density", x = 0.12, fontsize = "small")
				plt.tight_layout()
				
				
				#plt.title("Averaged accumulated regret over {} trials".format(max_trial))
				
				#plt.show()
				plt.savefig("fig/experiment{}_posterior.pdf".format(num), transparent = True)
				plt.close()

			




		t = np.arange(max_trial)



		fig, ax = plt.subplots(1, 3)
		fig.set_figwidth(6.4 * 3)
		fig.set_figheight(4.8)
		for i, a in enumerate(agents):
			value = data["value"][a]
			
			
			
			for s in range(1):
				for j in range(2):
					
					
					
					if j == 0:
						value_data = np.mean(value, axis = 0)[s][j]
						ci =  np.std(value[:, s, j], axis = 0) / np.sqrt(value[:, s, j].shape[0])
						ax[i].plot(t+1, np.mean(value, axis = 0)[s][j],  label = "{} Action {}".format(a, j+1), c = palette[i])
						ax[i].fill_between(t + 1, value_data + ci, value_data - ci, color = palette[i], alpha = 0.1)
					else:
						value_data = np.mean(value, axis = 0)[s][j]
						ci =  np.std(value[:, s, j], axis = 0) / np.sqrt(value[:, s, j].shape[0])
						ax[i].plot(t+1,  np.mean(value, axis = 0)[s][j], label = "{} Action {}".format(a, j+1), c = palette[i], linestyle = "dashed")
						ax[i].fill_between(t + 1, value_data + ci, value_data - ci, color = palette[i], alpha = 0.1)

		color_patch = [mlines.Line2D([], [], color=palette[i], label=agents[i]) for i in range(len(agents))]
		line_patch = [mlines.Line2D([], [],color="black", label=i, linestyle = t) for i, t in [("Left", "solid"), ("Right", "dashed")]]

		
		#legend_line = plt.legend(handles = line_patch,  title = "Action", frameon = False, bbox_to_anchor=(0.95, 0.5), loc="upper left")
		#legend_color = plt.legend(handles = color_patch, title = "Model", frameon = False, bbox_to_anchor=(0.95, 0.75), loc="upper left")
		fig.legend(handles = line_patch, bbox_to_anchor=(0.05, 1), loc = "upper left")
		#ax.add_artist(legend_color)
		fig.supxlabel("Trial", y = 0.04, fontsize = "medium")
		fig.supylabel("Estimated value", x = 0.01, fontsize = "medium")
		plt.suptitle("Estimated value over {} trials".format(max_trial), fontsize = "medium")
		sns.despine()
		plt.tight_layout()
		fig.savefig("fig/experiment{}_value.pdf".format(num), transparent = True)
		plt.close()

		for i, a in enumerate(agents):

			for j in range(3):
				fig, ax = plt.subplots()
				fig.set_figwidth(4.8 * ratio)
				fig.set_figheight(4.8 * ratio)


				action_data = data["action"][a][j, :]
			
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
				
				if a != "Thompson Sampling":
					quantile_data = data["quantile_data"][a][j, 0, :]

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

	# if num == 25:
	# 	data = load_dict("experiment{}_data".format(num))
		
	# 	agents = list(data["action"].keys())
	# 	print(agents)
	# 	max_trial = len(data["action"][agents[0]][0])
	# 	t = np.arange(max_trial)

	# 	fig, ax = plt.subplots(layout='constrained')
	# 	fig.set_figwidth(4.8)
	# 	fig.set_figheight(4.8)
	# 	width = 0.5
	# 	multiplier = 0
	# 	x = np.arange(len(agents))


	# 	for i, a in enumerate(agents):
	# 		error_bar =  np.std(data["regret"][a][:, -1]) / np.sqrt(data["regret"][a][:, -1].shape[0])
	# 		ax.bar(x[i] , np.mean(data["regret"][a][:, -1]), width, label = a, yerr = error_bar, color = palette_alt[i])
			

			
	# 	for i in x:
	# 		if i != 1:
	# 			y_max = np.max([np.mean(data["regret"][a][:, -1]) +  np.std(data["regret"][a][:, -1]) / np.sqrt(len(data["regret"][a][:, -1]))  for a in agents[i:i+1]])
	# 			y_min = np.min([data["regret"][a][:, -1] for a in agents[i:i+1]])
	# 			#print(i, y_max)
	# 			#ax.annotate("", xy=(i-1+width, y_max ), xycoords='data',
	# 	       # xytext=(i+width, y_max ), textcoords='data',
	# 	       # arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	# 	       #                 connectionstyle="bar,fraction=0.1"))
	# 			z, p = scipy.stats.mannwhitneyu(data["regret"][agents[i]][:, -1], data["regret"][agents[i+1]][:, -1])
	# 			#print(p)
	# 			#ax.text(i + 0.125, y_max + 4, stars(p * 2),horizontalalignment='center',verticalalignment='center')
			
		
	# 	plt.legend(loc="upper left", frameon = False)
	# 	plt.ylabel("Regret")
	# 	plt.ylim(bottom = 0)
	# 	ax.set_xticks(x, agents)
	# 	#plt.title("Averaged regret over different number of alternatives".format(max_trial))
	# 	sns.despine()
	# 	plt.savefig("fig/experiment{}_acc_regret.pdf".format(num), transparent = True)
	# 	plt.close()

	# if num == 26:
	# 	data = load_dict("experiment{}_data".format(num))
	# 	agents = list(data["action"].keys())
	# 	max_trial = len(data["action"][agents[0]][0])
	# 	t = np.arange(max_trial)

	# 	fig, ax = plt.subplots(layout='constrained')
	# 	fig.set_figwidth(4.8)
	# 	fig.set_figheight(4.8)
	# 	width = 0.5
	# 	multiplier = 0
	# 	x = np.arange(len(agents))


	# 	for i, a in enumerate(agents):
	# 		error_bar =  np.std(data["regret"][a][:, -1]) / np.sqrt(data["regret"][a][:, -1].shape[0])
	# 		ax.bar(x[i] , np.mean(data["regret"][a][:, -1]), width, label = a, yerr = error_bar, color = palette_alt[i])
			

			
	# 	for i in x:
	# 		if i != 1:
	# 			y_max = np.max([np.mean(data["regret"][a][:, -1]) +  np.std(data["regret"][a][:, -1]) / np.sqrt(len(data["regret"][a][:, -1]))  for a in agents[i:i+1]])
	# 			y_min = np.min([data["regret"][a][:, -1] for a in agents[i:i+1]])
	# 			print(i, y_max)
	# 			#ax.annotate("", xy=(i-1+width, y_max ), xycoords='data',
	# 	       # xytext=(i+width, y_max ), textcoords='data',
	# 	       # arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	# 	       #                 connectionstyle="bar,fraction=0.1"))
	# 			z, p = scipy.stats.mannwhitneyu(data["regret"][agents[i]][:, -1], data["regret"][agents[i+1]][:, -1])
	# 			print(p)
	# 			#ax.text(i + 0.125, y_max + 4, stars(p * 2),horizontalalignment='center',verticalalignment='center')
			
		
	# 	plt.legend(loc="upper left", frameon = False)
	# 	plt.ylabel("Regret")
	# 	plt.ylim(bottom = 0)
	# 	ax.set_xticks(x, agents)
	# 	#plt.title("Averaged regret over different number of alternatives".format(max_trial))
	# 	sns.despine()
	# 	plt.savefig("fig/experiment{}_acc_regret.pdf".format(num), transparent = True)
	# 	plt.close()


	if num == 20:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		agents = agents[:-1]
		max_trial = len(data["action"][agents[0]][0])
		t = np.arange(max_trial)
		name = [a[4:] for a in agents]

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(6.4 * ratio)
		fig.set_figheight(4.8 * ratio)
		width = 0.5
		multiplier = 0
		x = np.arange(len(agents))


		for i, a in enumerate(agents):
			error_bar =  np.std(data["regret"][a][:, -1]) / np.sqrt(data["regret"][a][:, -1].shape[0])
			ax.bar(x[i] , np.mean(data["regret"][a][:, -1]), width, label = name[i], yerr = error_bar, color = palette_gradient[i+1])
			

			
		for i in x:
			if i != 0:
				y_max = np.max([np.mean(data["regret"][a][:, -1]) +  np.std(data["regret"][a][:, -1]) / np.sqrt(len(data["regret"][a][:, -1]))  for a in agents[i-1:i+1]])
				y_min = np.min([data["regret"][a][:, -1] for a in agents[i-1:i+1]])
				print(i, y_max)
				z, p = scipy.stats.mannwhitneyu(data["regret"][agents[i-1]][:, -1], data["regret"][agents[i]][:, -1])
				print(p)
				if p * 2 <= 0.05:
					ax.annotate("", xy=(i-1, y_max + 0.1 ), xycoords='data',
		       xytext=(i, y_max + 0.1 ), textcoords='data',
		       arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
		                       connectionstyle="bar,fraction=0.1"))
			
					ax.text(i -0.5, y_max + 1, stars(p * 2),horizontalalignment='center',verticalalignment='center')
			
		
		plt.legend(loc="upper right", frameon = False)
		plt.ylabel("Regret")
		plt.xlabel("M (Ensemble size)")
		plt.ylim(bottom = 0)
		ax.set_xticks(x, labels = name)
		#plt.title("Averaged regret over different number of alternatives".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_acc_regret.pdf".format(num), transparent = True)
		plt.close()

		for a in agents:
			quantile_data = data["quantile_data"][a]

			bins = np.linspace(0, 1, 100)

			timepoints = [0, 10, 50]

			fig, ax = plt.subplots(len(timepoints),  sharey=True)
			fig.set_figwidth(6.4 / 3)
			fig.set_figheight(12 / 3)


			for i, t in enumerate(timepoints):
				for j in range(2):
					quantile_num = quantile_data.shape[3]

					if t == 0:
						df = np.arange(quantile_num) * 1.0 / quantile_num
					else:

						df = np.mean(quantile_data, axis = 0)[0, j, :, t-1]

					
					df = (np.expand_dims(df, 0) + np.expand_dims(df, 1))/2
					df = df.flatten()

					# df = (np.expand_dims(df, (0, 1)) + np.expand_dims(df, (1, 2)) + np.expand_dims(df, (0, 2)))/3
					# df = df.flatten()

					diff = np.expand_dims(bins, 1) - np.expand_dims(df, 0)
					std = 0.01
					q_data = np.sum(np.exp(-0.5 * diff ** 2 / std ** 2) / (np.sqrt(2 * np.pi) * std), axis = 1) / len(df)
					if j == 0:
						ax[i].plot(bins, q_data, label = "Left".format(j+1), color = palette_alt[j])
					if j == 1:
						ax[i].plot(bins, q_data, label = "Right".format(j+1), linestyle = "dashed", color = palette_alt[j])

				ax[i].title.set_text("Posterior at trial {}".format(t))
				
			handles, labels = ax[0].get_legend_handles_labels()
			
			fig.legend(handles, labels,  bbox_to_anchor=(1, 0.95), loc = "upper right", frameon=False)
			sns.despine()
			fig.supxlabel("Values", x= 0.625, y = 0.06,  fontsize = "small")
			fig.supylabel("Probability density", x = 0.12, fontsize = "small")
			plt.tight_layout()
			
			
			#plt.title("Averaged accumulated regret over {} trials".format(max_trial))
			
			#plt.show()
			plt.savefig("fig/experiment{}_{}posterior.pdf".format(num, a))
			plt.close()
		



	if num == 20 or num == 24:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		max_trial = len(data["action"][agents[0]][0])
		t = np.arange(max_trial)

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(6.4 * ratio)
		fig.set_figheight(4.8 * ratio)
		for i, a in enumerate(agents):

			regret = np.mean(data["regret"][a], axis = 0)
			ci =  np.std(data["regret"][a], axis = 0) / np.sqrt(data["regret"][a].shape[0])
			
			plt.plot(t, regret, label = a, c = palette[i])
			plt.fill_between(t, regret + ci, regret - ci, color = palette[i], alpha = 0.1)
		plt.legend(loc="upper left", frameon=False)
			
		plt.legend(loc="upper left")
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()




		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(6.4 * ratio)
		fig.set_figheight(4.8 * ratio)

		for i, a in enumerate(agents):

			
			choice_prob = np.mean(data["choice_prob"][a], axis = 0)
			ci =  np.std(data["choice_prob"][a], axis = 0) / np.sqrt(data["choice_prob"][a].shape[0])
			
			plt.plot(t + 1, choice_prob, label = a, c = palette[i])
			plt.fill_between(t + 1, choice_prob + ci, choice_prob - ci, color = palette[i], alpha = 0.1)
		
		plt.legend(loc="lower right", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accurate choice probability")
		ax.set_xlim(right = 500)
		plt.title("Choice probability over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		plt.close()

	
	
	if num == 19:
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

		print(max_episode)
		
		
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
		z, p = scipy.stats.kruskal(accuracy_data[0], accuracy_data[1], accuracy_data[2])
		print("kruskal wallis p value for accuracy is {}".format(p))
		
		p = sp.posthoc_dunn(accuracy_data, p_adjust='bonferroni')
		print(p)

			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
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
		z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
		print("kruskal wallis p value is {} for regret".format(p))
		p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
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
			print(x.shape)
			print(smooth_switch_data[a].shape)
			res = scipy.stats.permutation_test((x.flatten(), smooth_switch_data[a].flatten()), lambda x, y: scipy.stats.spearmanr(x, y).statistic, n_resamples=100000)
			print("{}: {}, {}".format(a, res.statistic, res.pvalue))
		
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
		z, p = scipy.stats.kruskal(switch_data[0], switch_data[1], switch_data[2])
		print("kruskal wallis p value is {} for switch".format(p))
	
		p = sp.posthoc_dunn(switch_data, p_adjust='bonferroni')
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



	


	if num == 9 or num == 11:
		data = load_dict("experiment{}_data".format(num))
		# print("Before filter: {}".format(len(data["switch"]["HMM Model"])))
		# print(data["switch"]["HMM Model"])
		print(data["switch"]["Thalamocortical Model"])
		
		mask = np.logical_and((data["switch"]["Thalamocortical Model"][:, -1] > -1) , (data["switch"]["Thalamocortical Model"][:, -2] > -1) )
		# mask = np.logical_and(mask, (data["switch"]["HMM Model"][:, -1] < 200) )
		# mask = np.logical_and(mask, (data["switch"]["HMM Model"][:, -2] < 200) )
		# mask = np.logical_and(mask, (data["switch"]["HMM Model"][:, -3] < 200) )
		print("After filter: {}".format(len(data["switch"]["Thalamocortical Model"][mask, :])))


	
		switch = len(data["switch"]["Thalamocortical Model"][0])

		max_trial = len(data["action"]["Thalamocortical Model"][0])
		max_episode = len(data["action"]["Thalamocortical Model"])
		max_trial = 1000
		block_size = 200

		agents = list(data["action"].keys())
		agents = [agents[1], agents[0]]

		switch_data = {}
		switch_bool = {}
		
		for a in agents:
			switch_data[a] = np.zeros((max_episode, switch))
			switch_bool[a] = [[False] * switch  for _ in range(max_episode)] 
			for i in range(max_episode):
				for j, action in enumerate(data["action"][a][i]): 
					if num == 9:
						block_size = 200
						idx = int(j / block_size) - 1

						if idx > 1:
							if not switch_bool[a][i][idx-2]:
								
								switch_data[a][i, idx-2] = block_size
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
									switch_data[a][i][idx] += j+1  - (idx + 1) * block_size
									switch_bool[a][i][idx] = True
									#print(self.switch_time[agent.name][idx])

							else:
								if data["choice_prob"][a][i][j] <= 0.2:
									switch_data[a][i][idx]  += j+1  - (idx + 1) * block_size
									switch_bool[a][i][idx] = True
		if num == 11:
			switch_data = data["switch"]
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
		

			regret = np.mean(data["regret"][a][mask, :], axis = 0)
			ci =  np.std(data["regret"][a][mask, :], axis = 0) / np.sqrt(data["regret"][a][mask, :].shape[0])
			
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
		z, p = scipy.stats.mannwhitneyu(regret_data[0], regret_data[1])
		print("rank sum p value is {} for regret".format(p))
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


		
			
		if num == 11:
			bt = np.concatenate((np.arange(0, 1000, 1000), np.arange(1000, 2001, 200)))
		else:
			bt = np.arange(0, 1001, 200)
	
		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		

		for i, a in enumerate(agents):
				
			choice_prob = np.mean(data["action"][a][mask, :], axis = 0)
			ci =  np.std(data["action"][a][mask, :], axis = 0) / np.sqrt(data["action"][a][mask, :].shape[0])
			
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

				

				if a != "Discounted Thompson Sampling":
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
		z, p = scipy.stats.mannwhitneyu(accuracy_data[0], accuracy_data[1])
		print("Rank sum p value for accuracy is {}".format(p))
		print("accuracy for {} = {}, sem = {}".format(agents[0], np.mean(accuracy_data[0]), np.std(accuracy_data[0]) / np.sqrt(len(accuracy_data[0]))))
		print("accuracy for {} = {}, sem = {}".format(agents[1],  np.mean(accuracy_data[1]), np.std(accuracy_data[1]) / np.sqrt(len(accuracy_data[1]))))
		
			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
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




		fig, ax = plt.subplots(1, 3, sharey = True)
		fig.set_figwidth(4.8 * 3)
		fig.set_figheight(4.8)
		for a in ax:
			a.xaxis.get_major_locator().set_params(integer=True)
		for i, a in enumerate(agents):
			
			switch_data = np.mean(smooth_switch_data[a][mask, :], axis = 0) - 8
		
			ci =  np.std(smooth_switch_data[a][mask, :]-8, axis = 0) / np.sqrt(smooth_switch_data[a][mask, :].shape[0])
			ax[i].plot(st, switch_data,  label = a, c = palette[i])
			ax[i].fill_between(st, switch_data + ci, switch_data - ci, color = palette[i], alpha = 0.1)

			r, p = scipy.stats.pearsonr(st, switch_data)
			ax[i].annotate("r = {0:.3f}\np = {1:.3f}".format(float(r), float(p)), xy = (0.85, 0.85), xycoords = "axes fraction" )
			
			
		
		color_patch = [mlines.Line2D([], [], color=palette[i], label=list(agents)[i]) for i in range(len(agents))]
		
		#legend_line = plt.legend(handles = line_patch,  title = "Action", frameon = False, bbox_to_anchor=(0.95, 0.5), loc="upper left")
		#legend_color = plt.legend(handles = color_patch, title = "Model", frameon = False, bbox_to_anchor=(0.95, 0.75), loc="upper left")
		# fig.legend(handles = line_patch, bbox_to_anchor=(0.05, 1), loc = "upper left")
		#ax.add_artist(legend_color)
		fig.legend(bbox_to_anchor=(0.05, 1),loc = "upper left", frameon = False)
		fig.supxlabel("Block", y = 0.04, fontsize = "medium")
		fig.supylabel("Trials to switch", x = 0.01, fontsize = "medium")
		plt.suptitle("Averaged trials to switch over blocks", fontsize = "medium")
		sns.despine()
		plt.tight_layout()
		fig.savefig("fig/experiment{}_switch_time.pdf".format(num), transparent = True)
		plt.close()


		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)
		ax.yaxis.get_major_locator().set_params(integer=True)
		a = "Thalamocortical Model"
		for i, a in enumerate(agents):
			
			switch_data = np.mean(smooth_switch_data[a][mask, :], axis = 0) - 8

			error_bar =  np.std(smooth_switch_data[a][mask, :], axis = 0) / np.sqrt(smooth_switch_data[a][mask, :].shape[0])
			
			plt.errorbar(st, switch_data, yerr = error_bar, c = palette[i], label = a)
			plt.plot(st, switch_data, label = a, c = palette[i])
			x = np.vstack([np.arange(4) for _ in range(max_episode)])
			print(x.shape)
			print(smooth_switch_data[a].shape)
			res = scipy.stats.permutation_test((x.flatten(), smooth_switch_data[a].flatten()), lambda x, y: scipy.stats.spearmanr(x, y).statistic, n_resamples=100000)
			print("{}: {}, {}".format(a, res.statistic, res.pvalue))
		
		
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
		z, p = scipy.stats.mannwhitneyu(switch_data[0], switch_data[1])
		print("rank sum p value is {} for switch".format(p))
		print("switch for {} = {}, sem = {}".format(agents[0], np.mean(switch_data[0]), np.std(switch_data[0]) / np.sqrt(len(switch_data[0]))))
		print("switch for {} = {}, sem = {}".format(agents[1],  np.mean(switch_data[1]), np.std(switch_data[1]) / np.sqrt(len(switch_data[1]))))

	

			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_switch_box.pdf".format(num), transparent = True)
		plt.close()



		for i, a in enumerate(agents):
			if a == "Thalamocortical Model" or a == "HMM Model":
				if i == 0:
					c = "lawngreen"
				elif i == 1:
					c = "purple"
				elif i == 2:
					c = "green"
				else:
					c = "black"

				evidence = np.mean(data["evidence"][a][mask, :], axis = 0)
				ci =  np.std(data["evidence"][a][mask, :], axis = 0) / np.sqrt(data["evidence"][a][mask, :].shape[0])
				
				
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

			if a == "Thalamocortical Model" or a == "HMM Model":
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

			if a == "Thalamocortical Model" or a == "HMM Model":
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

			if a == "Thalamocortical Model" or a == "HMM Model":
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

		ratio = 0.8
		fig, ax = plt.subplots()
		fig.set_figwidth(4.8 * ratio)
		fig.set_figheight(4.8 * ratio)

		for i, a in enumerate(agents):
			if a == "Thalamocortical Model" or a == "HMM Model":

				width = [1, 3]
				width_patch = [mlines.Line2D([], [], color = "grey",linewidth=width[i], label="Context {}".format(i+1)) for i in range(2)]
				legend_width = fig.legend(handles = width_patch, frameon = False, bbox_to_anchor=(0.18, 1), loc="upper left")
		
				
				for k in range(2):
					for s in range(1):
						for j in range(2):
							value = data["value"][a][mask, :]
							
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
			if a == "Thalamocortical Model" or a == "HMM Model":
				width = [1, 3]
				width_patch = [mlines.Line2D([], [], color = "grey",linewidth=width[i], label="Context {}".format(i+1)) for i in range(2)]
				legend_width = fig.legend(handles = width_patch, frameon = False, bbox_to_anchor=(0.18, 1), loc="upper left")
		
				for k in range(2):
					for s in range(1):
						for j in range(2):
							model = data["model"][a][mask, :][:, :, :, :, 1]
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



	if num == 21:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		max_trial = len(data["action"]["Distributional RPE Model"][0])
		t = np.arange(max_trial)

		ax = plt.figure().gca()
		ax.yaxis.get_major_locator().set_params(integer=True)
		ax1 = ax.twinx()

		for i, a in enumerate(agents):
			
			if i == 0:
				c = "blue"
				c1 = "lightblue"
			
			elif i == 1:
				c = "red"
				c1 = "lightcoral"
				
			elif i == 2:
				c = "green"
				c1 = "lightgreen"
				
			else:
				c = "black"
				c1 = "black"
			choice_prob = np.mean(data["choice_prob"][a], axis = 0)
			ax.scatter(t, data["action"][a][1] + 1 + (i-1) * 0.03, label = a,  facecolors='none', edgecolors=c, s = 3)
			ax1.plot(t, choice_prob, c = c1)
		
		ax.legend(loc = "upper center", bbox_to_anchor=(0.5, -0.05))
		plt.xlabel("Trial")
		ax.set_ylabel("Action")
		ax.set_yticks(np.arange(2)+1)
		ax1.set_ylabel("Choice Probability")
		ax1.set_yticks(np.arange(0, 1.1, 0.2))
		ax1.set_ylim(bottom = -0.2, top = 1.2)
		plt.title("Action over {} trials".format(max_trial))
		plt.tight_layout()
		plt.savefig("fig/experiment{}_action.pdf".format(num), transparent = True)
		plt.close()

		for i, a in enumerate(agents):
			if i == 0:
				c = "blue"
				
			elif i == 1:
				c = "red"
			
			elif i == 2:
				c = "green"

			else:
				c = "black"

			regret = np.mean(data["regret"][a], axis = 0)
			plt.plot(t, regret, label = a, c = c)
		plt.legend(loc="upper left")
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

	
		# for i in range(10):
		# 	c = (1 - i * 1.0 / (10-1), 0, i * 1.0 / (10-1) )
		# 	task = data["task"]
			
		# 	plt.plot(t, task[0][i], label = "Action {}".format(i+1), c = c)
		# plt.legend(loc="upper left", ncol=2)
		# plt.ylim(0.4, 0.7)
		# plt.xlabel("Trial")
		# plt.ylabel("Probability of reward")
		# plt.title("Task reward structure")
		# plt.savefig("fig/experiment{}_task.pdf".format(num), transparent = True)
		# plt.close()
		for i in range(2):
			if i == 0:
				c = "m"
			elif i == 1:
				c = "c"
			elif i == 2:
				c = "green"
			else:
				c = "red"
			task = data["task"]
			
			plt.plot(t, task[0][i], label = "Action {}".format(i+1), c = c)
		plt.legend(loc="upper left")
		plt.ylim(0, 1)
		plt.xlabel("Trial")
		plt.ylabel("Probability of reward")
		plt.title("Task reward structure")
		plt.savefig("fig/experiment{}_task.pdf".format(num), transparent = True)
		plt.close()

		for i, a in enumerate(agents):
			if a == "Distributional RPE Model" or a == "Normative Model":
				ax = plt.gca()
				quantile_data = data["quantile_data"][a]
				class_num, quantile_num, max_trial = quantile_data[0, 0].shape
				quantile_data = np.mean(quantile_data, axis = 0)[0].reshape( class_num * quantile_num, max_trial)
				quantile_data[quantile_data > 1] = 1

				im = ax.imshow(quantile_data)
				cbar = ax.figure.colorbar(im, ax = ax, orientation="horizontal")
				cbar.ax.set_xlabel("Activities")

				# ax.set_xticks(np.arange(0, 2001, 200))
				ax.set_yticks([50, 150])
				ax.set_yticklabels(["1", "2"])
				
				# plt.legend(loc="upper left")
				plt.xlabel("Trial")
				plt.ylabel("[Action]")
				plt.title("Striatal activities over {} trials".format(max_trial))
				plt.tight_layout()
				plt.savefig("fig/experiment{}_{}_quantilemap.pdf".format(num, a))
				plt.close()

	
		for i, a in enumerate(agents):
		
			
			for s in range(1):
				for j in range(2):
					if i == 0:
						c = "blue"
						c1 = "lightblue"
					elif i == 1:
						c = "red"
						c1 = "lightcoral"
					elif i == 2:
						c = "green"
						c1 = "lightgreen"
					else:
						c = "orange"
						c1 = "orange"
					value = np.mean(data["value"][a], axis = 0)
					value[value > 1] = 1 

					if j == 0:
						plt.plot(t, value[s][j],  label = "{} Action {}".format(a, j+1), c = c1)
					else:
						plt.plot(t, value[s][j], label = "{} Action {}".format(a, j+1), c = c)


		plt.legend(loc="upper right")
		plt.xlabel("Trial")
		plt.ylabel("Estimated value")
		plt.title("Estimated value over {} trials".format(max_trial))
		plt.savefig("fig/experiment{}_value.pdf".format(num), transparent = True)
		plt.close()

	if num == 27:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		
		max_trial = len(data["action"]["Full model"][0])
		t = np.arange(max_trial)

		switch = len(data["switch"]["Full model"][0])

		max_trial = len(data["action"]["Full model"][0])
		max_episode = len(data["action"]["Full model"])

		
		
		switch_data = data["switch"]
		smooth_switch_data = {}
		for a in agents:
			smooth_switch_data[a] = np.zeros((max_episode, switch))
			for i in range(max_episode):
				for j in range(switch):
					if j == 0:
						smooth_switch_data[a][i][j] = switch_data[a][i][j]
					smooth_switch_data[a][i][j] = (switch_data[a][i][j] + switch_data[a][i][j-1]) / 2.0

		st = np.arange(len(smooth_switch_data["Full model"][0])) + 1
		ax = plt.figure().gca()
		ax.yaxis.get_major_locator().set_params(integer=True)
		ax.xaxis.get_major_locator().set_params(integer=True)
	
		for i, a in enumerate(agents):
			
			switch_data = np.mean(smooth_switch_data[a], axis = 0) - 8
		
			ci =  np.std(smooth_switch_data[a]-8, axis = 0) / np.sqrt(smooth_switch_data[a].shape[0])
			ax.plot(st, switch_data,  label = a, c = palette[i])
			ax.fill_between(st, switch_data + ci, switch_data - ci, color = palette[i], alpha = 0.1)

		plt.legend(loc="upper right",  frameon = False)
		plt.xlabel("Block")
		plt.ylabel("Trials to switch")
		plt.title("Averaged trials to switch over blocks")
		sns.despine()
		plt.savefig("fig/experiment{}_switch_time.pdf".format(num), transparent = True)
		plt.close()

		
		ax = plt.figure().gca()
		ax.xaxis.get_major_locator().set_params(integer=True)

	
		
		fig, ax = plt.subplots()
		width = 0.25
		multiplier = 0
		x = np.arange(len(agents))

		for i, a in enumerate(agents):
			error_bar =  np.std(data["regret"][a][:,  -1], axis = 0) / np.sqrt(data["regret"][a][:,  -1].shape[0])
			ax.bar(x[i], np.mean(data["regret"][a][:, -1], axis = 0), width, label = a, yerr = error_bar)
		

			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a, alpha = 0.8)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		
		y_max = np.max([np.mean(data["regret"][a][:,  -1]) +  np.std(data["regret"][a][:,  -1], axis = 0) / np.sqrt(data["regret"][a][:,  -1].shape[0])  for a in agents])
		y_min = np.min([np.mean(data["regret"][a][:,  -1]) for a in agents])
			
		ax.annotate("", xy=(0, y_max + 2), xycoords='data',
	       xytext=(1, y_max + 2), textcoords='data',
	       arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	                       connectionstyle="bar,fraction=0.1"))
		z, p = scipy.stats.ttest_ind(data["regret"][agents[0]][:, -1], data["regret"][agents[1]][:, -1])
		print(p)
		ax.text(0.5, y_max + 19, stars(p),horizontalalignment='center',verticalalignment='center')
		
		
		#plt.legend(loc="upper left", frameon = False)
	
		plt.ylabel("Regret")
		plt.ylim(bottom = 0, top = y_max + 30)
		plt.xticks(x, agents)
		plt.title("Averaged regret in the environment with fixed volatility".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

	if num == 28:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		
		max_trial = len(data["action"]["Full model"][0])
		t = np.arange(max_trial)

		switch = len(data["switch"]["Full model"][0])

		max_trial = len(data["action"]["Full model"][0])
		max_episode = len(data["action"]["Full model"])

		
		
		switch_data = data["switch"]
		smooth_switch_data = {}
		for a in agents:
			smooth_switch_data[a] = np.zeros((max_episode, switch))
			for i in range(max_episode):
				for j in range(switch):
					if j == 0:
						smooth_switch_data[a][i][j] = switch_data[a][i][j]
					smooth_switch_data[a][i][j] = (switch_data[a][i][j] + switch_data[a][i][j-1]) / 2.0

		st = np.arange(len(smooth_switch_data["Full model"][0])) + 1
		ax = plt.figure().gca()
		ax.yaxis.get_major_locator().set_params(integer=True)
		ax.xaxis.get_major_locator().set_params(integer=True)
	
		for i, a in enumerate(agents):
			
			switch_data = np.mean(smooth_switch_data[a], axis = 0) - 8
		
			ci =  np.std(smooth_switch_data[a]-8, axis = 0) / np.sqrt(smooth_switch_data[a].shape[0])
			ax.plot(st, switch_data,  label = a, c = palette[i])
			ax.fill_between(st, switch_data + ci, switch_data - ci, color = palette[i], alpha = 0.1)

		plt.legend(loc="upper right",  frameon = False)
		plt.xlabel("Block")
		plt.ylabel("Trials to switch")
		plt.title("Averaged trials to switch over blocks")
		sns.despine()
		plt.savefig("fig/experiment{}_switch_time.pdf".format(num), transparent = True)
		plt.close()

		
		ax = plt.figure().gca()
		ax.xaxis.get_major_locator().set_params(integer=True)
	
		
		fig, ax = plt.subplots()
		width = 0.25
		multiplier = 0
		x = np.arange(len(agents))

		for i, a in enumerate(agents):
			error_bar =  np.std(data["regret"][a][:,  -1], axis = 0) / np.sqrt(data["regret"][a][:,  -1].shape[0])
			ax.bar(x[i], np.mean(data["regret"][a][:, -1], axis = 0), width, label = a, yerr = error_bar)
		

			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a, alpha = 0.8)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		
		y_max = np.max([np.mean(data["regret"][a][:,  -1]) +  np.std(data["regret"][a][:,  -1], axis = 0) / np.sqrt(data["regret"][a][:,  -1].shape[0])  for a in agents])
		y_min = np.min([np.mean(data["regret"][a][:,  -1]) for a in agents])
			
		ax.annotate("", xy=(0, y_max + 2), xycoords='data',
	       xytext=(1, y_max + 2), textcoords='data',
	       arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	                       connectionstyle="bar,fraction=0.1"))
		z, p = scipy.stats.ttest_ind(data["regret"][agents[0]][:, -1], data["regret"][agents[1]][:, -1])
		print(p)
		ax.text(0.5, y_max + 22, stars(p ),horizontalalignment='center',verticalalignment='center')
		
		
		#plt.legend(loc="upper left", frameon = False)
	
		plt.ylabel("Regret")
		plt.ylim(bottom = 0, top = y_max + 30)
		plt.xticks(x, agents)
		plt.title("Averaged regret in the environment with dynamic volatility".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

	if num == 29:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		
		max_trial = len(data["action"]["Full model"][0])
		t = np.arange(max_trial)

		switch = len(data["switch"]["Full model"][0])

		max_trial = len(data["action"]["Full model"][0])
		max_episode = len(data["action"]["Full model"])

		
		
		switch_data = data["switch"]
		print(switch_data)
		smooth_switch_data = {}
		for a in agents:
			smooth_switch_data[a] = np.zeros((max_episode, switch))
			for i in range(max_episode):
				for j in range(switch):
					if j == 0:
						smooth_switch_data[a][i][j] = switch_data[a][i][j]
					smooth_switch_data[a][i][j] = (switch_data[a][i][j] + switch_data[a][i][j-1]) / 2.0

		st = np.arange(len(smooth_switch_data["Full model"][0])) + 1
		ax = plt.figure().gca()
		ax.yaxis.get_major_locator().set_params(integer=True)
		ax.xaxis.get_major_locator().set_params(integer=True)
	
		for i, a in enumerate(agents):
			
			switch_data = np.mean(smooth_switch_data[a], axis = 0) - 8
		
			ci =  np.std(smooth_switch_data[a]-8, axis = 0) / np.sqrt(smooth_switch_data[a].shape[0])
			ax.plot(st, switch_data,  label = a, c = palette[i])
			ax.fill_between(st, switch_data + ci, switch_data - ci, color = palette[i], alpha = 0.1)

		plt.legend(loc="upper right",  frameon = False)
		plt.xlabel("Block")
		plt.ylabel("Trials to switch")
		plt.title("Averaged trials to switch over blocks")
		sns.despine()
		plt.savefig("fig/experiment{}_switch_time.pdf".format(num), transparent = True)
		plt.close()

		
		ax = plt.figure().gca()
		ax.xaxis.get_major_locator().set_params(integer=True)
	
		
		fig, ax = plt.subplots()
		width = 0.25
		multiplier = 0
		x = np.arange(len(agents))

		for i, a in enumerate(agents):
			error_bar =  np.std(data["regret"][a][:,  -1], axis = 0) / np.sqrt(data["regret"][a][:,  -1].shape[0])
			ax.bar(x[i], np.mean(data["regret"][a][:, -1], axis = 0), width, label = a, yerr = error_bar)
		

			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a, alpha = 0.8)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		
		y_max = np.max([np.mean(data["regret"][a][:,  -1]) +  np.std(data["regret"][a][:,  -1], axis = 0) / np.sqrt(data["regret"][a][:,  -1].shape[0])  for a in agents])
		y_min = np.min([np.mean(data["regret"][a][:,  -1]) for a in agents])
			
		ax.annotate("", xy=(0, y_max + 2), xycoords='data',
	       xytext=(1, y_max + 2), textcoords='data',
	       arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	                       connectionstyle="bar,fraction=0.1"))
		z, p = scipy.stats.ttest_ind(data["regret"][agents[0]][:, -1], data["regret"][agents[1]][:, -1])
		print(p)
		ax.text(0.5, y_max + 26, stars(p ),horizontalalignment='center',verticalalignment='center')
		
		
		#plt.legend(loc="upper left", frameon = False)
	
		plt.ylabel("Regret")
		plt.ylim(bottom = 0, top = y_max + 30)
		plt.xticks(x, agents)
		plt.title("Averaged regret in the environment with fixed volatility".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

		
		bt = np.arange(0, 2001, 200)
	
		ax = plt.figure().gca()
		

		for i, a in enumerate(agents):
				
			choice_prob = np.mean(data["choice_prob"][a], axis = 0)
			ci =  np.std(data["choice_prob"][a], axis = 0) / np.sqrt(data["choice_prob"][a].shape[0])
			
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

	if num == 30:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		
		max_trial = len(data["action"]["Full model"][0])
		t = np.arange(max_trial)

		switch = len(data["switch"]["Full model"][0])

		max_trial = len(data["action"]["Full model"][0])
		max_episode = len(data["action"]["Full model"])

		
		
		switch_data = data["switch"]
		smooth_switch_data = {}
		for a in agents:
			smooth_switch_data[a] = np.zeros((max_episode, switch))
			for i in range(max_episode):
				for j in range(switch):
					if j == 0:
						smooth_switch_data[a][i][j] = switch_data[a][i][j]
					smooth_switch_data[a][i][j] = (switch_data[a][i][j] + switch_data[a][i][j-1]) / 2.0

		st = np.arange(len(smooth_switch_data["Full model"][0])) + 1
		ax = plt.figure().gca()
		ax.yaxis.get_major_locator().set_params(integer=True)
		ax.xaxis.get_major_locator().set_params(integer=True)
	
		for i, a in enumerate(agents):
			
			switch_data = np.mean(smooth_switch_data[a], axis = 0) - 8
		
			ci =  np.std(smooth_switch_data[a]-8, axis = 0) / np.sqrt(smooth_switch_data[a].shape[0])
			ax.plot(st, switch_data,  label = a, c = palette[i])
			ax.fill_between(st, switch_data + ci, switch_data - ci, color = palette[i], alpha = 0.1)

		plt.legend(loc="upper right",  frameon = False)
		plt.xlabel("Block")
		plt.ylabel("Trials to switch")
		plt.title("Averaged trials to switch over blocks")
		sns.despine()
		plt.savefig("fig/experiment{}_switch_time.pdf".format(num), transparent = True)
		plt.close()


		
		ax = plt.figure().gca()
		ax.xaxis.get_major_locator().set_params(integer=True)
	
		
		fig, ax = plt.subplots()
		width = 0.25
		multiplier = 0
		x = np.arange(len(agents))

		for i, a in enumerate(agents):
			error_bar =  np.std(data["regret"][a][:,  -1], axis = 0) / np.sqrt(data["regret"][a][:,  -1].shape[0])
			ax.bar(x[i], np.mean(data["regret"][a][:, -1], axis = 0), width, label = a, yerr = error_bar)
		

			#plt.errorbar(delta,np.mean(data["regret"][a][:, :, -1], axis = 1), yerr = error_bar, c = palette[i], label = a, alpha = 0.8)
			#plt.plot(delta, np.mean(data["regret"][a][:, :, -1], axis = 1), label = a, c = palette[i])

		
		y_max = np.max([np.mean(data["regret"][a][:,  -1]) +  np.std(data["regret"][a][:,  -1], axis = 0) / np.sqrt(data["regret"][a][:,  -1].shape[0])  for a in agents])
		y_min = np.min([np.mean(data["regret"][a][:,  -1]) for a in agents])
			
		ax.annotate("", xy=(0, y_max + 2), xycoords='data',
	       xytext=(1, y_max + 2), textcoords='data',
	       arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
	                       connectionstyle="bar,fraction=0.1"))
		#print(data["regret"][agents[0]])
		#z, p = scipy.stats.mannwhitneyu(data["regret"][agents[0]][:, -1], data["regret"][agents[1]][:, -1])
		z, p = scipy.stats.ttest_ind(data["regret"][agents[0]][:, -1], data["regret"][agents[1]][:, -1])
					

		print(p)
		ax.text(0.5, y_max + 24, stars(p ),horizontalalignment='center',verticalalignment='center')
		
		
		#plt.legend(loc="upper left", frameon = False)
	
		plt.ylabel("Regret")
		plt.ylim(bottom = 0, top = y_max + 30)
		plt.xticks(x, agents)
		plt.title("Averaged regret in the environment with dynamic volatility".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_regret.pdf".format(num), transparent = True)
		plt.close()

	if num == 32 or num == 33:
		data = load_dict("experiment{}_data".format(num))
		agents = list(data["action"].keys())
		
	
		switch = len(data["switch"][agents[0]][0])

		max_trial = len(data["action"][agents[0]][0])
		max_episode = len(data["action"][agents[0]])
		print(max_episode)
		block_size = 200

		bt = np.arange(0, 1000, 200)

		
	

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
		if num == 32:
			z, p = scipy.stats.mannwhitneyu(regret_data[0], regret_data[1])
			print("rank sum p value is {} for regret".format(p))
		if num == 33:
			z, p = scipy.stats.kruskal(regret_data[0], regret_data[1], regret_data[2])
			print("kruskal wallis p value is {} for regret".format(p))
			p = sp.posthoc_dunn(regret_data, p_adjust='bonferroni')
			print(p)
			print("regret for {} = {}, sem = {}".format(agents[2], np.mean(regret_data[2]), np.std(regret_data[2]) / np.sqrt(len(regret_data[2]))))
		

		print("regret for {} = {}, sem = {}".format(agents[0], np.mean(regret_data[0]), np.std(regret_data[0]) / np.sqrt(len(regret_data[0]))))
		print("regret for {} = {}, sem = {}".format(agents[1],  np.mean(regret_data[1]), np.std(regret_data[1]) / np.sqrt(len(regret_data[1]))))



			
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
		if num == 32:
			z, p = scipy.stats.mannwhitneyu(accuracy_data[0], accuracy_data[1])
			print("Rank sum p value for accuracy is {}".format(p))

		if num == 33:
			z, p = scipy.stats.kruskal(accuracy_data[0], accuracy_data[1], accuracy_data[2])
			print("kruskal wallis p value for accuracy is {}".format(p))
			print("accuracy for {} = {}, sem = {}".format(agents[2], np.mean(accuracy_data[2]), np.std(accuracy_data[2]) / np.sqrt(len(accuracy_data[2]))))
		
		
			p = sp.posthoc_dunn(accuracy_data, p_adjust='bonferroni')
			print(p)

		print("accuracy for {} = {}, sem = {}".format(agents[0], np.mean(accuracy_data[0]), np.std(accuracy_data[0]) / np.sqrt(len(accuracy_data[0]))))
		print("accuracy for {} = {}, sem = {}".format(agents[1],  np.mean(accuracy_data[1]), np.std(accuracy_data[1]) / np.sqrt(len(accuracy_data[1]))))

		
			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
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

		if num == 33:
			z, p = scipy.stats.kruskal(win_switch_data[0],win_switch_data[1], win_switch_data[2])
			print("kruskal wallis p value is {} for win switch".format(p))
			
			p = sp.posthoc_dunn(win_switch_data, p_adjust='bonferroni')
			print(p)
			print("win switch for {} = {}, sem = {}".format(agents[2], np.mean(win_switch_data[2]), np.std(win_switch_data[2]) / np.sqrt(len(win_switch_data[2]))))
		if num == 32:
			z, p = scipy.stats.mannwhitneyu(win_switch_data[0],win_switch_data[1])

			print("Rank sum p value is {} for win switch".format(p))

		print("win switch for {} = {}, sem = {}".format(agents[0], np.mean(win_switch_data[0]), np.std(win_switch_data[0]) / np.sqrt(len(win_switch_data[0]))))
		print("win switch for {} = {}, sem = {}".format(agents[1],  np.mean(win_switch_data[1]), np.std(win_switch_data[1]) / np.sqrt(len(win_switch_data[1]))))

		if num == 33:
			z, p = scipy.stats.kruskal(win_stay_data[0],win_stay_data[1], win_stay_data[2])
			print("kruskal wallis p value is {} for win stay".format(p))
			
			p = sp.posthoc_dunn(win_stay_data, p_adjust='bonferroni')
			print(p)
			print("win stay for {} = {}, sem = {}".format(agents[2], np.mean(win_stay_data[2]), np.std(win_stay_data[2]) / np.sqrt(len(win_stay_data[2]))))
		if num == 32:
			z, p = scipy.stats.mannwhitneyu(win_stay_data[0],win_stay_data[1])

			print("Rank sum p value is {} for win stay".format(p))

		print("win stay for {} = {}, sem = {}".format(agents[0], np.mean(win_stay_data[0]), np.std(win_stay_data[0]) / np.sqrt(len(win_stay_data[0]))))
		print("win stay for {} = {}, sem = {}".format(agents[1],  np.mean(win_stay_data[1]), np.std(win_stay_data[1]) / np.sqrt(len(win_stay_data[1]))))

		if num == 33:
			z, p = scipy.stats.kruskal(lose_switch_data[0],lose_switch_data[1], lose_switch_data[2])
			print("kruskal wallis p value is {} for lose switch".format(p))
			
			p = sp.posthoc_dunn(lose_switch_data, p_adjust='bonferroni')
			print(p)
			print("lose switch for {} = {}, sem = {}".format(agents[2], np.mean(lose_switch_data[2]), np.std(lose_switch_data[2]) / np.sqrt(len(lose_switch_data[2]))))
		if num == 32:
			z, p = scipy.stats.mannwhitneyu(lose_switch_data[0],lose_switch_data[1])

			print("Rank sum p value is {} for lose switch".format(p))

		print("lose switch for {} = {}, sem = {}".format(agents[0], np.mean(lose_switch_data[0]), np.std(lose_switch_data[0]) / np.sqrt(len(lose_switch_data[0]))))
		print("lose switch for {} = {}, sem = {}".format(agents[1],  np.mean(lose_switch_data[1]), np.std(lose_switch_data[1]) / np.sqrt(len(lose_switch_data[1]))))

		if num == 33:
			z, p = scipy.stats.kruskal(lose_stay_data[0],lose_stay_data[1], lose_stay_data[2])
			print("kruskal wallis p value is {} for lose stay".format(p))
			
			p = sp.posthoc_dunn(lose_stay_data, p_adjust='bonferroni')
			print(p)
			print("lose stay for {} = {}, sem = {}".format(agents[2], np.mean(lose_stay_data[2]), np.std(lose_stay_data[2]) / np.sqrt(len(lose_stay_data[2]))))
		if num == 32:
			z, p = scipy.stats.mannwhitneyu(lose_stay_data[0],lose_stay_data[1])

			print("Rank sum p value is {} for lose stay".format(p))

		print("lose stay for {} = {}, sem = {}".format(agents[0], np.mean(lose_stay_data[0]), np.std(lose_stay_data[0]) / np.sqrt(len(lose_stay_data[0]))))
		print("lose stay for {} = {}, sem = {}".format(agents[1],  np.mean(lose_stay_data[1]), np.std(lose_stay_data[1]) / np.sqrt(len(lose_stay_data[1]))))

		

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
			res = scipy.stats.permutation_test((x.flatten(), smooth_switch_data[a].flatten()), lambda x, y: scipy.stats.spearmanr(x, y).statistic, n_resamples=100000)
			print("{}: {}, {}".format(a, res.statistic, res.pvalue))
		
		
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
		
		if num == 32:
			z, p = scipy.stats.mannwhitneyu(switch_data[0], switch_data[1])
			print("rank sum p value is {} for switch".format(p))
		if num == 33:
			z, p = scipy.stats.kruskal(switch_data[0], switch_data[1], switch_data[2])
			print("kruskal wallis p value is {} for switch".format(p))
		
			p = sp.posthoc_dunn(switch_data, p_adjust='bonferroni')
			print(p)

			print("switch for {} = {}, sem = {}".format(agents[2], np.mean(switch_data[2]), np.std(switch_data[2]) / np.sqrt(len(switch_data[2]))))
		


		print("switch for {} = {}, sem = {}".format(agents[0], np.mean(switch_data[0]), np.std(switch_data[0]) / np.sqrt(len(switch_data[0]))))
		print("switch for {} = {}, sem = {}".format(agents[1],  np.mean(switch_data[1]), np.std(switch_data[1]) / np.sqrt(len(switch_data[1]))))

	

			
		plt.legend(loc="upper left", frameon=False)
		plt.xlabel("Trial")
		plt.ylabel("Accumulated regret")
		plt.title("Averaged accumulated regret over {} trials".format(max_trial))
		sns.despine()
		plt.savefig("fig/experiment{}_switch_box.pdf".format(num), transparent = True)
		plt.close()




		if num == 33:
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
			plt.ylim(bottom = -0.05, top = 1)
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
			








