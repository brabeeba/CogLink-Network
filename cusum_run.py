import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from util import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
prob = [[0.7, 0.3], [0.3, 0.7]]

a = 1
tau = 2
s = 1.0 / (2 * tau)
context_num = 2
class_num = 2
dt = 0.005
thalamus = np.zeros(context_num)
thalamus[0] = 2 * a * tau
d_interval = 1000

cusum = np.zeros(200)
diff = np.zeros(200)
diff[0] = thalamus[1] - thalamus[0]

def f_scalar(x):
	if x >= a:
		return 2 *a
	elif x <= -a:
		return 0
	else:
		return x + a

def f(x):
	return np.vectorize(f_scalar)(x)
	

def g(x):
	return np.log(x)

ct = np.array([[[0.3, 0.7], [0.7, 0.3]], [[0.7, 0.3], [0.3, 0.7]]])

tt = - s * np.ones((context_num, context_num))
for i in range(0, context_num):
	tt[i, i] = s 


for trial in range(199):
	if trial < 100:
		context = 0
	else:
		context = 1

	action = np.random.randint(2)
	r = np.random.binomial(1, prob[context][action])
	
	cusum[trial+1] = np.max((0, cusum[trial] + np.log(ct[1][action][r]) - np.log(ct[0][action][r])))
	
	for i in range(d_interval):
		thalamus += dt * 0.2 * (- 2* s * thalamus + f(tt.dot(thalamus)) + g(ct[:, action, r]))
	
	diff[trial+1] = thalamus[1] - thalamus[0]
ratio = 0.8
fig, ax = plt.subplots()
fig.set_figwidth(4.8 * ratio)
fig.set_figheight(4.8 * ratio)
plt.plot(np.arange(1100, 1300), cusum - 2 * a * tau, label = "CUSUM")
plt.plot(np.arange(1100, 1300), diff  , label = "Thalamus", c = "lawngreen")
plt.xlabel("Trials")
plt.axvline(1200, c = "grey", linewidth = 1)
plt.axhline(0, c = "red", linewidth = 1)
sns.despine()
plt.legend(frameon = False)
plt.savefig("fig/experiment_cusum.pdf", transparent = True)
plt.close()










			