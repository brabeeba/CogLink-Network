import numpy as np
from util import relu, shift_matrix, one_hot, sparsemask
import matplotlib.pyplot as plt


class PFC(object):
	"""docstring for PFC"""
	def __init__(self, opt, name = "PFC"):
		super(PFC, self).__init__()
		self.input_size = opt["input_size"]
		self.name = name
		self.opt = opt
		self.mask = sparsemask(input_size, input_size, opt["ratio"])
		self.weights = np.random.normal((input_size, input_size)) * input_size / opt["ratio"]
		if opt["nonlinearity"] == "relu":
			self.f = lambda x: relu(x) + 1e-3
		if opt["nonlinearity"] == "tanh":
			self.f = lambda x: np.tanh(x) + 1
		self.c = np.zeros((input_size, input_size))
		self.c_decay = opt["c_decay"]
		self.dt = opt["dt"]
		self.c_gain = opt["c_gain"]

	def histogram(self):
		hist = {
		self.name + "/w/weights": self.weights,
		self.name + "/a/c": self.c
		}
		return hist
	def scalars(self):
		return {}

	def forward(self, x):
		self.c += self.dt * self.c_decay * (-self.c + self.c_gain * self.f(self.weights.dot(self.c) + x))
		return self.c

class MD(object):
	"""docstring for MD"""
	def __init__(self, opt, name = "MD"):
		super(MD, self).__init__()
		self.opt = opt
		self.core_size = opt["core_size"]
		self.matrix_size = opt["matrix_size"]
		self.a = np.zeros(core_size + matrix_size)
		self.name = name

		self.t_decay = opt["t_decay"]
		self.dt = opt["dt"]
		self.t_gain = opt["t_gain"]

	def histogram(self):
		hist = {
		self.name + "/a/core": self.a[:self.core_size],
		self.name + "/a/matrix": self.a[self.core_size:],
		}
		return hist

	def scalars(self):
		return {}

	def forward(self, x):
		self.a += self.dt * self.t_decay * (-self.a + self.t_gain * relu(x))
		return self.a

class Linear(object):
	"""docstring for Linear"""
	def __init__(self, arg):
		super(Linear, self).__init__()
		self.arg = arg
		


		




class Sequential(object):
	"""docstring for Sequential"""
	def __init__(self, input_size, time_size, opt, name = "cortex"):
		super(Sequential, self).__init__()
		self.input_size = input_size
		self.time_size = time_size
		
		self.c = np.zeros((input_size, time_size))
		self.cc = shift_matrix(time_size)
		self.c_decay = opt["c_decay"]
		self.dt = opt["dt"]
		self.name = name

	def histogram(self):
		hist = {
		self.name + "/a/c": self.c,
		self.name + "/w/cc": self.cc
		}
		return hist

	def scalars(self):
		return {}
		

	def forward(self, x):
		self.c += self.dt * self.c_decay * (-self.c  + self.cc.dot(self.c.T).T)
		self.c[:, 0] += self.dt * self.c_decay * x
		self.c = relu(self.c)
		
		return self.c.flatten()

	def plot(self):
		plt.plot(np.arange(self.time_size), self.c[0])
		plt.show()

class Cortex(object):
	"""docstring for Cortex"""
	def __init__(self, input_size, opt, name = "cortex"):
		super(Cortex, self).__init__()
		self.input_size = input_size
		self.time_size = time_size
		
		self.c = np.zeros((input_size, time_size))
		self.cc = shift_matrix(time_size)
		self.c_decay = opt["c_decay"]
		self.dt = opt["dt"]
		self.name = name

	def histogram(self):
		hist = {
		self.name + "/a/c": self.c,
		self.name + "/w/cc": self.cc
		}
		return hist

	def scalars(self):
		return {}
		

	def forward(self, x):
		self.c += self.dt * self.c_decay * (-self.c  + self.cc.dot(self.c.T).T)
		self.c[:, 0] += self.dt * self.c_decay * x
		self.c = relu(self.c)
		
		return self.c.flatten()

	def plot(self):
		plt.plot(np.arange(self.time_size), self.c[0])
		plt.show()





		

class EasyBG(object):
	"""docstring for EasyBG"""
	def __init__(self, input_size, output_size, opt, name = "BG"):
		super(EasyBG, self).__init__()
		self.name = name
		self.input_size = input_size
		self.output_size = output_size
		self.cp = np.random.rand(input_size, output_size) / np.sqrt(input_size)

		self.cv =  np.random.rand(input_size)  / np.sqrt(input_size)
		self.pp = np.ones((output_size, output_size)) / np.sqrt(output_size)

		for i in range(0, output_size):
			self.pp[i, i] = 0

		self.p = np.zeros(output_size)

		self.p_decay = opt["p_decay"]
		self.v_decay = opt["v_decay"]
		self.da_decay = opt["da_decay"]
		self.noise = opt["noise"]

		self.dt = opt["dt"]

		self.ep_decay = opt["ep_decay"]
		self.ev_decay = opt["ev_decay"]
		self.cs_lr_p = opt["cs_lr_p"]
		self.cs_lr_n = opt["cs_lr_n"]
		self.gamma = opt["gamma"]
		self.v = 0
		self.old_v = 0
		self.da = 0
		self.ep = np.zeros((input_size, output_size))

		self.ev = np.zeros((input_size))
		self.old_r = 0

	def histogram(self):
		hist = {
		self.name + "/a/p": self.p,
		self.name + "/w/cp": self.cp,
		self.name + "/w/cv": self.cv,
		self.name + "/e/ep": self.ep,
		self.name + "/e/ev": self.ev,
		self.name + "/w/pp": self.pp
		}
		return hist

	def scalars(self):
		return {self.name + "/da": self.da, self.name + "/v": self.v}

	def forward(self, x):
		self.p += self.dt * self.p_decay * (-self.p + self.cp.T.dot(x) - self.pp.dot(self.p) + self.noise * np.random.normal(self.output_size))
		self.p = relu(self.p)

		self.old_v = self.v
		self.input = self.cv.T.dot(x)
		self.v += self.dt * self.v_decay *  (-self.v  + self.cv.T.dot(x))
		self.v = relu(self.v)

		self.old_ep = self.ep.copy()
		self.old_ev = self.ev.copy()
		
		#mask = one_hot(np.argmax(x), x.shape[0])
		
		self.ev += self.dt * self.ev_decay * (-self.ev + x)
		self.ep = np.outer(self.ev, self.p)
	
		return self.p

	def update(self, r):
		self.da = self.old_r / 20  + self.gamma * self.v -self.old_v
		if self.da >= 0:
			lr = self.cs_lr_p
		else:
			lr = self.cs_lr_n
		self.cp +=  10 *  lr * self.dt * (self.da * self.old_ep )
		self.cv += lr * self.dt * self.da * self.old_ev
		self.old_r = r
		self.cp /= np.linalg.norm(self.cp, 2, 0)
		# threshold = 0.5
		# self.cp[self.cp > threshold] = threshold
		# self.cp[self.cp < -1 * threshold] = -1 * threshold
		# self.cv[self.cv > threshold] = threshold
		# self.cv[self.cv < -1 * threshold] = -1 * threshold
		

	def plot(self):
		print("ev:", self.ev)
		print("da:", self.da)
		print("v:", self.v)
		print("input:", self.input)
		plt.plot(np.arange(self.input_size), self.cp[:, 0])
		plt.show()
		print("1")
		plt.plot(np.arange(self.input_size), self.cp[:, 1])
		plt.show()

		



class BasalGanglia(object):
	"""docstring for BasalGanglia"""
	def __init__(self, input_size, msn_size, output_size, opt, name = "BG"):
		super(BasalGanglia, self).__init__()
		self.name = name
		self.msn_size = msn_size
		self.output_size = output_size
		self.cs = np.random.rand(input_size, msn_size) / np.sqrt(input_size)
		self.sp = np.random.rand(msn_size, output_size) / np.sqrt(msn_size)
		self.cp = np.random.rand(input_size, output_size) 
		self.pp = np.random.rand(output_size, output_size) / np.sqrt(output_size)

		self.s = np.zeros(msn_size)
		self.p = np.zeros(output_size)

		self.s_decay = opt["s_decay"]
		self.p_decay = opt["p_decay"]
		self.v_decay = opt["v_decay"]
		self.da_decay = opt["da_decay"]
		self.noise = opt["noise"]

		self.dt = opt["dt"]

		self.ep_decay = opt["ep_decay"]
		self.ec_decay = opt["ec_decay"]
		self.cs_lr_p = opt["cs_lr_p"]
		self.cs_lr_n = opt["cs_lr_n"]
		self.gamma = opt["gamma"]
		self.v = 0
		self.da = 0
		self.ep = np.zeros((input_size, msn_size))

		self.ec = np.zeros((input_size, output_size))

	def histogram(self):
		hist = {
		self.name + "/a/s": self.s,
		self.name + "/a/p": self.p,
		self.name + "/w/cs": self.cs,
		self.name + "/w/sp": self.sp,
		self.name + "/w/cp": self.cp,
		self.name + "/w/pp": self.pp,
		self.name + "/a/ep": self.ep
		}
		return hist

	def scalars(self):
		return {self.name + "/da": self.da, self.name + "/v": self.v}

	def forward(self, x):
		self.s += self.dt * self.s_decay * (-self.s  + self.cs.T.dot(x))
		self.s = relu(self.s)
		self.p += self.dt * self.p_decay * (-self.p - self.sp.T.dot(self.s) + self.cp.T.dot(x) - self.pp.dot(self.p) + self.noise * np.random.normal(self.output_size))
		self.p = relu(self.p)
		self.ep = self.dt * self.ep_decay * (-self.ep  + np.outer(x , self.sp.dot(self.p)))
		self.ec = self.dt * self.ec_decay * (-self.ec  + np.outer(x, self.p))

		return self.p

	def update(self, r):
		new_v = self.v + self.dt * self.v_decay *  (-self.v  + np.sum(self.p))
		self.da += self.dt * self.da_decay * (-self.da  + r + self.gamma * new_v - self.v)
		self.v = new_v
		if self.da >= 0:
			lr = self.cs_lr_p
		else:
			lr = self.cs_lr_n
		self.cs -= lr * self.dt * self.da * self.ep
		self.cs = relu(self.cs)
		self.cp += lr * self.dt * self.da * self.ec
		self.cp = relu(self.cp)


# opt = {}
# opt["c_decay"] = 1
# opt["s_decay"] = 1
# opt["p_decay"] = 1
# opt["ep_decay"] = 1
# opt["dt"] = 1e-2
# bg = BasalGanglia(4, 8, 4, opt)

# for x in range(0, 20000):
# 	output = bg.forward(np.array([1, 0, 0, 0]))
# 	if x % 100 == 0:
# 		print(output)


		
		


# class SelfAtt(nn.Module):
# 	"""docstring for SelfAtt"""
# 	def __init__(self, input_size, keepdim = False):
# 		super(SelfAtt, self).__init__()
# 		self.linear = nn.Linear(input_size, 1)
# 		self.keepdim = keepdim

# 	def forward(self, x):
# 		x_flat = x.view(-1, x.size(-1))
# 		score = self.linear(x_flat).view(x.size(0), x.size(1))
# 		score = F.softmax(score)
# 		if self.keepdim:
# 			return x * score.unsqueeze(2)
# 		else: 
# 			return score.unsqueeze(1).bmm(x).squeeze(1)

# class SplitModule(nn.Module):
# 	"""docstring for SplitModule"""
# 	def __init__(self, split_list, dim = -1):
# 		super(SplitModule, self).__init__()
# 		self.split_list = split_list
# 		self.dim = dim

# 	def forward(self, x):
# 		result = []
# 		start_idx = 0
# 		end_idx = 0
# 		for dim in self.split_list:
# 			end_idx += dim
# 			result.append(x.narrow(int(self.dim), start_idx, dim).contiguous())
# 			start_idx += dim
# 		return result


# class SelfMultiAtt(nn.Module):
# 	"""docstring for SelfMultiAtt"""
# 	def __init__(self, input_size, project_list):
# 		super(SelfMultiAtt, self).__init__()
# 		self.projection = nn.Linear(input_size, sum(project_list))
# 		self.split = SplitModule(project_list)
# 		self.attentions = nn.ModuleList([SelfAtt(dim) for dim in project_list])

# 	def forward(self, x):
# 		x_flat = x.view(-1, x.size(-1))
# 		current_input = self.projection(x_flat)
# 		current_input = current_input.view(x.size(0), -1, current_input.size(-1))

# 		current_input = self.split(current_input)
# 		current_input = map(lambda x: x[0](x[1]), zip(self.attentions, current_input))
# 		result = torch.cat(current_input, dim = -1)
# 		return result

		

# class DilatedConv(nn.Module):
# 	"""docstring for DilatedConv"""
# 	def __init__(self, dilation, input_channel, output_channel, kernel_size, groups = 1, padleft = True):
# 		super(DilatedConv, self).__init__()
# 		self.dilation = dilation
# 		self.conv = nn.Conv1d(input_channel, output_channel, kernel_size, dilation = dilation, groups = groups)
# 		self.padleft = padleft

# 	def forward(self, x):
# 		length = x.size(-1)
# 		kernel_size = self.conv.weight.size(-1)
# 		round_dilation = length - (length + self.dilation - 1) % self.dilation + self.dilation - 1
# 		dilated_length = round_dilation / self.dilation
# 		total_pad = round_dilation - length + (kernel_size - 1) * self.dilation
		
# 		if self.padleft:
# 			pad = (0, 0, total_pad, 0)
# 		else:
# 			pad = (0, 0, 0, total_pad)
# 		x = x.unsqueeze(3)
# 		x = F.pad(x, pad)
# 		x = x.squeeze(3)
# 		conv = self.conv(x)

# 		return conv[:, :, :length]

# class FancySelfAtt(nn.Module):
# 	"""docstring for FancySelfAtt"""
# 	def __init__(self, input_channel, key_dim, value_dim):
# 		super(FancySelfAtt, self).__init__()
# 		self.key_dim = key_dim
# 		self.value_dim = value_dim
# 		self.conv = nn.Conv1d(input_channel, key_dim * 2 + value_dim, 1)

# 	def forward(self, x):
# 		current_input = torch.transpose(x, 1, 2)
# 		current_input = self.conv(current_input)
# 		query, key, value = current_input[:, :self.key_dim, :], current_input[:, self.key_dim:self.key_dim *2, :], current_input[:, self.key_dim * 2:, :]
# 		score = F.softmax(query.transpose(1, 2).bmm(key))
# 		current_input = value.bmm(score)

# 		return current_input

# class Linear3d(nn.Module):
# 	"""docstring for Linear3D"""
# 	def __init__(self, input_size, output_size):
# 		super(Linear3d, self).__init__()
# 		self.linear = nn.Linear(input_size, output_size)

# 	def forward(self, x):
# 		batch_size = x.size(0)
# 		x_flat = x.view(-1, x.size(-1))
# 		result = self.linear(x_flat)
# 		return result.view(batch_size, x.size(1), -1) 
		








		
