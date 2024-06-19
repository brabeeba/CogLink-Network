import numpy as np
import logging
import pickle

logger = logging.getLogger('Bandit')

def model_parameter(opt, new_opt):
	opt = copy.deepcopy(opt)
	logger.info('Model parameter is the following:')
	for k, v in new_opt.items():
		opt[k] = v
		logger.info('{}: {}'.format(k, v))

	return opt

def relu(x):
    return np.maximum(0, x)


def shift_matrix(n):
	return np.vstack([np.zeros(n), np.identity(n)[:-1, :]])

def one_hot(k, n):
	return np.identity(n)[k]


def sparsemask(n, m, ratio):
	return np.random.binomial(1, ratio, (n, m))



def save_dict(state_dict, filename):
	with open(filename, 'wb') as handle:
		pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print("{} is saved.".format(filename))

def load_dict(filename):
	with open(filename, 'rb') as handle:
		state_dict = pickle.load(handle)
	return state_dict

def guassian_convolve(inputs, std):

	def convolve(x):

		def conv(mu):
			np.exp(-0.5 * (x-mu) ** 2 / std ** 2) / (np.sqrt(2 * np.pi) * std)
		np.vectorize(conv)(inputs)



# def pad_cik(original, length):
# 	pad_number = length - len(original)
# 	return ''.join(['0' for _ in xrange(pad_number)]) + original

# def unicode2str(ucode):
# 	return unicodedata.normalize('NFKD', ucode).encode('ascii','ignore')

# def update_opt(old_opt, new_opt):
# 	opt = copy.deepcopy(old_opt)
# 	for k, v in new_opt.items():
# 		opt[k] = v
# 	return opt

# def normalize_text(text):
# 	return unicodedata.normalize('NFKD', text)

# def load_embedding(opt, dictionary):
# 	print "Dictionary size is", len(dictionary)
# 	embeddings = torch.Tensor(len(dictionary), opt['embedding_size'])
# 	embeddings.normal_(0, 1)

# 	with open(opt['embedding_file']) as f:
# 		for line in f:
# 			parsed = line.rstrip().split(' ')
# 			assert(len(parsed) == opt['embedding_size'] + 1)
# 			w = normalize_text(parsed[0].decode('utf-8'))
# 			if w in dictionary.tok2ind:
# 				vec = torch.Tensor([float(i) for i in parsed[1:]])
# 				embeddings[dictionary.tok2ind[w]].copy_(vec)
# 		embeddings[len(dictionary.tok2ind)].fill_(0)

# 	return embeddings

# def summary_network(net, logger):
# 	result = 0
# 	for p in net.parameters():
# 		if p.requires_grad:
# 			weight = reduce(lambda x, y: x * y, p.size())
# 			result += weight
		
# 	logger.info("The number of parameter in the network is {}".format(result))




# def save(model, best_validate, step, optimizer, best = False):
# 	param = {
# 	'state_dict': model.state_dict(),
# 	'opt': model.opt,
# 	'best_validate': best_validate,
# 	'step': step,
# 	'optimizer': optimizer.state_dict()
# 	}
# 	if best:
# 		torch.save(param, model.opt["best_save_file"])
# 	else:
# 		torch.save(param, model.opt['save_file'])


# class AverageMeter(object):
# 	"""Computes and stores the average and current value."""
# 	def __init__(self):
# 		self.reset()

# 	def reset(self):
# 		self.val = 0
# 		self.avg = 0
# 		self.sum = 0
# 		self.count = 0

# 	def update(self, val, n=1):
# 		self.val = val
# 		self.sum += val * n
# 		self.count += n
# 		self.avg = self.sum / (self.count + 1e-8)

# class ReduceLROnPlateau(object):

# 	def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
# 				 verbose=False, threshold=1e-4, threshold_mode='rel',
# 				 cooldown=0, min_lr=0, eps=1e-8):

# 		if factor >= 1.0:
# 			raise ValueError('Factor should be < 1.0.')
# 		self.factor = factor

# 		if not isinstance(optimizer, Optimizer):
# 			raise TypeError('{} is not an Optimizer'.format(
# 				type(optimizer).__name__))
# 		self.optimizer = optimizer

# 		if isinstance(min_lr, list) or isinstance(min_lr, tuple):
# 			if len(min_lr) != len(optimizer.param_groups):
# 				raise ValueError("expected {} min_lrs, got {}".format(
# 					len(optimizer.param_groups), len(min_lr)))
# 			self.min_lrs = list(min_lr)
# 		else:
# 			self.min_lrs = [min_lr] * len(optimizer.param_groups)

# 		self.patience = patience
# 		self.verbose = verbose
# 		self.cooldown = cooldown
# 		self.cooldown_counter = 0
# 		self.mode = mode
# 		self.threshold = threshold
# 		self.threshold_mode = threshold_mode
# 		self.best = None
# 		self.num_bad_epochs = None
# 		self.mode_worse = None  # the worse value for the chosen mode
# 		self.is_better = None
# 		self.eps = eps
# 		self.last_epoch = -1
# 		self._init_is_better(mode=mode, threshold=threshold,
# 							 threshold_mode=threshold_mode)
# 		self._reset()

# 	def _reset(self):
# 		"""Resets num_bad_epochs counter and cooldown counter."""
# 		self.best = self.mode_worse
# 		self.cooldown_counter = 0
# 		self.num_bad_epochs = 0

# 	def step(self, metrics, epoch=None):
# 		current = metrics
# 		if epoch is None:
# 			epoch = self.last_epoch = self.last_epoch + 1
# 		self.last_epoch = epoch

# 		if self.is_better(current, self.best):
# 			self.best = current
# 			self.num_bad_epochs = 0
# 		else:
# 			self.num_bad_epochs += 1

# 		if self.in_cooldown:
# 			self.cooldown_counter -= 1
# 			self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

# 		if self.num_bad_epochs > self.patience:
# 			self._reduce_lr(epoch)
# 			self.cooldown_counter = self.cooldown
# 			self.num_bad_epochs = 0

# 	def _reduce_lr(self, epoch):
# 		for i, param_group in enumerate(self.optimizer.param_groups):
# 			old_lr = float(param_group['lr'])
# 			new_lr = max(old_lr * self.factor, self.min_lrs[i])
# 			if old_lr - new_lr > self.eps:
# 				param_group['lr'] = new_lr
# 				if self.verbose:
# 					print('Epoch {:5d}: reducing learning rate'
# 						  ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

# 	@property
# 	def in_cooldown(self):
# 		return self.cooldown_counter > 0

# 	def _init_is_better(self, mode, threshold, threshold_mode):
# 		if mode not in {'min', 'max'}:
# 			raise ValueError('mode ' + mode + ' is unknown!')
# 		if threshold_mode not in {'rel', 'abs'}:
# 			raise ValueError('threshold mode ' + mode + ' is unknown!')
# 		if mode == 'min' and threshold_mode == 'rel':
# 			rel_epsilon = 1. - threshold
# 			self.is_better = lambda a, best: a < best * rel_epsilon
# 			self.mode_worse = float('Inf')
# 		elif mode == 'min' and threshold_mode == 'abs':
# 			self.is_better = lambda a, best: a < best - threshold
# 			self.mode_worse = float('Inf')
# 		elif mode == 'max' and threshold_mode == 'rel':
# 			rel_epsilon = threshold + 1.
# 			self.is_better = lambda a, best: a > best * rel_epsilon
# 			self.mode_worse = -float('Inf')
# 		else:  # mode == 'max' and epsilon_mode == 'abs':
# 			self.is_better = lambda a, best: a > best + threshold
# 			self.mode_worse = -float('Inf')






		
