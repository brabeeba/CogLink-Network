import numpy as np
import logging

logger = logging.getLogger('Bandit')

class DynamicBandit(object):
	"""docstring for DynamicBandit"""
	def __init__(self, opt, name = "DynamicBandit"):
		super(DynamicBandit, self).__init__()
		self.opt = opt
		self.trial = 0
		self.context = 0
		self.context_num = opt["context_num"]
		self.stimuli_num = opt["stimuli_num"]
		self.class_num = opt["class_num"]
		self.stimuli = 0
		self.max_trial = opt["max_trial"]
		self.name = name
		
		self.prob = np.random.rand(self.context_num, self.stimuli_num, self.class_num)
		#self.prob = np.array([[[0.3, 0.4, 0.5, 0.6, 0.7]]])
		#self.prob = np.array([[[0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]]])
		#self.prob = np.array([[[0.3, 0.7]]])
		self.prob = np.array([[[0.7, 0.3]], [[0.3, 0.7]]])
		#self.prob = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]])
	
	def setprob(self, prob):
		self.prob = prob
		for x in range(0, self.context_num):
			print("The bandit task starts with prob {} in context {}".format(self.prob[x], x))


	def stimuli(self):
		return self.stimuli

	def expectation(self, action):
		return self.prob[self.context][self.stimuli][action]

	def step(self, action):
		last = False
		reward = np.random.binomial(1, self.prob[self.context][self.stimuli][action])
		self.trial += 1
		self.stimuli = np.random.randint(0, self.stimuli_num)
		regret = np.max(self.prob[self.context][self.stimuli]) - self.prob[self.context][self.stimuli][action]

		if self.trial % self.opt["block_size"] == 0:
			if self.context_num > 1:
				context = np.random.randint(0, self.context_num - 1)
				if context < self.context:
					self.context = context
				else:
					self.context = context + 1

		if self.trial >= self.opt["max_trial"]:
			logger.info("Reach the last trial at trial {}".format(self.opt["max_trial"]))
			last = True

		return reward, regret, last, self.stimuli

	def reset(self):
		self.trial = 0
		self.context = 0
		self.stimuli = 0
		logging.info("Reset the task")

class UnstructuredBandit(object):
	"""docstring for DynamicBandit"""
	def __init__(self, opt, name = "UnstructuredBandit"):
		super(UnstructuredBandit, self).__init__()
		self.opt = opt
		self.trial = 0
		self.context = 0
		self.context_num = 10
		self.class_num = opt["class_num"]
		self.stimuli_num = opt["stimuli_num"]
		self.stimuli = 0
		self.max_trial = opt["max_trial"]
		self.name = name
		
		self.prob = np.random.rand(self.context_num, self.stimuli_num, self.opt["class_num"])
		#self.prob[:, :, 1] = 1 - self.prob[:, :, 0]
		#self.prob = np.array([[[0.3, 0.4, 0.5, 0.6, 0.7]]])
		#self.prob = np.array([[[0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6]]])
		#self.prob = np.array([[[0.3, 0.7]]])
		#self.prob = np.array([[[0.7, 0.3]], [[0.3, 0.7]]])
		#self.prob = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]])
		for x in range(0, self.context_num):
			print("The bandit task starts with prob {} in context {}".format(self.prob[x], x))

	def stimuli(self):
		return self.stimuli

	def expectation(self, action):
		return self.prob[self.context][self.stimuli][action]

	def step(self, action):
		last = False
		reward = np.random.binomial(1, self.prob[self.context][self.stimuli][action])
		self.trial += 1
		self.stimuli = np.random.randint(0, self.stimuli_num)
		regret = np.max(self.prob[self.context][self.stimuli]) - self.prob[self.context][self.stimuli][action]

		if self.trial % self.opt["block_size"] == 0:
			self.context = self.context + 1

		if self.trial >= self.opt["max_trial"]:
			logger.info("Reach the last trial at trial {}".format(self.opt["max_trial"]))
			last = True

		return reward, regret, last, self.stimuli

	def reset(self):
		self.trial = 0
		self.context = 0
		self.stimuli = 0
		logging.info("Reset the task")

class VolatileBandit(object):
	"""docstring for DynamicBandit"""
	def __init__(self, opt, name = "VolatileBandit"):
		super(VolatileBandit, self).__init__()
		self.opt = opt
		self.trial = 0
		self.context = 0
		self.context_num = opt["context_num"]
		self.stimuli_num = opt["stimuli_num"]
		self.stimuli = 0
		self.max_trial = opt["max_trial"]
		self.name = name
		
		self.prob = np.random.rand(self.context_num, self.stimuli_num, self.opt["class_num"])
		self.prob = np.array([[[0.7, 0.3]], [[0.3, 0.7]]])
		#self.prob = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]])
		for x in range(0, self.context_num):
			print("The bandit task starts with prob {} in context {}".format(self.prob[x], x))

	def stimuli(self):
		return self.stimuli

	def expectation(self, action):
		return self.prob[self.context][self.stimuli][action]

	def step(self, action):
		last = False
		reward = np.random.binomial(1, self.prob[self.context][self.stimuli][action])
		self.trial += 1
		self.stimuli = np.random.randint(0, self.stimuli_num)
		regret = np.max(self.prob[self.context][self.stimuli]) - self.prob[self.context][self.stimuli][action]

		condition1 = (self.trial >= self.max_trial / 2) and (self.trial % self.opt["block_size"] == 0)
		
		if condition1:
		
			if self.context_num > 1:
				context = np.random.randint(0, self.context_num - 1)
				if context < self.context:
					self.context = context
				else:
					self.context = context + 1

		if self.trial >= self.opt["max_trial"]:
			logger.info("Reach the last trial at trial {}".format(self.opt["max_trial"]))
			last = True

		return reward, regret, last, self.stimuli

	def reset(self):
		self.trial = 0
		self.context = 0
		self.stimuli = 0
		logging.info("Reset the task")


class HMMBandit(object):
	"""docstring for HMMBandit"""
	def __init__(self, opt):
		super(HMMBandit, self).__init__()
		self.opt = opt
		self.trial = 0
		self.context = 0
		self.context_num = opt["context_num"]
		self.stimuli_num = opt["stimuli_num"]
		self.stimuli = 0
		self.max_trial = opt["max_trial"]
		
		self.prob = np.random.rand(self.context_num, self.stimuli_num, self.opt["class_num"])
		self.prob = np.array([[[0.7, 0.3]], [[0.3, 0.7]]])
		self.switch_prob = opt["switch_prob"] 

		for x in range(0, self.context_num):
			print("The bandit task starts with prob {} in context {}".format(self.prob[x], x))

	def stimuli(self):
		return self.stimuli

	def expectation(self, action):
		return self.prob[self.context][self.stimuli][action]

	def step(self, action):
		last = False
		reward = np.random.binomial(1, self.prob[self.context][self.stimuli][action])
		self.trial += 1
		self.stimuli = np.random.randint(0, self.stimuli_num)
		regret = np.max(self.prob[self.context][self.stimuli]) - self.prob[self.context][self.stimuli][action]

		if np.random.rand() < self.switch_prob:
			print("switch at trial {}".format(self.trial))
			if self.context_num > 1:
				context = np.random.randint(0, self.context_num - 1)
				if context < self.context:
					self.context = context
				else:
					self.context = context + 1

		if self.trial >= self.opt["max_trial"]:
			logger.info("Reach the last trial at trial {}".format(self.opt["max_trial"]))
			last = True

		return reward, regret, last, self.stimuli

	def reset(self):
		self.trial = 0
		self.context = 0
		self.stimuli = 0
		logging.info("Reset the task")






# class Dictionary10K(object):
# 	"""docstring for Dictionary10K"""
# 	def __init__(self, opt, max_word = 60000):
# 		super(Dictionary10K, self).__init__()
# 		self.opt = opt
# 		self.tok2ind = {}
# 		self.ind2tok = {}
# 		self.freq = defaultdict(int)
# 		self.embedding_word = self.build_embedding()
# 		self.UNK_TOKEN = "<UNK_TOKEN>"
# 		self.max_word = max_word


# 	def build_embedding(self):
# 		embedding = set()
# 		if self.opt['pretrain_word'] and os.path.isfile(self.opt['embedding_file']):
# 			with open(self.opt['embedding_file'], 'r') as f:
# 				for line in f:
# 					w = line.lower().rstrip().split(' ')[0]
# 					w = normalize_text(w.decode('utf-8'))
# 					embedding.add(w)
# 			print "Number of word in embedding is {}".format(len(embedding))
# 			return embedding
# 		return None

# 	def build_dict(self, paths):
# 		print "Building dictionary..."
# 		counter = 0
# 		for path in paths:
# 			if counter % 100 is 0:
# 				print "Building {} out of {}".format(counter, len(paths))
# 			counter += 1
# 			with open(path, 'rb') as f:
# 				doc = json.load(f)
# 				text = doc.setdefault(u"s1", None)
# 				if text is None:
# 					continue
# 				arr = text.lower().rstrip().split(' ')
# 				condition = lambda x: x is not ' ' and x is not '' and '*' not in x and '#' not in x and '|' not in x
# 		 		filtered = filter(condition, arr)
# 		 		text = u' '.join(filtered)

# 				self.add_to_dict(self.tokenize(text))

# 		self.prune_dictionary(self.max_word)

# 		print "Finish building dictionary with vocab size {}".format(len(self.tok2ind))

# 	def prune_dictionary(self, max_word):
# 		if len(self.freq) > max_word:
# 			sort_word = sorted(self.freq, key = self.freq.get)
# 			delete_word = sort_word[:-max_word]
# 			remain_word = sort_word[-max_word:]
# 			for word in delete_word:
# 				self.freq.pop(word, None)
# 			self.tok2ind = dict(zip(remain_word, xrange(len(remain_word))))
# 			self.ind2tok = dict(zip(xrange(len(remain_word)), remain_word))

# 	def tokenize(self, text):
# 		tokens = NLP.tokenizer(text)
# 		return [normalize_text(t.text) for t in tokens]

# 	def span_tokenize(self, text):
# 		tokens = NLP.tokenizer(text)
# 		return [(t.idx, t.idx + len(text)) for t in tokens]

# 	def add_to_dict(self, tokens):

# 		for token in tokens:
# 			token = normalize_text(token)
# 			if self.embedding_word is not None and token not in self.embedding_word:
# 				continue

# 			self.freq[token] += 1

# 			if token not in self.tok2ind:
# 				idx = len(self.tok2ind)
# 				self.tok2ind[token] = idx
# 				self.ind2tok[idx] = token

# 	def __len__(self):
# 		return len(self.tok2ind) + 1


# class Dataset10k(Dataset):
# 	"""docstring for ClassName"""
# 	def __init__(self, opt, batch = True, unique = False, reread = False):
# 		super(Dataset10k, self).__init__()
# 		self.opt = opt
# 		self.training = True
# 		self.batch = batch
# 		self.unique = unique
# 		self.reread = reread
# 		if os.path.isfile(opt['dict_file']):
# 			print 'load dictionary from ' + opt['dict_file']
# 			param = torch.load(opt['dict_file'])
# 			self.dictionary = param['dictionary']
			
# 			if reread:
# 				print 'recreating reader'
# 				self.reader = Reader10K(unique = unique)
# 				self.save()
# 			else:
# 				self.reader = param['reader']
				
# 		else:
# 			print 'creating dictionary'
# 			self.reader = Reader10K(unique = unique)
# 			self.dictionary = Dictionary10K(opt, opt['vocab_size'] - 1)
# 			self.dictionary.build_dict(zip(*zip(*self.reader.all_example)[0])[1])

# 			self.save()

# 	def save(self):
# 		params = {
# 		'reader': self.reader,
# 		'dictionary': self.dictionary
# 		}
# 		torch.save(params, self.opt['dict_file'])
# 		print "Saved Dictionary"
	
# 	def _get_index_mask(self, cik):
# 		label_length = len(self.reader.label2ind)
# 		indices = np.zeros([label_length], dtype = np.float32)
# 		mask = np.ones([label_length], dtype = np.float32)

# 		for i in xrange(label_length):
# 			if cik in self.reader.label2cik[i]:
# 				indices[i] = 1.0
# 			if cik in self.reader.mask2cik[i]:
# 				mask[i] = 0.0

# 		return indices, mask

# 	def _process_document_batch(self, index, batch_size):

# 		b_tokens, b_length, b_label, b_indices, b_mask = [], [], [], [], []

# 		#b_text, b_filer_name, b_cik = [], [], []

# 		pad_length = 0
# 		for i in xrange(batch_size):

# 			(cik, path), label = self.reader.example(mode = self.training)[index * batch_size + i]

# 			with open(path, 'rb') as f:
# 				doc = json.load(f)
# 				text = doc[u"s1"]
# 				cik = doc[u"cik"]
# 				filer_name = doc[u"filer_name"]
# 				arr = text.rstrip().split(' ')
# 				condition = lambda x: x is not ' ' and x is not '' and '*' not in x and '#' not in x and '|' not in x
# 			 	filtered = filter(condition, arr)
# 			 	if len(filtered) < 30:
# 			 		continue

# 			 	sample_length = np.random.randint(self.opt['minword'], self.opt['maxword'])
# 			 	text = u' '.join(filtered)

# 			 	indices, mask = self._get_index_mask(cik)

# 			 	# b_text.append(text)
# 			 	# b_filer_name.append(filer_name)
# 			 	# b_cik.append(cik)

# 			 	tokens = map(lambda x: self.dictionary.tok2ind.get(x, len(self.dictionary.tok2ind)), self.dictionary.tokenize(text))

		
# 			 	start_idx = 0

# 			 	if len(tokens) > sample_length:
# 			 		#if self.training:
# 				 	#	start_idx = np.random.randint(len(tokens) - sample_length)
# 					start_idx = np.random.randint(len(tokens) - sample_length)
					
# 				 	tokens = tokens[start_idx:start_idx + sample_length] 
				 
# 			 	tokens = np.array(tokens, dtype=np.int64)
# 			 	length = len(tokens)

# 			 	if length > pad_length:
# 			 		pad_length = length

# 				b_tokens.append(tokens)
# 				b_length.append(length)
# 				b_label.append(np.array(label, dtype=np.int64))
# 				b_indices.append(indices)
# 				b_mask.append(mask)

# 		result = sorted(zip(b_tokens, b_length, b_label, b_indices, b_mask), key = lambda x: x[1], reverse = True)
# 		b_tokens, b_length, b_label, b_indices, b_mask = zip(*result)

# 		b_tokens = map(lambda x: np.pad(x, (0, pad_length - len(x)), mode = 'constant', constant_values = len(self.dictionary.tok2ind)), b_tokens)

# 		b_tokens = np.vstack(b_tokens)
# 		b_length = np.vstack(b_length)
# 		b_label = np.vstack(b_label)
# 		b_indices = np.vstack(b_indices)
# 		b_mask = np.vstack(b_mask)

# 		# b_text = np.array(b_text)
# 		# b_filer_name = np.array(b_filer_name)
# 		# b_cik = np.array(b_cik)

# 		return b_tokens, b_length, b_label, b_indices, b_mask
# 		#return b_tokens, b_length, b_label, b_indices, b_mask, b_text, b_filer_name, b_cik

# 	def _process_document(self, index, istext = False):

# 		(cik, path), label = self.reader.example(mode = self.training)[index]

# 		with open(path, 'rb') as f:
# 			doc = json.load(f)
# 			text = doc[u"s1"]
# 			cik = doc[u"cik"]
# 			filer_name = doc[u"filer_name"]
# 			arr = text.rstrip().split(' ')
# 			condition = lambda x: x is not ' ' and x is not '' and '*' not in x and '#' not in x and '|' not in x
# 		 	filtered = filter(condition, arr)

# 		 	if len(filtered) >= 500:
# 			 	filtered = filtered[:500]

# 		 	text = u' '.join(filtered)

# 		 	if istext:
# 		 		return text, filer_name

# 		 	tokens = map(lambda x: self.dictionary.tok2ind.get(x, len(self.dictionary.tok2ind)), self.dictionary.tokenize(text))

# 		 	#if self.training:
# 		 	#	start_idx = np.random.randint(100)
# 		 	#else:
# 		 	#	start_idx = 0
# 		 	start_idx = 0

# 		 	max_length = 200
# 		 	length = max_length

# 		 	if len(tokens) > max_length:

# 		 		if len(tokens[start_idx:]) > max_length:
# 			 		tokens = tokens[start_idx:start_idx + max_length] 
			 
# 			 	else:
# 			 		tokens = tokens[-max_length:]
# 		 	else:
# 		 		tokens = tokens + [len(self.dictionary.tok2ind)] * (max_length - len(tokens))
		 		
# 		 	tokens = np.array(tokens, dtype=np.int64)
# 		 	assert len(tokens) > 0, "Tokens needs to have length larger than 0"

# 		 	return tokens, length, np.array(label, dtype=np.int64)

# 	def train(self, mode = True):
# 		self.training = mode

# 	def __len__(self):
# 		if self.batch:
# 			return len(self.reader.example(mode = self.training)) / self.opt['batch_size']
# 		else:
# 			return len(self.reader.example(mode = self.training))

# 	def __getitem__(self, index):
# 		if self.batch:
# 			return self._process_document_batch(index, self.opt['batch_size'])
# 		else:
# 			return self._process_document(index)
