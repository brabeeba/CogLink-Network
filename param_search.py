import torch
import logging
import argparse
import config
import numpy as np
import os
import sys
import util
from torch.utils.data import DataLoader
from reader import Dataset10k
import random
from inference import inference
import itertools
import time
from metric import metric
from tensor_logger import TensorLogger
from hyperband import Experiment
import glob
import copy
import shutil
import json

def init_model(opt, logger, dataset):
	net = inference(opt, opt)
	net.set_embedding(dataset.dictionary)
	best_validate = float('inf')
	i = 0

	parameters = [p for p in net.parameters() if p.requires_grad]
	optimizer = torch.optim.Adam(parameters, opt['lr'], betas= (opt['beta1'], opt['beta2']))
	scheduler = util.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5)

	return net, best_validate, i, optimizer, scheduler

def load_model(opt, logger, dataset):
	logger.info("Load model from {}".format(opt['save_file']))
	params = torch.load(opt['save_file'])
	state_dict = params['state_dict']
	
	net = inference(opt, opt)
	net.set_embedding(dataset.dictionary)
	net.load_state_dict(state_dict)

	best_validate = params['best_validate']
	i = params['step']

	optimizer = params['optimizer']
	scheduler = util.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5)
	scheduler.best = best_validate

	return net, best_validate, i, optimizer, scheduler


def validate(opt, dataset, net):
	dataset.train(mode = False)
	net.train(mode = False)

	dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = opt['num_worker'], pin_memory = False)
	if opt['cuda']:
		net = torch.nn.DataParallel(net)
		net.cuda()
		dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = opt['num_worker'], pin_memory = True)
	
	loss = metric(opt, net, dataloader)
	dataset.train(mode = True)

	return loss

def run_and_eval(param, max_step, opt):

	opt = copy.deepcopy(opt)

	run_id = len(next(os.walk('./param_search/'))[1])
	train_dir = './param_search/run{}'.format(run_id)
	os.makedirs(train_dir)

	opt = util.update_opt(opt, param)
	opt['save_file'] = os.path.join(train_dir, os.path.basename(opt['save_file']))
	opt['best_save_file'] = os.path.join(train_dir, os.path.basename(opt['best_save_file']))
	opt['log_file'] = os.path.join(train_dir, os.path.basename(opt['log_file']))

	logger = logging.getLogger('Classification{}'.format(run_id))
	logger.setLevel(logging.INFO)
	fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
	console = logging.StreamHandler()
	console.setFormatter(fmt)
	logger.addHandler(console)

	
	logfile = logging.FileHandler(opt['log_file'], 'w')
	logfile.setFormatter(fmt)
	logger.addHandler(logfile)

	logger.info("Budget is {}".format(max_step))
	logger.info("Model Training begin with options:")
	for k, v in opt.items():
		logger.info("{} : {}".format(k, v))


	dataset = Dataset10k(opt, batch = True)
	logger.info("There are {} train examples and {} validate example in dataset".format(len(dataset.reader.train_example), len(dataset.reader.validate_example)))
	

	if not opt['new_model'] and os.path.isfile(opt['save_file']):
		net, best_validate, i, optimizer, scheduler = load_model(opt, logger, dataset)
	else:
		net, best_validate, i, optimizer, scheduler = init_model(opt, logger, dataset)


	util.summary_network(net, logger)

	

	dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = opt['num_worker'], pin_memory = False)
	if opt['cuda']:
		net = torch.nn.DataParallel(net)
		net.cuda()
		dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = opt['num_worker'], pin_memory = True)

	cycleloader = itertools.cycle(dataloader)

	parameters = [p for p in net.parameters() if p.requires_grad]

	parameters_dict = {}
	for p in net.named_parameters():
		if p[1].requires_grad:
			parameters_dict[p[0]] = p[1].size()
			parameters_dict[p[0] + '_grad'] = p[1].size()

	summary_writer = TensorLogger(train_dir, parameters_dict, {"loss": (), "lr": ()})

	net.train()
	dataset.train(mode = True)

	patience = 0
	
	while patience < opt['patience'] and i < max_step * opt['unit']:
		net.train()
		dataset.train()
		
		begin = time.time()
		data, length, target = cycleloader.next()
		data = data.squeeze(0)
		length = length.squeeze(0)
		target = target.squeeze(0)

		if opt['cuda']:
			data, target = data.cuda(async = True), target.cuda(async = True)

		data = torch.autograd.Variable(data)
		target = torch.autograd.Variable(target)
		length = torch.autograd.Variable(length)
	
		loss = net(data, length, target)
		loss = loss.mean()
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm(parameters, opt['max_grad_norm'])
		
		optimizer.step()
		i += 1
		end = time.time()

		assert not np.isnan(loss.data[0]), "Model diverse with loss = NaN"
		
		if i % 10 == 0:
			duration = end - begin
			example_per_sec = opt['batch_size'] / duration
			logger.info('Step {}, loss = {:.3f} ({:.1f} exampls/sec; {:.3f} sec/batch)'.format(i, loss.data[0], example_per_sec, duration))

		if i % opt['logging_interval'] == 0:
			parameters_dict = {}
			for p in net.named_parameters():
				if p[1].requires_grad:
					parameters_dict[p[0]] = p[1].cpu().data.numpy()
					parameters_dict[p[0] + '_grad'] = p[1].grad.cpu().data.numpy()

			value_dict = {"loss": loss.data[0], "lr": float(scheduler.optimizer.param_groups[0]['lr'])}
			summary_writer.update(parameters_dict, value_dict, i)

		if i % opt['training_interval'] == 0:
			l = validate(opt, dataset, net)
			scheduler.step(l)
			if l < best_validate:
				patience = 0
				best_validate = l
				if opt['cuda']:
					util.save(net.module, best_validate, i, optimizer, best = True)
				else:
					util.save(net, best_validate, i, optimizer, best = True)
			else:
				patience += 1
			print patience
			
	
	l = validate(opt, dataset, net)
	if l < best_validate:
		patience = 0
		best_validate = l
		if opt['cuda']:
			util.save(net.module, best_validate, i, optimizer, best = True)
		else:
			util.save(net, best_validate, i, optimizer, best = True)
	logger.info("best validate is {}".format(best_validate)) 

	return best_validate


if __name__ == '__main__':
	# Get command line arguments
	argparser = argparse.ArgumentParser()
	config.add_cmd_argument(argparser)
	argparser.add_argument('--patience', type=int, default=10)
	argparser.add_argument('--logging_interval', type=int, default=50)
	argparser.add_argument('--training_interval', type=int, default=500)
	argparser.add_argument('--unit', type=int, default=100)

	opt = vars(argparser.parse_args())

	logger = logging.getLogger('Param Search')
	logger.setLevel(logging.INFO)
	fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
	console = logging.StreamHandler()
	console.setFormatter(fmt)
	logger.addHandler(console)

	logfile = logging.FileHandler('param.log', 'w')
	logfile.setFormatter(fmt)
	logger.addHandler(logfile)

	# Set cuda
	opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
	if opt['cuda']:
		logger.info('[ Using CUDA (GPU %d) ]' % opt['gpu'])
		torch.cuda.set_device(opt['gpu'])

	# Set random state
	np.random.seed(opt['random_seed'])
	random.seed(opt['random_seed'])
	torch.manual_seed(opt['random_seed'])
	if opt['cuda']:
		torch.cuda.manual_seed(opt['random_seed'])

	if os.path.isdir('param_search'):
		shutil.rmtree('param_search')
	os.makedirs('param_search')

	with open('hyperparam.json', 'rb') as f:
		HYPERPARAM = json.load(f)

	experiment = Experiment(300, 4, logger)	
	experiment.set_parames(HYPERPARAM)
	best = experiment.run(lambda x, y: run_and_eval(x,y, opt))

	print best[0]
	print best[1]
