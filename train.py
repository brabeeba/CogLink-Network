import torch
import logging
import argparse
import config
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import random
import time
import util
import glob
from pathlib import Path
from task import DynamicBandit
from inference import create_experiment



def main(opt):
	logger.info("Model Training begin with options:")
	for k, v in opt.items():
		logger.info("{} : {}".format(k, v))

	experiment, agent, task, writer = create_experiment(opt)
	experiment.run(agent, task, writer)
	 

if __name__ == '__main__':
	# Get command line arguments
	argparser = argparse.ArgumentParser()
	config.add_cmd_argument(argparser)
	opt = vars(argparser.parse_args())
	# Set logging
	logger = logging.getLogger('Bandit')
	logger.setLevel(logging.INFO)
	fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
	console = logging.StreamHandler()
	console.setFormatter(fmt)
	logger.addHandler(console)

	if 'train_dir' in opt:
		Path(opt['train_dir']).mkdir(parents=True, exist_ok=True)

	if 'tensorboard' in opt:
		files = glob.glob(opt['tensorboard'] + "/*")
		for f in files:
			os.remove(f)


	if 'log_file' in opt:
		logfile = logging.FileHandler(opt['log_file'], 'w')
		logfile.setFormatter(fmt)
		logger.addHandler(logfile)
	
	logger.info('[ COMMAND: %s ]' % ' '.join(sys.argv))

	# Set cuda
	opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
	if opt['cuda']:
		logger.info('[ Using CUDA (GPU %d) ]' % opt['gpu'])
		torch.cuda.set_device(opt['gpu'])

	# Set random state
	#random.seed(opt['random_seed'])
	#np.random.seed(opt['random_seed'])
	#torch.manual_seed(opt['random_seed'])
	#if opt['cuda']:
	#	torch.cuda.manual_seed(opt['random_seed'])

	main(opt)
