def add_cmd_argument(parser):
	parser.register('type', 'bool', (lambda x: x.lower() in ("yes", "true", "t", "1")))
	parser.add_argument('--train_dir', type=str, default='./train_dir')
	parser.add_argument('--tensorboard', type=str, default='./train_dir/tensorboard')
	parser.add_argument('--log_file', type=str, default='./train_dir/run.log')
	parser.add_argument('--no_cuda', type='bool', default=False)
	parser.add_argument('--gpu', type=list, default = 0)
	parser.add_argument('--random_seed', type=int, default=123)
	parser.add_argument('--context_num', type=int, default=2)
	parser.add_argument('--stimuli_num', type=int, default=1)
	parser.add_argument('--class_num', type=int, default=2)
	parser.add_argument('--block_size', type=int, default=200)
	parser.add_argument('--max_trial', type=int, default=1000)
	parser.add_argument('--logging_interval', type=int, default=1)
	parser.add_argument('--dt', type=int, default=0.005)
	parser.add_argument('--experiment_num', type=int, default=1)
	

	
	#parser.add_argument('--save_file', type=str, default='./train_dir/model.ckpt')
	#parser.add_argument('--best_save_file', type=str, default='./train_dir/best_model.ckpt')
	#parser.add_argument('--embedding_file', type=str, default='./glove.6B.100d.txt')
	#parser.add_argument('--pretrain_word', type='bool', default=True)

	
	#parser.add_argument('--data_dir', type=str, default='../data')
	#parser.add_argument('--current_model', type=int, default=3)

	#parser.add_argument('--finetune', type='bool', default=False)

	#parser.add_argument('--new_model', type='bool', default=False)