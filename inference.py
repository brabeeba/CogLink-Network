from model import *
from task import *
from experiment import *
from torch.utils.tensorboard import SummaryWriter
import sys
import copy


logger = logging.getLogger('Bandit')
thismodule = sys.modules[__name__]


def create_experiment(opt):
	current_infer = getattr(thismodule, "experiment{}".format(opt['experiment_num']))
	return current_infer(opt)

def experiment1(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = SimpleBG(opt)
	task = DynamicBandit(opt)
	experiment = TimeExperiment(opt)
	return experiment, agent, task, writer

def experiment2(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = SimpleActBG(opt)
	task = DynamicBandit(opt)
	experiment = TimeExperiment(opt)
	return experiment, agent, task, writer

def experiment3(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = TwoTimeScaleNeuralAgent(opt)
	opt["context_num"] = 1
	opt["max_trial"] = 500
	opt["block_size"] = 500
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.3, 0.5, 0.7]]]))
	experiment = Experiment(opt)
	return experiment, agent, task, writer

def experiment4(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = TwoTimeScaleNeuralAgent(opt)
	task = VolatileBandit(opt)
	experiment = Experiment(opt)
	return experiment, agent, task, writer

def experiment5(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = ThompsonDCAgent(opt)
	task = DynamicBandit(opt)
	experiment = Experiment(opt)
	return experiment, agent, task, writer

def experiment6(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = TwoTimeScaleAgent(opt)
	task = HMMBandit(opt)
	experiment = Experiment(opt)
	return experiment, agent, task, writer

def experiment7(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = HMMAgent(opt)
	task = DynamicBandit(opt)
	experiment = Experiment(opt)
	return experiment, agent, task, writer

def experiment8(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [ThompsonDCAgent(opt, name = "Discounted Thompson sampling"),TwoTimeScaleNeuralAgent(opt)]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.7, 0.3]], [[0.3, 0.7]]]))
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment9(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [TwoTimeScaleNeuralAgent(opt), ThompsonDCAgent(opt, name = "Discounted Thompson Sampling")]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.7, 0.3]], [[0.3, 0.7]]]))
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment10(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt1 = model_parameter(opt, opt['experiment_num'])
	opt2 = model_parameter(opt, opt['experiment_num'])
	opt3 = model_parameter(opt, opt['experiment_num'])
	opt4 = model_parameter(opt, opt['experiment_num'])
	opt1["learning"] = True
	opt1["modulate"] = True
	agent1 = TwoTimeScaleNeuralAgent(opt1, name = "Full Model")
	opt2["learning"] = False
	opt2["modulate"] = True
	agent2 = TwoTimeScaleNeuralAgent(opt2, name = "KO learning modulation")
	opt3["learning"] = True
	opt3["modulate"] = False
	agent3 = TwoTimeScaleNeuralAgent(opt3, name = "KO activity modulation")
	opt4["learning"] = False
	opt4["modulate"] = False
	agent4 = TwoTimeScaleNeuralAgent(opt4, name = "KO all modulation")

	writer = SummaryWriter(opt['tensorboard'])
	agent = [agent1, agent2, agent3, agent4]
	task = DynamicBandit(opt)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment11(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [TwoTimeScaleNeuralAgent(opt), ThompsonDCAgent(opt, name = "Discounted Thompson Sampling"), HMMAgent(opt)]
	#agent = [TwoTimeScaleNeuralAgent(opt), ThompsonDCAgent(opt, name = "Discounted Thompson Sampling")]
	
	task = VolatileBandit(opt)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment12(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["class_num"] = 2
	opt["block_size"] = 500
	opt["max_trial"] = 500
	writer = SummaryWriter(opt['tensorboard'])
	agent = [ThompsonDCAgent(opt, name = "Thompson Sampling"), NeuralQuantileAgent(opt, name = "Distributional RPE Model")]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.3, 0.7]]]))
	print(task.prob)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment13(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = RandomNeuralAgent(opt)
	task = DynamicBandit(opt)
	experiment = Experiment(opt)
	return experiment, agent, task, writer

def experiment14(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = NeuralQuantileAgent(opt)
	task = DynamicBandit(opt)
	experiment = Experiment(opt)
	return experiment, agent, task, writer

def experiment15(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["block_size"] = 500
	opt["max_trial"] = 500
	opt["class_num"] = 3
	writer = SummaryWriter(opt['tensorboard'])
	agent = [[ThompsonDCAgent(opt, name = "Thompson Sampling"),  NeuralQuantileAgent(opt, name = "Distributional RPE Model")] for _ in range(10)]
	task = [DynamicBandit(opt) for _ in range(4)]
	for i in range(4):
		task[i].prob = np.array([[[0.2 + i * 0.1, (0.9 + i* 0.1)/2,  0.7]]])
		task[i].name = "gap {}".format(i)

	for i in range(4):
		print(task[i].prob)
	experiment = MultiPlotExperiment(opt)

	return experiment, agent, task, writer

def experiment16(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["block_size"] = 500
	opt["max_trial"] = 500
	option_num = 4
	writer = SummaryWriter(opt['tensorboard'])
	agent = []
	task = [DynamicBandit(opt) for _ in range(option_num)]
	for i in range(option_num):
		task[i].prob = np.linspace(0.3, 0.7, num = i+3)
		task[i].prob = np.expand_dims(task[i].prob, axis = [0, 1])
		task[i].name = "action number {}".format(i)
		task[i].class_num = i+3
		new_opt = copy.deepcopy(opt)
		new_opt["class_num"] = i+3
		a = [ThompsonDCAgent(new_opt, name = "Thompson Sampling"),  NeuralQuantileAgent(new_opt, name = "Distributional RPE Model")]
		agent.append(a)
		

	for i in range(option_num):
		print(task[i].prob)
	experiment = MultiPlotExperiment(opt)

	return experiment, agent, task, writer


def experiment17(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [ThompsonDCAgent(opt, name = "Discounted Thompson Sampling"), TwoTimeScaleNeuralAgent(opt), NeuralQuantileAgent(opt, name = "Distributional RPE Model")]
	task = UnstructuredBandit(opt)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer


def experiment18(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = TwoTimeScaleNeuralAgent(opt)
	task = UnstructuredBandit(opt)
	experiment = Experiment(opt)
	return experiment, agent, task, writer


def experiment19(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [ThompsonDCAgent(opt, name = "Discounted Thompson Sampling"), NeuralContextQuantileAgent(opt), NeuralQuantileAgent(opt, name = "Distributional RPE Model")]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.7, 0.3]], [[0.3, 0.7]]]))
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer


def experiment20(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["class_num"] = 3
	opt["block_size"] = 500
	opt["max_trial"] = 500
	writer = SummaryWriter(opt['tensorboard'])
	agent = []
	for i in range(5):
		new_opt = copy.deepcopy(opt)
		new_opt["quantile_num"] = 20 * (i+1)
		new_opt["K"] = i+1
		agent.append(NeuralQuantileAgent(new_opt, name = "M = {}".format(20 * (i+1))))
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.3, 0.5, 0.7]]]))
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment21(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["class_num"] = 2
	opt["block_size"] = 500
	opt["max_trial"] = 500
	writer = SummaryWriter(opt['tensorboard'])
	agent = [NeuralQuantileAgent(opt, name = "Distributional RPE Model"), ThompsonDCAgent(opt, name = "Thompson Sampling"), QuantileLRAgent(opt, name = "Normative Model")]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.4, 0.6]]]))
	print(task.prob)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment22(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["block_size"] = 500
	opt["max_trial"] = 500
	writer = SummaryWriter(opt['tensorboard'])
	agent = [[NeuralQuantileAgent(opt, name = "Distributional RPE Model"), ThompsonDCAgent(opt, name = "Thompson Sampling"), QuantileLRAgent(opt, name = "Normative Model")] for _ in range(10)]
	task = [DynamicBandit(opt) for _ in range(10)]
	for i in range(10):
		task[i].prob = np.array([[[0.2 + i * 0.05, 0.7]]])
		task[i].name = "gap {}".format(i)

	for i in range(10):
		print(task[i].prob)
	experiment = MultiPlotExperiment(opt)

	return experiment, agent, task, writer

def experiment23(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["block_size"] = 500
	opt["max_trial"] = 500
	option_num = 5
	writer = SummaryWriter(opt['tensorboard'])
	agent = []
	task = [DynamicBandit(opt) for _ in range(option_num)]
	for i in range(option_num):
		task[i].prob = np.linspace(0.3, 0.7, num = i+2)
		task[i].prob = np.expand_dims(task[i].prob, axis = [0, 1])
		task[i].name = "action number {}".format(i)
		new_opt = copy.deepcopy(opt)
		new_opt["class_num"] = i+2
		a = [NeuralQuantileAgent(new_opt, name = "Distributional RPE Model"), ThompsonDCAgent(new_opt, name = "Thompson Sampling"), QuantileLRAgent(new_opt, name = "Normative Model")]
		agent.append(a)
		

	for i in range(option_num):
		print(task[i].prob)
	experiment = MultiPlotExperiment(opt)

	return experiment, agent, task, writer

def experiment24(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["class_num"] = 2
	opt["block_size"] = 500
	opt["max_trial"] = 500
	writer = SummaryWriter(opt['tensorboard'])
	K_list = [1, 3, 10, 30, 90]
	agent = []
	for i in K_list:
		new_opt = copy.deepcopy(opt)
		new_opt["K"] = i
		agent.append(NeuralQuantileAgent(new_opt, name = "K = {}".format(i)))
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.4, 0.6]]]))
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer



def experiment25(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["block_size"] = 500
	opt["max_trial"] = 500
	opt["class_num"] = 3
	writer = SummaryWriter(opt['tensorboard'])
	new_opt = copy.deepcopy(opt)
	new_opt["uniform"] = True
	new_opt["uniform_init"] = True
	agent = [[ NeuralQuantileAgent(opt, name = "Diverse synapses"), NeuralQuantileAgent(new_opt, name = "Uniform synapses")] for _ in range(4)]
	task = [DynamicBandit(opt) for _ in range(4)]
	for i in range(4):
		task[i].prob = np.array([[[0.2 + i * 0.1, (0.9 + i* 0.1)/2,  0.7]]])
		task[i].name = "gap {}".format(i)

	for i in range(4):
		print(task[i].prob)
	experiment = MultiPlotExperiment(opt)

	return experiment, agent, task, writer

def experiment26(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["block_size"] = 500
	opt["max_trial"] = 500
	option_num = 1
	writer = SummaryWriter(opt['tensorboard'])
	agent = []
	task = [DynamicBandit(opt) for _ in range(option_num)]
	for i in range(option_num):
		task[i].prob = np.linspace(0.3, 0.7, num = i+3)
		task[i].prob = np.expand_dims(task[i].prob, axis = [0, 1])
		task[i].name = "action number {}".format(i)
		task[i].class_num = i+3
		opt["class_num"] = i+3
		new_opt = copy.deepcopy(opt)
		new_opt["uniform"] = True
		new_opt["uniform_init"] = True
		a = [ NeuralQuantileAgent(opt, name = "Diverse synapses"), NeuralQuantileAgent(new_opt, name = "Uniform synapses")]
		agent.append(a)
		

	for i in range(option_num):
		print(task[i].prob)
	experiment = MultiPlotExperiment(opt)

	return experiment, agent, task, writer

def experiment27(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt1 = model_parameter(opt, opt['experiment_num'])
	opt2 = model_parameter(opt, opt['experiment_num'])
	
	opt1["learning"] = True
	agent1 = TwoTimeScaleNeuralAgent(opt1, name = "Full model")
	opt2["learning"] = False
	agent2 = TwoTimeScaleNeuralAgent(opt2, name = "No thalamic modulation")

	writer = SummaryWriter(opt['tensorboard'])
	agent = [agent1, agent2]
	task = DynamicBandit(opt)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment28(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt1 = model_parameter(opt, opt['experiment_num'])
	opt2 = model_parameter(opt, opt['experiment_num'])
	
	opt1["learning"] = True
	agent1 = TwoTimeScaleNeuralAgent(opt1, name = "Full model")
	opt2["learning"] = False
	agent2 = TwoTimeScaleNeuralAgent(opt2, name = "No thalamic modulation")

	writer = SummaryWriter(opt['tensorboard'])
	agent = [agent1, agent2]
	task = VolatileBandit(opt)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment29(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt1 = model_parameter(opt, opt['experiment_num'])
	opt2 = model_parameter(opt, opt['experiment_num'])
	
	opt1["nonlinear"] = True
	agent1 = TwoTimeScaleNeuralAgent(opt1, name = "Full model")
	opt2["nonlinear"] = False
	agent2 = TwoTimeScaleNeuralAgent(opt2, name = "RELU")

	writer = SummaryWriter(opt['tensorboard'])
	agent = [agent1, agent2]
	task = DynamicBandit(opt)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment30(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt1 = model_parameter(opt, opt['experiment_num'])
	opt2 = model_parameter(opt, opt['experiment_num'])
	
	opt1["nonlinear"] = True
	agent1 = TwoTimeScaleNeuralAgent(opt1, name = "Full model")
	opt2["nonlinear"] = False
	agent2 = TwoTimeScaleNeuralAgent(opt2, name = "RELU")

	writer = SummaryWriter(opt['tensorboard'])
	agent = [agent1, agent2]
	task = VolatileBandit(opt)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer


def experiment31(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt1 = model_parameter(opt, opt['experiment_num'])
	opt2 = model_parameter(opt, opt['experiment_num'])

	tau_list = [2, 1.5, 1, 0.5]
	agent = []
	for i in tau_list:
		new_opt = copy.deepcopy(opt)
		new_opt["tau"] = i
		agent.append(TwoTimeScaleNeuralAgent(new_opt, name = "tau = {}".format(i)))
	
	writer = SummaryWriter(opt['tensorboard'])
	task = DynamicBandit(opt)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment32(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt1 = copy.deepcopy(opt)
	opt2 = copy.deepcopy(opt)
	
	opt1["inhibit"] = False
	agent1 = TwoTimeScaleNeuralAgent(opt1, name = "Full model")
	opt2["inhibit"] = True
	agent2 = TwoTimeScaleNeuralAgent(opt2, name = "MD inhibition")

	writer = SummaryWriter(opt['tensorboard'])
	agent = [agent1, agent2]
	task = DynamicBandit(opt)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment33(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt1 = copy.deepcopy(opt)
	opt2 = copy.deepcopy(opt)
	opt3 = copy.deepcopy(opt)
	
	opt1["d2"] = False
	agent1 = TwoTimeScaleNeuralAgent(opt1, name = "Full model")
	opt2["d2"] = True
	agent2 = TwoTimeScaleNeuralAgent(opt2, name = "Excess dopamine on MD")
	opt3["d2"] = True
	opt3["rescue"] = True
	agent3 = TwoTimeScaleNeuralAgent(opt3, name = "MD activation rescue")

	writer = SummaryWriter(opt['tensorboard'])
	agent = [agent1, agent2, agent3]
	task = DynamicBandit(opt)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment34(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 2
	opt["class_num"] = 2
	opt["block_size"] = 500
	opt["max_trial"] = 500
	opt1 = copy.deepcopy(opt)
	opt2 = copy.deepcopy(opt)
	opt["context_num"] = 1
	

	opt1["inhibit"] = False
	agent1 = TwoTimeScaleNeuralAgent(opt1, name = "Full model")
	opt2["inhibit"] = True
	agent2 = TwoTimeScaleNeuralAgent(opt2, name = "MD inhibition")


	writer = SummaryWriter(opt['tensorboard'])
	agent = [agent1, agent2, ThompsonDCAgent(opt, name = "Thompson Sampling")]
	task = DynamicBandit(opt)
	task.prob = np.expand_dims(np.array([0.3, 0.7]), axis = [0, 1])
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment35(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["class_num"] = 2
	opt["block_size"] = 500
	opt["max_trial"] = 500
	new_opt = copy.deepcopy(opt)
	new_opt["uniform"] = True
	new_opt["uniform_init"] = True
	writer = SummaryWriter(opt['tensorboard'])
	agent = [NeuralQuantileAgent(opt, name = "Diverse synapses"), NeuralQuantileAgent(new_opt, name = "Uniform synapses")]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.3, 0.7]]]))
	print(task.prob)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment36(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 2
	opt["class_num"] = 2
	opt["block_size"] = 500
	opt["max_trial"] = 500
	opt1 = copy.deepcopy(opt)
	opt2 = copy.deepcopy(opt)
	opt["context_num"] = 1
	

	opt1["inhibit"] = False
	agent1 = TwoTimeScaleNeuralAgent(opt1, name = "Full model")
	opt2["inhibit"] = True
	agent2 = TwoTimeScaleNeuralAgent(opt2, name = "MD inhibition")

	writer = SummaryWriter(opt['tensorboard'])
	agent = [[ agent1, agent2,ThompsonDCAgent(opt, name = "Thompson Sampling") ] for _ in range(4)]
	task = [DynamicBandit(opt) for _ in range(4)]
	for i in range(4):
		task[i].prob = np.array([[[0.2 + i * 0.1,  0.7]]])
		task[i].name = "gap {}".format(i)

	for i in range(4):
		print(task[i].prob)
	experiment = MultiPlotExperiment(opt)

	return experiment, agent, task, writer

def experiment37(opt):
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["class_num"] = 3
	opt["block_size"] = 500
	opt["max_trial"] = 500
	

	writer = SummaryWriter(opt['tensorboard'])
	agent = []
	task = [DynamicBandit(opt) for _ in range(option_num)]
	for i in range(option_num):
		task[i].prob = np.linspace(0.3, 0.7, num = i+3)
		task[i].prob = np.expand_dims(task[i].prob, axis = [0, 1])
		task[i].name = "action number {}".format(i)
		task[i].class_num = i+3
		opt1 = copy.deepcopy(opt)
		opt2 = copy.deepcopy(opt)
		opt1["inhibit"] = False
		opt1["class_num"] = i+3
		opt2["inhibit"] = True
		opt2["class_num"] = i+3
		opt1["context_num"] = 2
		opt2["context_num"] = 2
		agent1 = TwoTimeScaleNeuralAgent(opt1, name = "Full model")
		agent2 = TwoTimeScaleNeuralAgent(opt2, name = "MD inhibition")
		
		a = [ agent1, agent2, ThompsonDCAgent(new_opt, name = "Thompson Sampling"),  agent1, agent2]
		agent.append(a)
		

	for i in range(option_num):
		print(task[i].prob)
	experiment = MultiPlotExperiment(opt)

	return experiment, agent, task, writer






def model_parameter(opt, model_num):
	opt = copy.deepcopy(opt)

	new_opt = {}
	if model_num == 1:
		#experiment parameter
		new_opt["stimuli_time"] = 0.1
		new_opt["waiting_time"] = 0.5
		new_opt["reward_time"] = 0.1
		new_opt["inter_trial"] = 0.5

		#model parameter
		# new_opt["time_size"] = 150
		# new_opt["msn_size"] = 600
		# new_opt["c_decay"] = 50
		# new_opt["s_decay"] = 50
		# new_opt["p_decay"] = 50
		# new_opt["v_decay"] = 100
		# new_opt["da_decay"] = 100
		# new_opt["ep_decay"] = 50
		# new_opt["ec_decay"] = 50
		# new_opt["cs_lr_p"] = 5
		# new_opt["cs_lr_n"] = 25
		# new_opt["gamma"] = 0.99
		# new_opt["noise"] = 0.01
		new_opt["time_size"] = 170
		new_opt["c_decay"] = 200
		#new_opt["time_size"] = 120
		#new_opt["c_decay"] = 150
		#new_opt["time_size"] = 60
		#new_opt["c_decay"] = 80
		new_opt["p_decay"] = 20
		new_opt["v_decay"] = 20
		new_opt["da_decay"] = 100
		new_opt["ep_decay"] = 1
		new_opt["ev_decay"] = 1
		new_opt["cs_lr_p"] = 10
		new_opt["cs_lr_n"] = 10
		new_opt["gamma"] = 0.999
		new_opt["noise"] = 0.1

	if model_num == 2:
		#experiment parameter
		new_opt["stimuli_time"] = 0.1
		new_opt["waiting_time"] = 0.5
		new_opt["reward_time"] = 0.1
		new_opt["inter_trial"] = 0.5

		#model parameter
		#new_opt["time_size"] = 170
		#new_opt["c_decay"] = 200
		#new_opt["time_size"] = 120
		#new_opt["c_decay"] = 150
		new_opt["time_size"] = 60
		new_opt["c_decay"] = 80
		new_opt["p_decay"] = 20
		new_opt["v_decay"] = 20
		new_opt["da_decay"] = 100
		new_opt["ep_decay"] = 1
		new_opt["ev_decay"] = 1
		new_opt["cs_lr_p"] = 50
		new_opt["cs_lr_n"] = 50
		new_opt["gamma"] = 0.999
		new_opt["noise"] = 0.1

	if model_num == 3:
		new_opt["history"] = 2.5
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["iter"] = 50
		new_opt["learning"] = True
		new_opt["nonlinear"] = True
		new_opt["inhibit"] = False
		new_opt["d2"] = False
		new_opt["rescue"] = False
		new_opt["lr_exp"] = 1

		
		new_opt["quantile_num"] = 100
		new_opt["N"] = 50
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau1"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000

	if model_num == 4:
		new_opt["history"] = 2.5
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["iter"] = 80
		new_opt["N"] = 20
		new_opt["learning"] = True
		new_opt["nonlinear"] = True


		new_opt["gamma"] = 1
		new_opt["quantile_num"] = 100
		new_opt["N"] = 20
		new_opt["K"] = 3
		new_opt["a"] = 1.25
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau1"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000



		

	

	if model_num == 5:
		new_opt["gamma"] = 0.9

	if model_num == 6:
		new_opt["switch_prob"] = 0.005
		new_opt["history"] = 3.5
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 1
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30

	if model_num == 7:
		new_opt["iter"] = 50

	if model_num == 8:
		new_opt["history"] = 2.5
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["N"] = 50
		new_opt["learning"] = True
		new_opt["nonlinear"] = True

		new_opt["gamma"] = 1
		new_opt["quantile_num"] = 100
		new_opt["N"] = 50
		new_opt["A"] = 7
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 1
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau1"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 1
		new_opt["d_interval"] = 1000
		new_opt["A"] = 7


	if model_num == 9:
		new_opt["history"] = 2.5
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["iter"] = 40
		new_opt["learning"] = True
		new_opt["nonlinear"] = True
		new_opt["inhibit"] = False
		new_opt["d2"] = False
		new_opt["rescue"] = False

		
		new_opt["quantile_num"] = 100
		new_opt["N"] = 50
		
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau1"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000

	if model_num == 10:
		new_opt["history"] = 2.5
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["iter"] = 80
		new_opt["N"] = 50

	if model_num == 11:
		new_opt["history"] = 2.5
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["iter"] = 50
		new_opt["learning"] = True
		new_opt["nonlinear"] = True

		new_opt["quantile_num"] = 100
		new_opt["N"] = 10
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau1"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000


	if model_num == 12:
		new_opt["gamma"] = 1
		new_opt["quantile_num"] = 100
		
		new_opt["N"] = 50
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000
		new_opt["A"] = 1
		new_opt["uniform"] = False
		new_opt["uniform_init"] = False


	if model_num == 13:
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["tau"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 1
		new_opt["d_interval"] = 500

	if model_num == 14:
		new_opt["a1"] = 1
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 1
		new_opt["d_interval"] = 500
		new_opt["quantile_num"] = 100
		new_opt["A"] = 7

	if model_num == 15:
		new_opt["gamma"] = 1
		new_opt["quantile_num"] = 100
		new_opt["N"] = 50
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000
		new_opt["A"] = 7
		new_opt["uniform"] = False
		new_opt["uniform_init"] = False

	if model_num == 16:
		new_opt["gamma"] = 1
		new_opt["quantile_num"] = 100
		new_opt["N"] = 50
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000
		new_opt["A"] = 7
		new_opt["uniform"] = False
		new_opt["uniform_init"] = False
		

	if model_num == 17:
		new_opt["quantile_num"] = 100
		new_opt["N"] = 20
		new_opt["A"] = 7
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 1
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 1
		new_opt["d_interval"] = 1000
		new_opt["A"] = 7


		new_opt["history"] = 2.5
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["iter"] = 80
		new_opt["learning"] = True
		new_opt["nonlinear"] = True

	if model_num == 18:
		
		new_opt["history"] = 2.5
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["iter"] = 80
		new_opt["learning"] = True
		new_opt["nonlinear"] = True

	if model_num == 19:
		new_opt["gamma"] = 0.93
		new_opt["quantile_num"] = 100
		new_opt["uniform"] = False
		new_opt["uniform_init"] = False
		new_opt["N"] = 50
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000
		
	if model_num == 20 or model_num == 24 or model_num == 25 or model_num == 26 or model_num == 35:
		new_opt["gamma"] = 1
		new_opt["quantile_num"] = 100
		new_opt["uniform"] = False
		new_opt["uniform_init"] = False
		new_opt["N"] = 50
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000
		new_opt["A"] = 1

	if model_num == 21 or model_num == 22 or model_num == 23:
		new_opt["gamma"] = 1
		new_opt["quantile_num"] = 100
		new_opt["N"] = 100
		new_opt["A"] = 7
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 1
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000
		new_opt["A"] = 7

	if model_num == 27 or model_num == 28 or model_num == 29 or model_num == 30:
		new_opt["history"] = 2.5
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["iter"] = 50
		new_opt["learning"] = True
		new_opt["nonlinear"] = True

		
		new_opt["quantile_num"] = 100
		new_opt["N"] = 50
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau1"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000

	if model_num == 31 or model_num == 32 or model_num == 33 or model_num == 34 or model_num == 36 or model_num == 37:
		new_opt["history"] = 2.5
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 1
		new_opt["iter"] = 40
		new_opt["learning"] = True
		new_opt["nonlinear"] = True
		new_opt["inhibit"] = False
		new_opt["d2"] = False
		new_opt["rescue"] = False

		
		new_opt["quantile_num"] = 100
		new_opt["N"] = 50
		
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1
		new_opt["b2"] = 1
		new_opt["tau1"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 0.8
		new_opt["d_interval"] = 1000






	
	logger.info('Model parameter is the following:')
	for k, v in new_opt.items():
		opt[k] = v
		logger.info('{}: {}'.format(k, v))

	return opt

