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
	#Figure 2b, S1a, S1c
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["block_size"] = 500
	opt["max_trial"] = 500
	opt["class_num"] = 3
	writer = SummaryWriter(opt['tensorboard'])
	agent = [[ThompsonDCAgent(opt, name = "Thompson Sampling") , NeuralQuantileAgent(opt, name = "Distributional RPE Model")] for _ in range(4)]
	task = [DynamicBandit(opt) for _ in range(4)]
	for i in range(4):
		task[i].prob = np.array([[[0.2 + i * 0.1, (0.9 + i* 0.1)/2,  0.7]]])
		task[i].name = "gap {}".format(i)

	for i in range(4):
		print(task[i].prob)
	experiment = MultiPlotExperiment(opt)

	return experiment, agent, task, writer

def experiment2(opt):
	#Figure 2c, S1b, S1c
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
		a = [ThompsonDCAgent(opt, name = "Thompson Sampling"),  NeuralQuantileAgent(new_opt, name = "Distributional RPE Model")]
		agent.append(a)
		

	for i in range(option_num):
		print(task[i].prob)
	experiment = MultiPlotExperiment(opt)

	return experiment, agent, task, writer


def experiment3(opt):
	#Figure NA
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

def experiment4(opt):
	#Figure NA
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


def experiment5(opt):
	#Figure S3
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [ThompsonDCAgent(opt, name = "Discounted Thompson Sampling"), NeuralContextQuantileAgent(opt), NeuralQuantileAgent(opt, name = "Distributional RPE Model")]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.7, 0.3]], [[0.3, 0.7]]]))
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment6(opt):
	#Figure NA
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [ThompsonDCAgent(opt, name = "Discounted Thompson Sampling"), TwoTimeScaleNeuralAgent(opt)]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.7, 0.3]], [[0.3, 0.7]]]))
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer



def experiment7(opt):
	#Figure 6
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

def experiment8(opt):
	#Figure 6i, S6
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

def experiment9(opt):
	#Figure 7, S7
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

def experiment10(opt):
	#Figure 5, 6 DEMO
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [TwoTimeScaleNeuralAgent(opt), ThompsonDCAgent(opt, name = "Discounted Thompson Sampling")]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.7, 0.3]], [[0.3, 0.7]]]))
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer


def experiment11(opt):
	#Figure S2d
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [ThompsonRLAgent(opt), DQNRL(opt), NeuralQuantileRLAgent(opt)]
	task = SimpleMDP(opt)
	experiment = PlotRLExperiment(opt)
	return experiment, agent, task, writer

def experiment12(opt): 
	#Figure 2i
	opt = model_parameter(opt, opt['experiment_num'])
	opt["context_num"] = 1
	opt["block_size"] = 500
	opt["max_trial"] = 500
	opt["class_num"] = 3
	new_opt = copy.deepcopy(opt)
	new_opt["quantile_num"] = 1
	new_opt["uniform_init"] = True

	new_opt1 = copy.deepcopy(opt)
	new_opt1["K"] = 80
	writer = SummaryWriter(opt['tensorboard'])
	agent = [ NeuralQuantileAgent(opt, name = "Distributional RPE Model"), NeuralQuantileAgent(new_opt1, name = "Non-sparse Sampling"), NeuralQuantileAgent(new_opt, name = "Scalar RPE Model")]
	task = DynamicBandit(opt)
	task.prob = [[[0.3, 0.5, 0.7]]]
	experiment = PlotExperiment(opt)

	return experiment, agent, task, writer

def experiment13(opt):
	#Figure 4f, 4k
	opt = model_parameter(opt, opt['experiment_num'])
	opt1 = copy.deepcopy(opt)
	opt2 = copy.deepcopy(opt)
	opt3 = copy.deepcopy(opt)
	
	
	agent1 = TwoTimeScaleNeuralAgent(opt1, name = "Full model")
	opt2["md_learning"] = False
	agent2 = TwoTimeScaleNeuralAgent(opt2, name = "MD KO")
	opt3["in_learning"] = False
	agent3 = TwoTimeScaleNeuralAgent(opt3, name = "IN KO")

	writer = SummaryWriter(opt['tensorboard'])
	agent = [agent1, agent2, agent3]
	task = DynamicBandit(opt)
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment14(opt):
	#Figure 4, 5
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [ HMMThompsonAgent(opt, name = "HMM"),  DQN(opt, name = "DQN"),  TwoTimeScaleNeuralAgent(opt, name = "Thalamocortical Model")]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.7, 0.3]], [[0.3, 0.7]]]))
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer

def experiment15(opt):
	#Figure S2d
	opt = model_parameter(opt, opt['experiment_num'])
	writer = SummaryWriter(opt['tensorboard'])
	agent = [ThompsonRLAgent(opt), DQNRL(opt), NeuralQuantileRLAgent(opt)]
	task = SimpleMDP(opt)
	task.prob1 = np.array([[2/3, 1/3]])
	experiment = PlotRLExperiment(opt)
	return experiment, agent, task, writer

def experiment16(opt):
	#Figure S2b
	opt = model_parameter(opt, opt['experiment_num'])
	opt["stimuli_num"] = 2
	opt["context_num"] = 1
	opt["block_size"] = 500
	opt["max_trial"] = 500
	opt["class_num"] = 2
	writer = SummaryWriter(opt['tensorboard'])

	agent = [[ThompsonDCAgent(opt, name = "Thompson Sampling"), DQN(opt, name = "DQN"), NeuralQuantileAgent(opt, name = "Distributional RPE Model")] for _ in range(2)]
	task = [DynamicBandit(opt) for _ in range(2)]
	for i in range(2):
		task[i].prob = np.array([[[0.7, 0.3 + 0.2 * i], [0.3 + 0.2 * i, 0.7]]])
		task[i].name = "gap {}".format(i)

	for i in range(2):
		print(task[i].prob)
	experiment = MultiPlotExperiment(opt)

	return experiment, agent, task, writer

def experiment17(opt):
	#Figure 5i
	opt = model_parameter(opt, opt['experiment_num'])
	
	writer = SummaryWriter(opt['tensorboard'])
	agent = [ HMMThompsonAgent(opt, name = "HMM"),  TwoTimeScaleNeuralAgent(opt, name = "Thalamocortical Model")]
	task = DynamicBandit(opt)
	task.setprob(np.array([[[0.9, 0.1]], [[0.1, 0.9]]]))
	experiment = PlotExperiment(opt)
	return experiment, agent, task, writer








def model_parameter(opt, model_num):
	opt = copy.deepcopy(opt)

	new_opt = {}

	if model_num == 1 or model_num == 2 or model_num == 3 or model_num == 4 or model_num == 12 or model_num == 16:
		new_opt["gamma"] = 1
		new_opt["quantile_num"] = 100
		new_opt["N"] = 50
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 2
		new_opt["b2"] = 1
		new_opt["tau"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 1
		new_opt["d_interval"] = 1000
		new_opt["uniform"] = False
		new_opt["uniform_init"] = False
		new_opt["epsilon"] = 1
		new_opt["lr"] = 0.1
		new_opt["hidden_size"] = 10
		new_opt["md_learning"] = True
		new_opt["in_learning"] = True
		new_opt["fixmd"] = False
		new_opt["volatile"] = False



	if model_num == 5:
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
		new_opt["hidden_size"] = 10
		new_opt["md_learning"] = True
		new_opt["in_learning"] = True
		new_opt["fixmd"] = False
		new_opt["volatile"] = False


	if model_num == 6:
		
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["iter"] = 40
		new_opt["hidden_size"] = 10
		new_opt["md_learning"] = True
		new_opt["in_learning"] = True
		new_opt["fixmd"] = False
		new_opt["volatile"] = False

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
		new_opt["a2"] = 1.2
		new_opt["b2"] = 1
		new_opt["tau1"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 1
		new_opt["d_interval"] = 1000
		


	if model_num == 7 or model_num == 8 or model_num == 9 or model_num == 13:
		
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 1
		new_opt["iter"] = 40
		new_opt["hidden_size"] = 10
		new_opt["md_learning"] = True
		new_opt["in_learning"] = True
		new_opt["fixmd"] = False
		new_opt["volatile"] = False

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
		new_opt["a2"] = 1.2
		new_opt["b2"] = 1
		new_opt["tau1"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 1
		new_opt["d_interval"] = 1000

	if model_num == 10:
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 0.93
		new_opt["iter"] = 40
		new_opt["hidden_size"] = 10
		new_opt["md_learning"] = True
		new_opt["in_learning"] = True
		new_opt["fixmd"] = False
		new_opt["volatile"] = False

		new_opt["inhibit"] = False
		new_opt["d2"] = False
		new_opt["rescue"] = False
		new_opt["quantile_num"] = 100
		new_opt["N"] = 5
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

	if model_num == 11 or model_num == 15:
		new_opt["gamma"] = 1
		new_opt["quantile_num"] = 100
		new_opt["stimuli_num"] = 3
		new_opt["context_num"] = 1
		new_opt["block_size"] = 500
		new_opt["max_trial"] = 500

		new_opt["N"] = 50
		new_opt["K"] = 3
		new_opt["a"] = 1
		new_opt["b"] = 1
		new_opt["a1"] = 0.75
		new_opt["b1"] = 1
		new_opt["a2"] = 1.2
		new_opt["b2"] = 1
		new_opt["tau"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 2
		new_opt["d_interval"] = 1000
		new_opt["uniform"] = False
		new_opt["uniform_init"] = False
		new_opt["epsilon"] = 1
		new_opt["lr"] = 0.1
		new_opt["hidden_size"] = 10
		new_opt["md_learning"] = True
		new_opt["in_learning"] = True
		new_opt["fixmd"] = False
		new_opt["volatile"] = False

	if model_num == 14 or model_num == 17:
		new_opt["gamma1"] = 0.99
		new_opt["gamma2"] = 0.8
		new_opt["lr"] = 0.1
		new_opt["temperature"] = 30
		new_opt["tau"] = 2
		new_opt["a"] =  1
		new_opt["gamma"] = 1
		new_opt["iter"] = 40
		new_opt["hidden_size"] = 10
		new_opt["md_learning"] = True
		new_opt["in_learning"] = True
		new_opt["fixmd"] = False
		new_opt["volatile"] = False

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
		new_opt["a2"] = 1.2
		new_opt["b2"] = 1
		new_opt["tau1"] = 10
		new_opt["eta"] = 0.5
		new_opt["threshold"] = 2
		new_opt["d_interval"] = 1000
		new_opt["learned"] = False
		new_opt["hidden_size"] = 10
		new_opt["epsilon"] = 1
		new_opt["lr"] = 0.1
		new_opt['volatile'] = True







	
	logger.info('Model parameter is the following:')
	for k, v in new_opt.items():
		opt[k] = v
		logger.info('{}: {}'.format(k, v))

	return opt

