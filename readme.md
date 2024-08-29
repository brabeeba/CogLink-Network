# Code for "Revealing how uncertainty computations drive hierarchical reasoning via CogLink Networks"
DOI: 10.5281/zenodo.13152289

## Installation
The code has been tested on python 3.8.2. It is highly recommended to use virtual environment to install the package. 
> virtualenv env

To install all the dependency, we recommend to use pip to install. This process will take about 5 minutes.
>pip install -r requirements.txt

## Data Availability
To run the code, please download the corresponding dataset deposited at Mendeley. The DOI is 10.17632/3ffrk7bw9h.1.

## Inspect the dataset
Dataset is stored via pickle.dump. The name of the dataset for experiment 1 is experiment1_data and the rest of the dataset follows similar naming convention. To load the dataset, simply use load_dict from util.py.
>from util import load_dict
>
>data = load_dict("experiment1_data")

The above code loads the data from experiment 1.

## File description
config.py is used to setup the parameter of the experiments.

train.py is used to run the experiment and generate data. Following code is an example to run experiment 1.
> python train.py --experiment_num 1

plot.py is used to analyze the data and generate the figures. Following code is an example to analyze experiment 1.
>python plot.py --experiment_num 1

model.py specifies all the models in the experiments. 

task.py specifies which task to run.

inference.py specifies the hyperparameters of the models in each experiment. 

cumsum_run.py is used to generate figure 5m 

experiment.py is a helper script to conduct and log the experiments. 

gmail_logger.py, tensor_logger.py, util.py, run.py are utility scripts that helps running the exepriments.

## Experiments description
Experiment 1, 2 generates figure 2, S2. Experiment 3, 4 generates figure 3, S3. Experiment 5 generates figure 4, S4. Experiment 6 generates figure 5, 6, S6. Experiment 7 generates figure 7. Experiment 8 generates figure 7i, S7. Experiment 9 generates figure 8, S8.


