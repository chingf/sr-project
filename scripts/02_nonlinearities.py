import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
import numpy as np
import torch.nn as nn
from joblib import Parallel, delayed

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR
from train import train

experiment_dir = '../trained_models/02_nonlinearities/'

datasets = [inputs.Sim1DWalk]
datasets_config_ranges = [{
    'num_steps': [3, 10, 20, 30, 40],
    'left_right_stay_prob': [[1, 1, 1], [7, 1, 1], [1, 4, 1]],
    'num_states': [5, 10, 15, 25]
    }]


gammas = [0.4, 0.9, 0.95]
nonlinearities = ['None', 'sigmoid', 'tanh', 'clamp']

args = []
for gamma in gammas:
    for nonlinearity in nonlinearities:
        args.append((gamma, nonlinearity))

def grid_train(arg):
    gamma, nonlinearity = arg
    gamma_dir = experiment_dir + f'{gamma}/'
    nonlinearity_dir = gamma_dir + f'{nonlinearity}/'
    losses = []

    if gamma == 0.4:
        num_iters = 30 # 1E-12
    elif gamma == 0.8:
        num_iters = 50 # 1E-5
    elif gamma == 0.9:
        num_iters = 100 # 2E-5
    elif gamma == 0.95:
        num_iters = 200 # 2E-5
    else:
        raise ValueError('Unspecified gamma.')

    output_params = {
        'num_iterations':num_iters, 'input_clamp':num_iters,
        'nonlinearity': nonlinearity
        }
    net_configs = {
        'num_states': 2, 'gamma':gamma,
        'ca3_kwargs':{'output_params':output_params}
        }
    for idx in range(5):
        iteration_dir = nonlinearity_dir + f'{idx}/'
        if os.path.isdir(iteration_dir): continue
        net = STDP_SR(**net_configs)
        net, loss = train(
            iteration_dir, net, datasets, datasets_config_ranges,
            train_steps=801, print_every_steps=25,
            early_stop=True, return_test_error=True
            )
    with open(nonlinearity_dir + 'net_configs.p', 'wb') as f:
        pickle.dump(net_configs, f)

Parallel(n_jobs=7)(delayed(grid_train)(arg) for arg in args)

