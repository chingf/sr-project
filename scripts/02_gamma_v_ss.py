import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import torch.nn as nn
from joblib import Parallel, delayed

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR
from train import train
import pickle

experiment_dir = '../../engram/Ching/02_gamma_v_ss/'
n_jobs = 56
n_iters = 5

datasets = [
    inputs.Sim1DWalk,
    inputs.Sim1DWalk,
    ]
datasets_config_ranges = [
    {
    'num_steps': [3, 10, 20, 30],
    'left_right_stay_prob': [[1, 1, 1], [1, 1, 5], [5, 1, 0], [1, 5, 0]],
    'num_states': [5, 10, 15, 25]
    },
    {
    'num_steps': [100, 200],
    'left_right_stay_prob': [[1, 1, 1], [5, 1, 1]],
    'num_states': [10, 15]
    },
    ]

gammas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
nonlinearity_args = [None, 1.0, 1.5, 2.0]
args = []
for gamma in gammas:
    for nonlinearity_arg in nonlinearity_args:
        for _iter in range(n_iters):
            args.append((gamma, nonlinearity_arg, _iter))

def grid_train(arg):
    gamma, nonlinearity_arg, _iter = arg
    iter_dir = experiment_dir + f'{gamma}/{nonlinearity_arg}/{_iter}/'
    if os.path.isfile(iter_dir + 'model.pt'):
        return
    if not os.path.exists(iter_dir):
        os.makedirs(iter_dir)

    if nonlinearity_arg is None:
        output_params = {}
        train_M = False
    else:
        rstep = int(np.log(1E-5)/np.log(gamma))
        output_params = {
            'num_iterations':rstep, 'input_clamp':rstep,
            'nonlinearity': 'tanh', 'nonlinearity_args': nonlinearity_arg
            }
        train_M = True
    net_configs = {
        'gamma':gamma,
        'ca3_kwargs':{
            'A_pos_sign':1, 'A_neg_sign':-1,
            'output_params': output_params
            }
        }

    net = STDP_SR(
        num_states=2, gamma=gamma, ca3_kwargs=net_configs['ca3_kwargs']
        )
    train(
        iter_dir, net, datasets, datasets_config_ranges,
        train_steps=301, print_every_steps=10, train_M=train_M
        )
    with open(iter_dir + 'net_configs.p', 'wb') as f:
        pickle.dump(net_configs, f)

Parallel(n_jobs=n_jobs)(delayed(grid_train)(arg) for arg in args)
