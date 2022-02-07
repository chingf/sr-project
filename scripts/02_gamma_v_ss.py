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

experiment_dir = '../../engram/Ching/02_gamma_v_ss/'
n_jobs = 7

datasets = [
    inputs.Sim1DWalk,
    inputs.Sim1DWalk,
    ]
datasets_config_ranges = [
    {
    'num_steps': [3, 10, 20, 30],
    'left_right_stay_prob': [[1, 1, 1], [1, 1, 5], [7, 1, 0], [1, 7, 0]],
    'num_states': [5, 10, 15, 25]
    },
    {
    'num_steps': [100, 200],
    'num_states': [36, 64]
    },
    ]

gammas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def grid_train(gamma):
    gamma_exp_dir = experiment_dir + f'{gamma}/'
    if not os.path.exists(gamma_exp_dir):
        os.makedirs(gamma_exp_dir)
    last_idx = os.listdir(gamma_exp_dir)
    if len(last_idx) > 0:
        last_idx = max([int(idx) for idx in last_idx])
    else:
        last_idx = 0
    for idx in range(last_idx+1, last_idx+4):
        save_path = gamma_exp_dir + f'{idx}/'
        net = STDP_SR(num_states=2, gamma=gamma)
        train(
            save_path, net, datasets, datasets_config_ranges,
            train_steps=601, print_every_steps=10, train_M=False
            )
        net_configs = {'gamma':gamma}
        with open(save_path + 'net_configs.p', 'wb') as f:
            pickle.dump(net_configs, f)

Parallel(n_jobs=n_jobs)(delayed(grid_train)(gamma) for gamma in gammas)
