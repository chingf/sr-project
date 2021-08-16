import os
os.chdir('../')

import pickle
import numpy as np
import torch.nn as nn
from joblib import Parallel, delayed

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR
from train import train

experiment_dir = './trained_models/02_gamma_v_ss/'

datasets = [inputs.Sim1DWalk]
datasets_config_ranges = [{
    'num_steps': [3, 10, 20, 30],
    'left_right_stay_prob': [[1, 1, 1], [7, 1, 1], [1, 4, 1]],
    'num_states': [5, 10, 15, 25, 36]
    }]


gammas = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

def grid_train(gamma):
    gamma_exp_dir = experiment_dir + f'{gamma}/'
    last_idx = os.listdir(gamma_exp_dir)
    if len(last_idx) > 0:
        last_idx = max([int(idx) for idx in last_idx])
    else:
        last_idx = 0
    for idx in range(last_idx+1, 20):
        save_path = gamma_exp_dir + f'{idx}/'
        net = STDP_SR(num_states=2, gamma=gamma)
        train(
            save_path, net, datasets, datasets_config_ranges,
            train_steps=301, print_every_steps=10
            )
        net_configs = {'gamma':gamma}
        with open(save_path + 'net_configs.p', 'wb') as f:
            pickle.dump(net_configs, f)

Parallel(n_jobs=6)(delayed(grid_train)(gamma) for gamma in gammas)
