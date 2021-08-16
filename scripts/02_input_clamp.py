import os
os.chdir('../')

import pickle
import numpy as np
import torch.nn as nn
from joblib import Parallel, delayed

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR
from train import train

experiment_dir = './trained_models/02_input_clamps/'

datasets = [inputs.Sim1DWalk]
datasets_config_ranges = [{
    'num_steps': [3, 10, 20, 30],
    'left_right_stay_prob': [[1, 1, 1], [7, 1, 1], [1, 4, 1]],
    'num_states': [5, 10, 15, 25, 36]
    }]

input_clamps = [5, 20, 25, 28, 29, 30]
gamma = 0.4

def grid_train(input_clamp):
    ic_exp_dir = experiment_dir + f'{input_clamp}/'
    for idx in range(15):
        save_path = ic_exp_dir + f'{idx}/'
        output_params = {
            'input_clamp':input_clamp, 'num_iterations':30,
            'nonlinearity': None, 'transform_activity': False,
            'clamp_activity': True
            }
        net_configs = {
            'gamma':gamma,
            'ca3_kwargs':{'output_params': output_params}
            }
        net = STDP_SR(
            num_states=2, gamma=gamma, ca3_kwargs=net_configs['ca3_kwargs']
            )
        train(
            save_path, net, datasets, datasets_config_ranges,
            train_steps=301, print_every_steps=10
            )
        with open(save_path + 'net_configs.p', 'wb') as f:
            pickle.dump(net_configs, f)

Parallel(n_jobs=6)(delayed(grid_train)(i_c) for i_c in input_clamps)
