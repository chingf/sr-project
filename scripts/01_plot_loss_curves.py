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

experiment_dir = '../trained_models/01_loss_curves/'
experiment_dir = '../../engram/Ching/01_loss_curves/'

datasets = [inputs.Sim1DWalk]
datasets_config_ranges = [
    {
    'num_steps': [3, 10, 20, 30, 40],
    'left_right_stay_prob': [[1, 1, 1], [7, 1, 1], [1, 4, 1]],
    'num_states': [5, 10, 15, 25]
    },
    ]

args = np.arange(50)

def grid_train(idx):
    save_path = experiment_dir + f'{idx}/'
    net = STDP_SR(num_states=2, gamma=0.4)
    train(
        save_path, net, datasets, datasets_config_ranges,
        train_steps=801, print_every_steps=25,
        early_stop=False, return_test_error=False
        )

Parallel(n_jobs=50)(delayed(grid_train)(arg) for arg in args)
