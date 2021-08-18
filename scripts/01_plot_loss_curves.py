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

datasets = [inputs.Sim1DWalk]
datasets_config_ranges = [
    {
    'num_steps': [3, 10, 20, 30],
    'left_right_stay_prob': [[1, 1, 1], [7, 1, 1], [1, 4, 1]],
    'num_states': [5, 10, 15, 25, 36]
    },
    ]

for idx in range(15):
    save_path = experiment_dir + f'{idx}/'
    net = STDP_SR(num_states=2, gamma=0.4)
    train(
        save_path, net, datasets, datasets_config_ranges,
        train_steps=601, print_every_steps=50
        )


