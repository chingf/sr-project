import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
import numpy as np
import seaborn as sns
import pandas as pd
from copy import deepcopy
import torch
import torch.nn as nn
from joblib import Parallel, delayed
import argparse
from shutil import rmtree

from datasets import inputs, sf_inputs_discrete
from sr_model.models.models import AnalyticSR, STDP_SR, Linear, MLP
from run_td_rnn import run as run_rnn
from eval import eval
import matplotlib.pyplot as plt

# Save parameters
exp_path = '../trained_models/05_oja_sf_loss/'
os.makedirs(exp_path, exist_ok=True)

# Simulation parameters
n_jobs = 7
iters = 100
gamma = 0.5
args = np.arange(iters)

# Dataset parameters
num_states = 25
num_steps = 4001
dataset = sf_inputs_discrete.Sim1DWalk
sprs = 0.03
sig = 2.0
feature_maker_kwargs = {
    'feature_dim': num_states,
    'feature_type': 'correlated_distributed',
    'feature_vals_p': [1-sprs, sprs], 'feature_vals': None,
    'spatial_sigma': sig
    }
dataset_config = {
    'num_steps': num_steps, 'num_states': num_states,
    'feature_maker_kwargs': feature_maker_kwargs
    }
transition_probs = [[1,1,1], [7,1,1], [1,1,7]]

# Network parameters
ca3_kwargs = {
    'use_dynamic_lr':False, 'parameterize': False
    }
rnn_lr = 1E-2
oja_lrs = [1E-4, 5E-4, 1E-3]

def main():
    Parallel(n_jobs=n_jobs)(delayed(grid_train)(arg) for arg in args)

def grid_train(_iter):
    if _iter != 0: return
    # RNN-SF
    rnnsf_ca3_kwargs = deepcopy(ca3_kwargs)
    rnnsf_ca3_kwargs['lr'] = rnn_lr
    net = AnalyticSR( # Initialize network
        num_states=num_states, gamma=gamma,
        ca3_kwargs=rnnsf_ca3_kwargs
        )
    rnn_save_path = exp_path + f'rnn/{_iter}/'
    dataset_config['left_right_stay_prob'] = transition_probs[_iter%3]
    outputs, _, dset, _ = run_rnn(
        rnn_save_path, net, dataset, dataset_config, gamma=gamma,
        train_net=False, test_over_all=False,
        print_every_steps=5,
        )
    if _iter == 0:
        with open(exp_path + f'rnn/output_example.p', 'wb') as f:
            results = {'outputs': outputs, 'dset': dset}
            pickle.dump(results, f)

    # RNN-SR
#    save_path = '../trained_models/baseline/'
#    model_path = save_path + 'model.pt'
#    net_configs_path = save_path + 'net_configs.p'
#    with open(net_configs_path, 'rb') as f:
#        net_configs = pickle.load(f)
#    net_configs.pop('num_states')
#    net = STDP_SR(num_states=num_states, **net_configs)
#    net.load_state_dict(torch.load(model_path))
#    sr_save_path = exp_path + f'sr/{_iter}/'
#    dataset_config['left_right_stay_prob'] = transition_probs[_iter%3]
#    outputs, _, dset, _ = run_rnn(
#        sr_save_path, net, dataset, dataset_config, gamma=gamma,
#        train_net=False, test_over_all=False,
#        print_every_steps=5,
#        )

    # RNN-SF Oja
    for oja_lr in oja_lrs:
        rnnsf_ca3_kwargs = deepcopy(ca3_kwargs)
        rnnsf_ca3_kwargs['lr'] = oja_lr
        rnnsf_ca3_kwargs['forget'] = 'oja'
        net = AnalyticSR( # Initialize network
            num_states=num_states, gamma=gamma,
            ca3_kwargs=rnnsf_ca3_kwargs
            )
        rnn_save_path = exp_path + f'oja_{oja_lr}/{_iter}/'
        dataset_config['left_right_stay_prob'] = transition_probs[_iter%3]
        outputs, _, dset, _ = run_rnn(
            rnn_save_path, net, dataset, dataset_config, gamma=gamma,
            print_every_steps=5,
            train_net=False, test_over_all=False
            )
        if _iter == 0:
            with open(exp_path + f'oja_{oja_lr}/output_example.p', 'wb') as f:
                results = {'outputs': outputs, 'dset': dset}
                pickle.dump(results, f)

if __name__ == "__main__":
    main()
