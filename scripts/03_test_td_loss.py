import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
import argparse

from datasets import inputs, sf_inputs_discrete
from sr_model.models.models import AnalyticSR, STDP_SR, Linear, MLP
from run_td_rnn import run as run_rnn
from run_td_mlp import run as run_mlp
from run_td_linear import run as run_linear

parser = argparse.ArgumentParser(description='Choose model to run.')
parser.add_argument('model', metavar='M', type=str, nargs='+',
                    help='model type')
args = parser.parse_args()
model_type = args.model[0]

save_path = '../trained_models/03_test_td_loss/'
dataset = sf_inputs_discrete.Sim2DWalk
feature_maker_kwargs = {
    'feature_dim': 64, 'feature_type': 'correlated_sparse',
    'feature_vals': [0, 1.], 'spatial_sigma':3
    }
dataset_config = {
    'num_steps': 8000, 'num_states': 64,
    'feature_maker_kwargs': feature_maker_kwargs
    }
gamma=0.4
input_size = feature_maker_kwargs['feature_dim']

# Analytic RNN 
if model_type == 'analytic':
    rnn_save_path = save_path + 'rnn/'
    net = AnalyticSR(
        num_states=input_size, gamma=gamma,
        ca3_kwargs={'use_dynamic_lr': False, 'lr': 1E-3}
        )
    run_rnn(rnn_save_path, net, dataset, dataset_config, gamma=gamma)

# RNN
if model_type == 'stdp':
    rnn_save_path = save_path + 'rnn/'
    net = STDP_SR(
        num_states=input_size, gamma=gamma,
        ca3_kwargs={'gamma_T':1}
        )
    net.ca3.set_differentiability(False)
    state_dict_path = '../trained_models/baseline/model.pt'
    net.load_state_dict(torch.load(state_dict_path))
    run_rnn(rnn_save_path, net, dataset, dataset_config, gamma=gamma)

# Linear
if model_type == 'linear':
    linear_save_path = save_path + 'linear/'
    net = Linear(input_size=input_size)
    run_linear(
        linear_save_path, net, dataset, dataset_config, buffer_batch_size=1,
        lr=1E-2, gamma=gamma
        )

# MLP
if model_type == 'mlp':
    mlp_save_path = save_path + 'mlp/'
    net = MLP(
        input_size=input_size,
        hidden_size=input_size*2)
    run_mlp(mlp_save_path, net, dataset, dataset_config, gamma=gamma)

