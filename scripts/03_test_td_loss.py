import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
import argparse

from datasets import inputs, sf_inputs
from sr_model.models.models import AnalyticSR, STDP_SR, Linear, MLP
from train_td_rnn import train as train_rnn
from train_td_mlp import train as train_mlp
from train_td_linear import train as train_linear

parser = argparse.ArgumentParser(description='Choose model to run.')
parser.add_argument('model', metavar='M', type=str, nargs='+',
                    help='model type')
args = parser.parse_args()
model_type = args.model[0]

save_path = '../trained_models/03_nhot_td_loss/'
dataset = sf_inputs.Sim2DWalk
dataset_config = {
    'num_steps': 8000, 'num_states': 25,
    'feature_dim': 30
    }
gamma=0.4
input_size = dataset_config['feature_dim']

# Analytic RNN 
if model_type == 'analytic':
    rnn_save_path = save_path + 'rnn/'
    net = AnalyticSR(num_states=input_size, gamma=gamma)
    train_rnn(rnn_save_path, net, dataset, dataset_config, gamma=gamma)

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
    train_rnn(rnn_save_path, net, dataset, dataset_config, gamma=gamma)

# Linear
if model_type == 'linear':
    linear_save_path = save_path + 'linear/'
    net = Linear(input_size=input_size)
    train_linear(
        linear_save_path, net, dataset, dataset_config, buffer_batch_size=1,
        lr=1E-1, gamma=gamma
        )

# MLP
if model_type == 'mlp':
    mlp_save_path = save_path + 'mlp/'
    net = MLP(
        input_size=input_size,
        hidden_size=input_size*2)
    train_mlp(mlp_save_path, net, dataset, dataset_config, gamma=gamma)

