import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from datasets import sf_inputs
from sr_model.models.models import AnalyticSR, STDP_SR, Linear, MLP
from train_td_rnn import train as train_rnn
from train_td_mlp import train as train_mlp
from train_td_linear import train as train_linear

save_path = '../trained_models/03_nhot_td_loss/'
dataset = sf_inputs.Sim1DWalk
dataset_config = {
    'num_steps': 8000, 'left_right_stay_prob': [1,1,1], 'num_states': 10,
    'feature_dim': 20
    }

# Analytic RNN 
rnn_save_path = save_path + 'rnn/'
net = AnalyticSR(num_states=dataset_config['feature_dim'], gamma=0.4)
train_rnn(rnn_save_path, net, dataset, dataset_config)

# RNN 
rnn_save_path = save_path + 'rnn/'
net = STDP_SR(
    num_states=dataset_config['feature_dim'], gamma=0.4,
    ca3_kwargs={'gamma_T':1}
    )
net.ca3.set_differentiability(False)
state_dict_path = '../trained_models/baseline/model.pt'
net.load_state_dict(torch.load(state_dict_path))
train_rnn(rnn_save_path, net, dataset, dataset_config)

# Linear
linear_save_path = save_path + 'linear/'
net = Linear(input_size=dataset_config['feature_dim'])
train_linear(
    linear_save_path, net, dataset, dataset_config, buffer_batch_size=1,
    lr=1E-1
    )

# MLP
mlp_save_path = save_path + 'mlp/'
net = MLP(
    input_size=dataset_config['feature_dim'],
    hidden_size=dataset_config['feature_dim']*2)
train_mlp(mlp_save_path, net, dataset, dataset_config)

