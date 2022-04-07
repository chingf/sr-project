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
from sr_model.models.models import AnalyticSR, STDP_SR, Linear
from run_td_rnn import run as _run_rnn
from td_utils import run_models
from eval import eval
import matplotlib.pyplot as plt

# Save parameters
np.random.seed(0)
exp_path = '../trained_models/03_stability_vis/'
gamma = 0.5
num_states = 25
num_steps = 3001
sprs = 0.03
sig = 2.0
dataset = sf_inputs_discrete.Sim1DWalk
feature_maker_kwargs = {
    'feature_dim': num_states,
    'feature_type': 'correlated_distributed',
    'feature_vals_p': [1-sprs, sprs], 'feature_vals': None,
    'spatial_sigma': sig
    }
dataset_config = {
    'num_steps': num_steps, 'num_states': num_states,
    'feature_maker_kwargs': feature_maker_kwargs,
    'left_right_stay_prob': [1,3,1]
    }
dset = dataset(**dataset_config)
dg_inputs = torch.Tensor(dset.dg_inputs.T)
dg_inputs = dg_inputs.unsqueeze(1)
n_jobs = 4
os.makedirs(exp_path, exist_ok=True)


# Args
args = [
    (AnalyticSR(num_states=num_states, gamma=gamma, ca3_kwargs={'lr':1E-3}), 'SF_1E-3'),
    (AnalyticSR(num_states=num_states, gamma=gamma, ca3_kwargs={'lr':1E-3, 'forget':'oja'}), 'Oja_1E-3'),
    (AnalyticSR(num_states=num_states, gamma=gamma, ca3_kwargs={'lr':1E-2}), 'SF_1E-2'),
    (AnalyticSR(num_states=num_states, gamma=gamma, ca3_kwargs={'lr':1E-2, 'forget':'oja'}), 'Oja_1E-2'),
    ]

def grid(arg):
    net, netname = arg
    _, outputs = net(dg_inputs)
    with open(f'{exp_path}{netname}.p', 'wb') as f:
        pickle.dump({'outputs': outputs, 'dset': dset}, f)

job_results = Parallel(n_jobs=n_jobs)(delayed(grid)(arg) for arg in args)

