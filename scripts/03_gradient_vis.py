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
exp_path = '../trained_models/03_oja_sf_gradient_vis/'
gamma = 0.5
dataset = sf_inputs_discrete.Sim1DWalk
num_states = 25
num_steps = 3001
n_jobs = 5
iters = 25
os.makedirs(exp_path, exist_ok=True)

# Args
args = [
    (0.01, 1.0, 1E-3),
    (0.01, 2.0, 1E-3),
    (0.03, 1.0, 1E-3),
    (0.03, 2.0, 1E-3),
    (None, None, 1E-2)
    ]

def grid(arg):
    # Dataset
    sprs, sig, lr = arg
    onehot = sprs is None
    if onehot:
        dataset = inputs.Sim1DWalk
        dataset_config = {
            'num_steps': num_steps, 'num_states': num_states,
            'left_right_stay_prob': [4,1,1]
            }
    else:
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
            'left_right_stay_prob': [4,1,1]
            }
    saved_grads = {}

    for _iter in range(iters):
        # Initialize dataset
        dset = dataset(**dataset_config)
        dg_inputs = torch.Tensor(dset.dg_inputs.T)
        dg_inputs = dg_inputs.unsqueeze(1)
    
        # Network
        net = AnalyticSR( # Initialize network
            num_states=num_states, gamma=gamma, ca3_kwargs={'lr':lr}
            )
        ojanet = AnalyticSR( # Initialize network
            num_states=num_states, gamma=gamma, ca3_kwargs={'lr':lr}
            )
        saved_grads[_iter] = {}
        saved_grads[_iter]['steps'] = []
        saved_grads[_iter]['update'] = []
        saved_grads[_iter]['forget'] = []
        saved_grads[_iter]['oja_update'] = []
        saved_grads[_iter]['oja_forget'] = []
        saved_grads[_iter]['steps_mat'] = []
        saved_grads[_iter]['update_mat'] = []
        saved_grads[_iter]['forget_mat'] = []
        for i in range(num_steps):
            phi_prime = dg_inputs[i].unsqueeze(0)
            _, psi_prime = net(
                phi_prime, reset=False, update=False, update_transition=True
                )
            _, psi_prime = ojanet(
                phi_prime, reset=False, update=False, update_transition=True
                )
            if i == 0: continue
            # Get gradients
            oja_update, oja_forget = ojanet.ca3.update(update_type='oja')
            grad_update, grad_forget = net.ca3.update()

            # Record grads if needed
            if i % 100 == 0:
                saved_grads[_iter]['steps'].append(i)
                saved_grads[_iter]['update'].append(
                    np.mean(grad_update.detach().numpy())
                    )
                saved_grads[_iter]['forget'].append(
                    np.mean(grad_forget.detach().numpy())
                    )
                saved_grads[_iter]['oja_update'].append(
                    np.mean(oja_update.detach().numpy())
                    )
                saved_grads[_iter]['oja_forget'].append(
                    np.mean(oja_forget.detach().numpy())
                    )
                if _iter == 0:
                    saved_grads[_iter]['steps_mat'].append(i)
                    saved_grads[_iter]['update_mat'].append(
                        grad_update.detach().numpy()
                        )
                    saved_grads[_iter]['forget_mat'].append(
                        grad_forget.detach().numpy()
                        )
                    if i == 3000:
                        saved_grads[_iter]['J_mat'] = net.get_T()
                        saved_grads[_iter]['oja_J_mat'] = ojanet.get_T()

    with open(f'{exp_path}{sprs}_{sig}.p', 'wb') as f:
        pickle.dump({'saved_grads': saved_grads}, f)

job_results = Parallel(n_jobs=n_jobs)(delayed(grid)(arg) for arg in args)

