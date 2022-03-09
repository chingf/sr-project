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
from run_td_rnn import run as _run_rnn
from td_utils import run_models
from eval import eval
import matplotlib.pyplot as plt

def gradient_comparison(
    exp_path, lr,
    iters, gamma, dataset, dataset_config, transition_probs, ca3_kwargs,
    delete_dir=False
    ):

    if delete_dir:
        rmtree(exp_path, ignore_errors=True)
    os.makedirs(exp_path, exist_ok=True)
    criterion = nn.MSELoss()
    num_steps = dataset_config['num_steps']
    num_states = dataset_config['num_states']
    ca3_kwargs = deepcopy(ca3_kwargs)
    ca3_kwargs['lr'] = 1
    saved_grads = {}

    for _iter in range(iters):
        net = AnalyticSR( # Initialize network
            num_states=num_states, gamma=gamma, ca3_kwargs=ca3_kwargs
            )
    
        optim = torch.optim.Adam(net.parameters(), lr=lr)
   
        dataset_config['left_right_stay_prob'] = transition_probs[_iter%3]
        dset = dataset(**dataset_config)
        dg_inputs = torch.Tensor(dset.dg_inputs.T)
        dg_inputs = dg_inputs.unsqueeze(1)
        outputs = []
        if _iter not in saved_grads.keys():
            saved_grads[_iter] = {}
            saved_grads[_iter]['steps'] = []
            saved_grads[_iter]['rnn'] = []
            saved_grads[_iter]['oja'] = []
            saved_grads[_iter]['dset'] = dset
        for i in range(num_steps):
            phi_prime = dg_inputs[i].unsqueeze(0)
            _, psi_prime = net(
                phi_prime, reset=False, update=False, update_transition=True
                )
            outputs.append(psi_prime)
            if i == 0: continue
            phi = dg_inputs[i-1].unsqueeze(0)
   
            # Simplified loss
            response = torch.matmul(dg_inputs[i-1], net.ca3.T)
            target = phi_prime.squeeze(0)

            # Backwards pass
            optim.zero_grad()
            td_loss = criterion(response, target)
            td_loss.backward()
   
            # Get gradients
            rnn_grad = net.ca3.update(view_only=True)
            oja_grad = net.ca3.update(update_type='oja', view_only=True)

            # Record grads if needed
            if i % 100 == 0:
                saved_grads[_iter]['steps'].append(i)
                saved_grads[_iter]['rnn'].append(rnn_grad.detach().numpy())
                saved_grads[_iter]['oja'].append(oja_grad.detach().numpy())

            # Optimizer Step
            optim.step()

    with open(f'{exp_path}results.p', 'wb') as f:
        pickle.dump({'saved_grads': saved_grads}, f)

if __name__ == "__main__":
    # Save parameters
    exp_path = '../trained_models/03_oja_sf_gradient_vis/'

    # Dataset parameters
    iters = 3
    gamma = 0.5
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
    dset_params = {'dataset': dataset, 'dataset_config': dataset_config}
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    with open(exp_path + "dset_configs.p", 'wb') as f:
        pickle.dump(dset_params, f)

    # Network parameters
    ca3_kwargs = {
        'use_dynamic_lr':False, 'parameterize': False,
        'T_grad_on':True, 'rollout': int(np.log(1E-5)/np.log(gamma))
        }

    # Run scripts
    gradient_comparison(
        exp_path, (1E-2)/5,
        iters, gamma, dataset, dataset_config, transition_probs, ca3_kwargs,
        delete_dir=False
        )

