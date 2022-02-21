import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
import numpy as np
import seaborn as sns
import pandas as pd
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
torch.autograd.set_detect_anomaly(True)
def main(delete_dir=False):
    experiment_path = 'trained_models/03_oja_sf_gradients/'
    if delete_dir:
        rmtree(experiment_path, ignore_errors=True)
    os.makedirs(experiment_path)
    
    iters = 100
    gamma = 0.6
    models = ['rnn_sf']
    lr_range = [1E-2]
    num_states = 25
    num_steps = 251
    dataset = sf_inputs_discrete.Sim1DWalk
    sprs = 0.03
    sig = 2.5
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
    input_size = num_states
    dset_params = {'dataset': dataset, 'dataset_config': dataset_config}
    with open(experiment_path + "dset_configs.p", 'wb') as f:
        pickle.dump(dset_params, f)
    
    ca3_kwargs = {
        'use_dynamic_lr':False, 'parameterize': False, 'lr': lr_range[0],
        'T_grad_on':True
        }

    dist = []
    step = []
    model = []
    completed_iters = 0

    for _ in range(iters):
        try:
            net = AnalyticSR( # Initialize network
                num_states=input_size, gamma=gamma, ca3_kwargs=ca3_kwargs
                )
        
            criterion = nn.MSELoss()
            optim = torch.optim.Adam(net.parameters(), lr=0.001)
        
            dset = dataset(**dataset_config)
            dg_inputs = torch.Tensor(dset.dg_inputs.T)
            dg_inputs = dg_inputs.unsqueeze(1)
            outputs = []
            for i in range(num_steps):
                dg_input = dg_inputs[i]
                dg_input = dg_input.unsqueeze(0)
                _, out = net(
                    dg_input, reset=False, update=False, update_transition=True
                    )
                outputs.append(out)
                if i == 0: continue
        
                # Calculate TD Loss
                psi = outputs[-2]
                psi_prime = outputs[-1]
                phi_prime = dg_input
                response = psi.unsqueeze(1)
                target = phi_prime + net.gamma*psi_prime
                td_loss = criterion(response, target.detach())
        
                # Backwards pass
                optim.zero_grad()
                td_loss.backward()
        
                # Optimizer step
                bp_grad = net.ca3.T.grad
                rnn_grad = net.ca3.update(view_only=True)
                oja_grad = net.ca3.update(update_type='oja', view_only=True)
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                _rnn_dist = cos(bp_grad.view(-1,1), rnn_grad.view(-1,1))
                _oja_dist = cos(bp_grad.view(-1,1), oja_grad.view(-1,1))
                _shuffle_dist = cos(bp_grad.view(-1,1), bp_grad.view(-1,1)[torch.randperm(625)])
    
                dist.append(_rnn_dist.item()*-1)
                step.append(i)
                model.append('RNN')
                dist.append(_oja_dist.item()*-1)
                step.append(i)
                model.append('Oja')
                dist.append(_shuffle_dist.item())
                step.append(i)
                model.append('Shuffle')
                optim.step()
        except:
            continue

        idxs = np.random.choice(num_steps-1, size=32)
        all_s = torch.stack([dg_inputs[idx] for idx in idxs], dim=2).squeeze(0)
        all_next_s = torch.stack([dg_inputs[idx+1] for idx in idxs], dim=2).squeeze(0)

        phi = all_s
        psi_s = torch.stack(
            [net(s.view(1,1,-1), reset=False, update=False)[1].squeeze()\
                for s in all_s.t()]
            ).unsqueeze(0)
        psi_s_prime = torch.stack(
            [net(s.view(1,1,-1), reset=False, update=False)[1].squeeze()\
                for s in all_next_s.t()]
            ).unsqueeze(0)
        value_function = psi_s
        exp_value_function = phi.t().unsqueeze(0) + gamma*psi_s_prime
        test_loss = criterion(value_function, exp_value_function)
        print(test_loss)

    df = pd.DataFrame({'Distance': dist, 'Model': model, 'Step': step})
    plt.figure()
    sns.lineplot(x='Step', y='Distance', hue='Model', data=df)
    plt.show()
    plt.savefig(f'gradient_alignment_sprs{sprs}_sig{sig}_gamma{gamma}.png', dpi=300)

if __name__ == "__main__":
    main(delete_dir=True)

