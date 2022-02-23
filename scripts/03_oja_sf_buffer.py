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
import random
from collections import namedtuple, deque

from datasets import inputs, sf_inputs_discrete
from sr_model.models.models import AnalyticSR, STDP_SR, Linear, MLP
from run_td_rnn import run as _run_rnn
from td_utils import run_models
from eval import eval
import matplotlib.pyplot as plt

def main(delete_dir=False):
    experiment_path = '../trained_models/03_oja_sf_buffer/'
    if delete_dir:
        rmtree(experiment_path, ignore_errors=True)
    os.makedirs(experiment_path)
    
    iters = 50
    gamma = 0.5
    models = ['rnn_sf']
    lr = 1E-2
    num_states = 25
    num_steps = 251
    dataset = sf_inputs_discrete.Sim1DWalk
    sprs = 0.03 #0.03
    sig = 2.0 #2.5
    buffer = ReplayMemory(5000)
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
    input_size = num_states
    dset_params = {'dataset': dataset, 'dataset_config': dataset_config}
    with open(experiment_path + "dset_configs.p", 'wb') as f:
        pickle.dump(dset_params, f)
    
    ca3_kwargs = {
        'use_dynamic_lr':False, 'parameterize': False, 'lr': lr,
        'T_grad_on':True, 'rollout': int(np.log(1E-5)/np.log(gamma))
        }

    dist = []
    step = []
    model = []
    ojasf_dist = []
    ojasf_step = []
    test_losses = []

    for _iter in range(iters):
        try:
            net = AnalyticSR( # Initialize network
                num_states=input_size, gamma=gamma, ca3_kwargs=ca3_kwargs
                )
        
            criterion = nn.MSELoss()
            optim = torch.optim.Adam(net.parameters(), lr=lr)
       
            dataset_config['left_right_stay_prob'] = transition_probs[_iter%3]
            dset = dataset(**dataset_config)
            dg_inputs = torch.Tensor(dset.dg_inputs.T)
            dg_inputs = dg_inputs.unsqueeze(1)
            outputs = []
            for i in range(num_steps):
                _, out = net(
                    dg_inputs[i].unsqueeze(0),
                    reset=False, update=False, update_transition=True
                    )
                outputs.append(out)
                if i == 0: continue
                buffer.push((dg_inputs[i-1].unsqueeze(0), dg_inputs[i].unsqueeze(0)))
                transitions = buffer.sample(min(i, 32))
                phis = torch.stack([t[0] for t in transitions], dim=2).squeeze(0)
                phi_primes = torch.stack([t[1] for t in transitions], dim=2).squeeze(0)

                # Gather terms for TD Loss
                psis = torch.stack(
                    [net(phi.view(1,1,-1), reset=False, update=False)[1].squeeze()\
                        for phi in phis.squeeze(0)]
                    ).unsqueeze(0)
                psi_primes = torch.stack(
                    [net(phi.view(1,1,-1), reset=False, update=False)[1].squeeze()\
                        for phi in phi_primes.squeeze(0)]
                    ).unsqueeze(0)
                response = psis
                target = phis.squeeze(1) + net.gamma*psi_primes
                target = target.detach()
                retain_graph = False

                # Simplified loss
#                response = torch.stack([torch.matmul(phi, net.ca3.T)
#                    for phi in phis.squeeze(0)
#                    ])
#                target = phi_primes.squeeze(0)
#                retain_graph = False

                # Backwards pass
                optim.zero_grad()
                td_loss = criterion(response, target)
                td_loss.backward(retain_graph=retain_graph)
        
                # Optimizer step
                bp_grad = net.ca3.T.grad
                rnn_grad = net.ca3.update(view_only=True)
                oja_grad = net.ca3.update(update_type='oja', view_only=True)
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                _rnn_dist = cos(bp_grad.view(-1,1), rnn_grad.view(-1,1))
                _oja_dist = cos(bp_grad.view(-1,1), oja_grad.view(-1,1))
                _shuffle_dist = cos(bp_grad.view(-1,1), bp_grad.view(-1,1)[torch.randperm(625)])
                _ojasf_dist = cos(oja_grad.view(-1,1), rnn_grad.view(-1,1))
    
                dist.append(_rnn_dist.item()*-1)
                step.append(i)
                model.append('RNN')
                dist.append(_oja_dist.item()*-1)
                step.append(i)
                model.append('Oja')
                dist.append(_shuffle_dist.item())
                step.append(i)
                model.append('Shuffle')
                ojasf_dist.append(_ojasf_dist.item())
                ojasf_step.append(i)
                optim.step()
    
            net.ca3.add_perturb = False
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
            test_losses.append(test_loss)
        except Exception as inst:
            import traceback
            traceback.print_exc()

    autodiff_dist = pd.DataFrame({'Distance': dist, 'Model': model, 'Step': step})
    ojasf_dist = pd.DataFrame({'Distance': ojasf_dist, 'Step': ojasf_step})
    with open('trained_models/gradient_dists.p', 'wb') as f:
        pickle.dump({
            'autodiff_dist':autodiff_dist, 'ojasf_dist': ojasf_dist,
            'test_losses': test_losses
            }, f)
    print(np.mean([x.item() for x in test_losses]))
    plt.figure()
    sns.lineplot(x='Step', y='Distance', hue='Model', data=autodiff_dist)
    plt.show()
    plt.savefig(f'gradient_alignment_sprs{sprs}_sig{sig}_gamma{gamma}.png', dpi=300)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, _list):
        """Save a transition _list is (state, next_state)"""
        self.memory.append(_list)

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        samples.append(self.memory[-1])
        return samples

    def __len__(self):
        return len(self.memory)

if __name__ == "__main__":
    main(delete_dir=True)

