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
    exp_path, exp_name, lr,
    iters, gamma, dataset, dataset_config, transition_probs, ca3_kwargs,
    save_grads=False, delete_dir=False, use_td_loss=False
    ):

    if delete_dir:
        rmtree(exp_path, ignore_errors=True)
    os.makedirs(exp_path, exist_ok=True)
    criterion = nn.MSELoss()
    num_steps = dataset_config['num_steps']
    num_states = dataset_config['num_states']
    ca3_kwargs = deepcopy(ca3_kwargs)
    ca3_kwargs['lr'] = 1

    dist = []
    step = []
    model = []
    ojasf_dist = []
    ojasf_step = []
    test_losses = []
    saved_grads = {}

    for _iter in range(iters):
        try:
            net = AnalyticSR( # Initialize network
                num_states=num_states, gamma=gamma, ca3_kwargs=ca3_kwargs
                )
        
            optim = torch.optim.Adam(net.parameters(), lr=lr)
       
            dataset_config['left_right_stay_prob'] = transition_probs[_iter%3]
            dset = dataset(**dataset_config)
            dg_inputs = torch.Tensor(dset.dg_inputs.T)
            dg_inputs = dg_inputs.unsqueeze(1)
            outputs = []
            for i in range(num_steps):
                phi_prime = dg_inputs[i].unsqueeze(0)
                _, psi_prime = net(
                    phi_prime, reset=False, update=False, update_transition=True
                    )
                outputs.append(psi_prime)
                if i == 0: continue
                phi = dg_inputs[i-1].unsqueeze(0)
       
                if use_td_loss:
                    # Gather terms for TD Loss
                    _, psi = net(
                        phi, reset=False, update=False, update_transition=False
                        )
                    response = psi
                    target = phi.squeeze(0) + net.gamma*psi_prime
                    target = target #.detach()
                else:
                    # Simplified loss
                    response = torch.matmul(dg_inputs[i-1], net.ca3.T)
                    target = phi_prime.squeeze(0)

                # Backwards pass
                optim.zero_grad()
                td_loss = criterion(response, target)
                td_loss.backward()
       
                # Get gradients
                bp_grad = net.ca3.T.grad
                rnn_grad = net.ca3.update(view_only=True)
                oja_grad = net.ca3.update(update_type='oja', view_only=True)

                # Get cosine similarities
                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                _rnn_dist = cos(bp_grad.view(-1,1), rnn_grad.view(-1,1))
                _oja_dist = cos(bp_grad.view(-1,1), oja_grad.view(-1,1))
                _shuffle_dist = cos(
                    bp_grad.view(-1,1),
                    bp_grad.view(-1,1)[torch.randperm(num_states**2)]
                    )
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

                # Record grads if needed
                if save_grads and _iter > iters-4:
                    if _iter not in saved_grads.keys():
                        saved_grads[_iter] = {}
                        saved_grads[_iter]['steps'] = []
                        saved_grads[_iter]['bp'] = []
                        saved_grads[_iter]['rnn'] = []
                        saved_grads[_iter]['oja'] = []
                    if i % 10 == 0:
                        saved_grads[_iter]['steps'].append(i)
                        saved_grads[_iter]['bp'].append(bp_grad.detach().numpy())
                        saved_grads[_iter]['rnn'].append(rnn_grad.detach().numpy())
                        saved_grads[_iter]['oja'].append(oja_grad.detach().numpy())

                # Optimizer Step
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
    with open(f'{exp_path}{exp_name}.p', 'wb') as f:
        pickle.dump({
            'autodiff_dist':autodiff_dist, 'ojasf_dist': ojasf_dist,
            'test_losses': test_losses, 'saved_grads': saved_grads
            }, f)
    print(np.mean([x.item() for x in test_losses]))

def rnn_baseline(
    exp_path, exp_name, forget, lr,
    iters, gamma, dataset, dataset_config, transition_probs, ca3_kwargs
    ):

    criterion = nn.MSELoss()
    num_steps = dataset_config['num_steps']
    num_states = dataset_config['num_states']
    ca3_kwargs = deepcopy(ca3_kwargs)
    ca3_kwargs['forget'] = forget
    ca3_kwargs['lr'] = lr
    ca3_kwargs['T_grad_on'] = False

    test_losses = []

    for _iter in range(iters):
        net = AnalyticSR( # Initialize network
            num_states=num_states, gamma=gamma, ca3_kwargs=ca3_kwargs
            )

        dataset_config['left_right_stay_prob'] = transition_probs[_iter%3]
        dset = dataset(**dataset_config)
        dg_inputs = torch.Tensor(dset.dg_inputs.T)
        dg_inputs = dg_inputs.unsqueeze(1)
        outputs = []
        for i in range(num_steps):
            dg_input = dg_inputs[i]
            dg_input = dg_input.unsqueeze(0)
            _, out = net(
                dg_input, reset=False, update=True, update_transition=True
                )
            outputs.append(out)
            if i == 0: continue

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
        test_losses.append(test_loss.item())

    with open(f'{exp_path}{exp_name}.p', 'wb') as f:
        pickle.dump({
            'test_losses': test_losses
            }, f)
    print(np.mean(test_losses))
    print(np.std(test_losses))

if __name__ == "__main__":
    # Save parameters
    exp_path = '../trained_models/03_oja_sf_gradients/'
    exp_name = 'simplif_gradient_update'
    use_td_loss = False

    # Dataset parameters
    iters = 100
    gamma = 0.5
    num_states = 25
    num_steps = 251
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
    with open(exp_path + "dset_configs.p", 'wb') as f:
        pickle.dump(dset_params, f)

    # Network parameters
    ca3_kwargs = {
        'use_dynamic_lr':False, 'parameterize': False,
        'T_grad_on':True, 'rollout': int(np.log(1E-5)/np.log(gamma))
        }

    # Run scripts
    gradient_comparison(
        exp_path, exp_name, (1E-2)/5
        iters, gamma, dataset, dataset_config, transition_probs, ca3_kwargs,
        delete_dir=False, use_td_loss=use_td_loss, save_grads=True
        )
    rnn_baseline(
        exp_path, 'rnn_baseline', None, 1E-2,
        iters, gamma, dataset, dataset_config, transition_probs, ca3_kwargs
        )
    rnn_baseline(
        exp_path, 'oja_baseline', 'oja', 1E-3,
        iters, gamma, dataset, dataset_config, transition_probs, ca3_kwargs
        )

