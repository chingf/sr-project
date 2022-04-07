import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
import argparse
from shutil import rmtree

from datasets import inputs, sf_inputs_discrete
from sr_model.models.models import AnalyticSR, STDP_SR, Linear, Hopfield
from run_td_rnn import run as run_rnn
from run_td_linear import run as run_linear

def run_models(
    save_path, iters, lr_range, dataset, dataset_config, gamma, input_size,
    save_outputs=False, test_over_all=True, print_file=None, run_hopfield=False,
    load_from_dir=None
    ):

    # Hopfield
    if run_hopfield: # Gamma doesn't matter for Hopfield so only run once
        print(f'Running {save_path} for Hopfield')
        net = Hopfield(input_size, lr=1E-3, clamp=np.inf)
        for _iter in range(iters):
            hopfield_save_path = save_path + f'hopfield/{_iter}'
            if os.path.isfile(f'{hopfield_save_path}/results.p'):
                print(f'{hopfield_save_path} already calculated. Skipping...')
                continue
            net.reset()
            dset = dataset(**dataset_config)
            dg_inputs = torch.from_numpy(dset.dg_inputs.T).float().to('cpu').unsqueeze(1)
            outputs = net(dg_inputs)
            if save_outputs:
                results = {'outputs': outputs, 'dset': dset}
                if not os.path.isdir(hopfield_save_path):
                    os.makedirs(hopfield_save_path)
                with open(f'{hopfield_save_path}/results.p', 'wb') as f:
                    pickle.dump(results, f)

    # RNN-SF
    if os.path.isfile(save_path + f'rnn/{iters-1}/results.p'):
        print(f'{save_path}/rnn already calculated. Skipping...')
    else:
        num_iters = int(np.log(1E-5)/np.log(gamma))
        ca3_kwargs = {
            'use_dynamic_lr':False, 'parameterize': True,
            'output_params':{'num_iterations': num_iters,'nonlinearity': 'clamp'}
            }
        net = AnalyticSR( # Initialize network
            num_states=input_size, gamma=gamma,
            ca3_kwargs=ca3_kwargs
            )
        net_loaded = False
        if load_from_dir is not None:
            load_from_path = load_from_dir + 'rnn/0/model.pt'
            if os.path.isfile(load_from_path):
                net.load_state_dict(torch.load(load_from_path))
            net_loaded = True
            print(f'Loading net from {load_from_dir}')
        if not net_loaded:
            _, _, _, net = run_rnn( # Meta-learn LR and nonlinearity
                save_path + 'test/', net, dataset, dataset_config, gamma=gamma,
                train_net=True, test_over_all=False
                )
        for _iter in range(iters):
            net.reset()
            rnn_save_path = save_path + f'rnn/{_iter}'
            if os.path.isfile(f'{rnn_save_path}/results.p'):
                print(f'{rnn_save_path} already calculated. Skipping...')
                continue
            try:
                outputs, _, dset, _ = run_rnn(
                    rnn_save_path, net, dataset, dataset_config, gamma=gamma,
                    train_net=False, test_over_all=False
                    )
            except RuntimeError as e:
                if 'svd' in str(e):
                    continue
                else:
                    raise
            if save_outputs:
                results = {'outputs': outputs, 'dset': dset}
                with open(f'{rnn_save_path}/results.p', 'wb') as f:
                    pickle.dump(results, f)

    # RNN-SF Oja
    best_net = None; best_lr_val = np.inf;
    if os.path.isfile(save_path + f'rnn_oja/{iters-1}/results.p'):
        print(f'{save_path}/rnn already calculated. Skipping...')
    else:
        num_iters = int(np.log(1E-5)/np.log(gamma))
        ca3_kwargs = {
            'use_dynamic_lr':False, 'parameterize': True,
            'output_params':{'num_iterations': num_iters,'nonlinearity': 'clamp'},
            'forget': 'oja'
            }
        net = AnalyticSR( # Initialize network
            num_states=input_size, gamma=gamma,
            ca3_kwargs=ca3_kwargs
            )

        net_loaded = False
        if load_from_dir is not None:
            load_from_path = load_from_dir + 'rnn_oja/0/model.pt'
            if os.path.isfile(load_from_path):
                net.load_state_dict(torch.load(load_from_path))
            net_loaded = True
            print(f'Loading net from {load_from_dir}')
        if not net_loaded:
            _, _, _, net = run_rnn( # Meta-learn LR and nonlinearity
                save_path + 'test/', net, dataset, dataset_config, gamma=gamma,
                train_net=True, test_over_all=False
                )
        for _iter in range(iters):
            net.reset()
            rnn_save_path = save_path + f'rnn_oja/{_iter}'
            if os.path.isfile(f'{rnn_save_path}/results.p'):
                print(f'{rnn_save_path} already calculated. Skipping...')
                continue
            try:
                outputs, _, dset, _ = run_rnn(
                    rnn_save_path, net, dataset, dataset_config, gamma=gamma,
                    train_net=False, test_over_all=False
                    )
            except RuntimeError as e:
                if 'svd' in str(e):
                    continue
                else:
                    raise
            if save_outputs:
                results = {'outputs': outputs, 'dset': dset}
                with open(f'{rnn_save_path}/results.p', 'wb') as f:
                    pickle.dump(results, f)

    # Linear
    print(f'Running {save_path} for Linear')
    best_lr = np.inf; best_lr_val = np.inf;
    net = Linear(input_size=input_size)
    for lr in lr_range:
        net.reset()
        _, loss, _, _ = run_linear(
            save_path + 'test/', net, dataset, dataset_config, gamma=gamma, lr=lr,
            test_over_all=test_over_all, print_file=print_file
            )
        if loss < best_lr_val:
            best_lr = lr; best_lr_val = loss;
    for _iter in range(iters):
        net.reset()
        linear_save_path = save_path + f'linear/{_iter}'
        outputs, _, dset, _ = run_linear(
            linear_save_path, net, dataset, dataset_config, lr=best_lr, gamma=gamma,
            test_over_all=test_over_all, print_file=print_file
            )
        if save_outputs:
            results = {'outputs': outputs, 'dset': dset}
            with open(f'{linear_save_path}/results.p', 'wb') as f:
                pickle.dump(results, f)
   
    # MLP
    print(f'Running {save_path} for MLP')
    net = MLP(input_size=input_size, hidden_size=input_size*2)
    for _iter in range(iters):
        net.reset()
        mlp_save_path = save_path + f'mlp/{_iter}'
        outputs, _, dset,_ = run_mlp(
            mlp_save_path, net, dataset, dataset_config, gamma=gamma,
            test_over_all=test_over_all, print_file=print_file
            )
        if save_outputs:
            results = {'outputs': outputs, 'dset': dset}
            with open(f'{mlp_save_path}/results.p', 'wb') as f:
                pickle.dump(results, f)

