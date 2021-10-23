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
from sr_model.models.models import AnalyticSR, STDP_SR, Linear, MLP, Hopfield
from run_td_rnn import run as run_rnn
from run_td_mlp import run as run_mlp
from run_td_linear import run as run_linear

def run_models(
    save_path, iters, lr_range, dataset, dataset_config, gamma, input_size,
    save_outputs=False, test_over_all=True
    ):

    # Hopfield
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

    # Analytic RNN with fixed LR and no alpha/beta
    best_net = None; best_lr_val = np.inf;
    for lr in lr_range:
        net = AnalyticSR(
            num_states=input_size, gamma=gamma,
            ca3_kwargs={'use_dynamic_lr':False, 'lr': lr}
            )
        _, loss = run_rnn(
            save_path + 'test/', net, dataset, dataset_config, gamma=gamma
            )
        if loss < best_lr_val:
            best_net = net; best_lr_val = loss;
    for _iter in range(iters):
        best_net.reset()
        rnn_save_path = save_path + f'rnn_fixedlr/{_iter}'
        outputs, _, dset = run_rnn(
            rnn_save_path, best_net, dataset, dataset_config, gamma=gamma,
            return_dset=True
            )
        if save_outputs:
            results = {'outputs': outputs, 'dset': dset}
            with open(f'{rnn_save_path}/results.p', 'wb') as f:
                pickle.dump(results, f)

    # Analytic RNN with fixed LR and alpha/beta scaling
    best_net = None; best_lr_val = np.inf;
    if os.path.isfile(save_path + f'rnn_fixedlr_alpha/{iters-1}/results.p'):
        print(f'{save_path}/rnn_fixed_lr_alpha already calculated. Skipping...')
        return
    for lr in lr_range:
        net = AnalyticSR(
            num_states=input_size, gamma=gamma,
            ca3_kwargs={'use_dynamic_lr':False, 'lr': lr}
            )
        train_net = False
        try:
            _, loss = run_rnn(
                save_path + 'test/', net, dataset, dataset_config, gamma=gamma,
                train_net=False, test_over_all=False
                )
        except RuntimeError as e:
            if 'svd' in str(e):
                continue
            else:
                raise
        if loss < best_lr_val:
            best_net = net; best_lr_val = loss;
    for _iter in range(iters):
        best_net.reset()
        rnn_save_path = save_path + f'rnn_fixedlr_alpha/{_iter}'
        if os.path.isfile(f'{rnn_save_path}/results.p'):
            print(f'{rnn_save_path} already calculated. Skipping...')
            continue
        try:
            outputs, _, dset = run_rnn(
                rnn_save_path, best_net, dataset, dataset_config, gamma=gamma,
                return_dset=True, test_over_all=False
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

    # Analytic RNN with dynamic LR and alpha/beta scaling
    alpha = best_net.ca3.alpha # Get alpha from previous grid search
    best_net = None; best_lr_val = np.inf;
    for lr in lr_range:
        net = AnalyticSR(
            num_states=input_size, gamma=gamma,
            ca3_kwargs={'use_dynamic_lr':True,
            'lr': lr, 'alpha': alpha, 'beta': alpha}
            )
        _, loss = run_rnn(
            save_path + 'test/', net, dataset, dataset_config, gamma=gamma
            )
        if loss < best_lr_val:
            best_net = net; best_lr_val = loss;
    for _iter in range(iters):
        best_net.reset()
        rnn_save_path = save_path + f'rnn_dynamiclr/{_iter}'
        outputs, _, dset = run_rnn(
            rnn_save_path, best_net, dataset, dataset_config, gamma=gamma,
            return_dset=True
            )
        if save_outputs:
            results = {'outputs': outputs, 'dset': dset}
            with open(f'{rnn_save_path}/results.p', 'wb') as f:
                pickle.dump(results, f)

    # Linear
    best_lr = np.inf; best_lr_val = np.inf;
    net = Linear(input_size=input_size)
    for lr in lr_range:
        net.reset()
        _, loss = run_linear(
            save_path + 'test/', net, dataset, dataset_config, gamma=gamma, lr=lr
            )
        if loss < best_lr_val:
            best_lr = lr; best_lr_val = loss;
    for _iter in range(iters):
        net.reset()
        linear_save_path = save_path + f'linear/{_iter}'
        outputs, _ = run_linear(
            linear_save_path, net, dataset, dataset_config, lr=best_lr, gamma=gamma
            )
    
    # MLP
    net = MLP(input_size=input_size, hidden_size=input_size*2)
    for _iter in range(iters):
        net.reset()
        mlp_save_path = save_path + f'mlp/{_iter}'
        outputs, _ = run_mlp(mlp_save_path, net, dataset, dataset_config, gamma=gamma)

