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
from sr_model.models.models import AnalyticSR, STDP_SR, Linear, MLP
from run_td_rnn import run as run_rnn
from run_td_mlp import run as run_mlp
from run_td_linear import run as run_linear

def main(delete_dir=False):
    save_path = '../trained_models/03_td_discrete_corr/'
    if delete_dir:
        rmtree(save_path, ignore_errors=True)
    iters = 5
    lr_range = [1E-1, 1E-2, 1E-3]
    gamma=0.4

    feature_val_sets = [
        ([0, 1.], 'binary'),
        (None, 'contin')
        ]
    feature_type_sets = [
        ('correlated_sparse', 'sparse'),
        ('correlated_distributed', 'distrib')
        ]
    for feature_vals, val_name in feature_val_sets:
        for feature_type, type_name in feature_type_sets:
            dataset = sf_inputs_discrete.Sim2DLevyFlight
            feature_maker_kwargs = {
                'feature_dim':64, 'feature_type': feature_type,
                'feature_vals': feature_vals, 'spatial_sigma': 1.25
                }
            dataset_config = {
                'num_steps': 2000, 'walls': 7,
                'feature_maker_kwargs': feature_maker_kwargs
                }
            dset_path = save_path + f'/{type_name}_{val_name}/'
            input_size = feature_maker_kwargs['feature_dim']
            run_models(
                dset_path, iters, lr_range,
                dataset, dataset_config, gamma, input_size
                )

def run_models(
    save_path, iters, lr_range, dataset, dataset_config, gamma, input_size
    ):

    input_size = int(input_size)

    # Analytic RNN with fixed LR
    best_lr = np.inf; best_lr_val = np.inf;
    for lr in lr_range:
        net = AnalyticSR(
            num_states=input_size, gamma=gamma,
            ca3_kwargs={'use_dynamic_lr':False, 'lr': lr}
            )
        _, loss = run_rnn(
            save_path + 'test/', net, dataset, dataset_config, gamma=gamma
            )
        if loss < best_lr_val:
            best_lr = lr; best_lr_val = loss;
    net = AnalyticSR(
        num_states=input_size, gamma=gamma,
        ca3_kwargs={'use_dynamic_lr':False, 'lr': best_lr}
        )
    for _iter in range(iters):
        rnn_save_path = save_path + f'rnn_fixedlr/{_iter}'
        try:
            net.reset()
            run_rnn(rnn_save_path, net, dataset, dataset_config, gamma=gamma)
        except:
            rmtree(rnn_save_path, ignore_errors=True)
            net.reset()
            run_rnn(rnn_save_path, net, dataset, dataset_config, gamma=gamma)
    
    # Analytic RNN with dynamic LR
    net = AnalyticSR(
        num_states=input_size, gamma=gamma, ca3_kwargs={'use_dynamic_lr':True}
        )
    for _iter in range(iters):
        rnn_save_path = save_path + f'rnn_dynamiclr/{_iter}'
        try:
            net.reset()
            run_rnn(rnn_save_path, net, dataset, dataset_config, gamma=gamma)
        except:
            rmtree(rnn_save_path, ignore_errors=True)
            net.reset()
            run_rnn(rnn_save_path, net, dataset, dataset_config, gamma=gamma)   

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
        run_linear(
            linear_save_path, net, dataset, dataset_config, lr=best_lr, gamma=gamma
            )
    
    # MLP
    net = MLP(input_size=input_size, hidden_size=input_size*2)
    for _iter in range(iters):
        net.reset()
        mlp_save_path = save_path + f'mlp/{_iter}'
        run_mlp(mlp_save_path, net, dataset, dataset_config, gamma=gamma)

if __name__ == "__main__":
    main()
