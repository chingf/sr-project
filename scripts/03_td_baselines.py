import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from joblib import Parallel, delayed
import argparse
from shutil import rmtree

from datasets import inputs, sf_inputs_discrete
from sr_model.models.models import AnalyticSR, STDP_SR, Linear, MLP
from sr_model import configs
from run_td_rnn import run as _run_rnn
from td_utils import run_models
from eval import eval

def main(delete_dir=False):
    experiment_path = f'{configs.engram_dir}03_td_baselines/'
    if delete_dir:
        rmtree(experiment_path, ignore_errors=True)
    os.makedirs(experiment_path)
    
    iters = 10
    gammas = [0.4]
    models = ['rnn_sr', 'rnn_sf']
    n_jobs = 8
    lr_range = [1E-2]
    num_states = 25
    num_steps = 6001
    dataset = inputs.Sim1DWalk
    dataset_config = {'num_steps': num_steps, 'num_states': num_states}
    input_size = num_states
    dset_params = {'dataset': dataset, 'dataset_config': dataset_config}
    with open(experiment_path + "dset_configs.p", 'wb') as f:
        pickle.dump(dset_params, f)
    
    def grid_train(arg):
        gamma, model = arg
        gamma_path = f'{experiment_path}{gamma}/'
        model_path = f'{gamma_path}{model}/'
    
        if 'sf' in model_path:
            net = get_rnn_sf(
                model_path, iters, lr_range, dataset, dataset_config, gamma,
                input_size
                )
        elif 'sr' in model_path:
            net = get_rnn_sr(input_size)
    
        # Get TD loss over walk
        for _iter in range(iters):
            net.reset()
            model_iter_path = model_path + f'{_iter}'
            if os.path.isfile(f'{model_iter_path}/results.p'):
                print(f'{model_iter_path} already calculated. Skipping...')
                continue
            try:
                outputs, _, dset, _ = _run_rnn(
                    model_iter_path, net, dataset, dataset_config, gamma=gamma,
                    train_net=False, test_over_all=False, print_every_steps=100
                    )
            except RuntimeError as e:
                if 'svd' in str(e):
                    continue
                else:
                    raise
            results = {
                'outputs': outputs, 'dset': dset,
                'rnn_T': net.get_T(), 'rnn_M': net.get_M()
                }
            with open(f'{model_iter_path}/results.p', 'wb') as f:
                pickle.dump(results, f)
    
        # Get T and M error over walk
        net.reset()
        T_error, M_error, T_row_norm, T_col_norm =\
            eval(net, [dataset(**dataset_config)]*iters)
        results = {
            'T_error': T_error, 'M_error': M_error,
            'T_row_norm': T_row_norm, 'T_col_norm': T_col_norm
            }
        with open(f'{model_path}results_{gamma}.p', 'wb') as f:
            pickle.dump(results, f)

    # Run in parallel
    args = []
    for gamma in gammas:
        for model in models:
            args.append([gamma, model])
    Parallel(n_jobs=n_jobs)(delayed(grid_train)(arg) for arg in args)
    
def get_rnn_sr(num_states):
    """ Load and return baseline SR model (pre-metalearned) """
    
    exp_dir = '../trained_models/baseline/'
    with open(exp_dir + 'net_configs.p', 'rb') as f:
        net_configs = pickle.load(f)
    net = STDP_SR(**net_configs)
    net.load_state_dict(torch.load(exp_dir + 'model.pt'))
    net.set_num_states(num_states)
    return net

def get_rnn_sf(
    save_path, iters, lr_range, dataset, dataset_config, gamma, input_size
    ):
    """ If needed, runs grid search over learning rates"""

    if len(lr_range) > 1:
        best_net = None; best_lr_val = np.inf;
        for lr in lr_range:
            ca3_kwargs = {
                'use_dynamic_lr':False, 'parameterize': False, 'lr': lr
                }
            net = AnalyticSR( # Initialize network
                num_states=input_size, gamma=gamma, ca3_kwargs=ca3_kwargs
                )
            _, loss, _, net = _run_rnn(
                save_path + 'test/', net, dataset, dataset_config, gamma=gamma,
                train_net=False, test_over_all=False
                )
    
            if loss < best_lr_val:
                best_net = net; best_lr_val = loss;
        net = best_net
    else:
        ca3_kwargs = {
            'use_dynamic_lr':False, 'parameterize': False, 'lr': lr_range[0]
            }
        net = AnalyticSR( # Initialize network
            num_states=input_size, gamma=gamma, ca3_kwargs=ca3_kwargs
            )
    net.reset()
    return net

if __name__ == "__main__":
    main(delete_dir=True)

