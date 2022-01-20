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
from run_td_rnn import run as _run_rnn
from td_utils import run_models
from eval import eval

def main(delete_dir=False):
    save_path = '/home/chingf/engram/Ching/03_td_baselines/'
    if delete_dir:
        rmtree(save_path, ignore_errors=True)

    iters = 1
    gammas = [0.95]
    lr_range = [5E-2]
    num_states = 100
    num_steps = 20001
   
    # One-hot inputs
    for gamma in gammas:
        dataset = inputs.Sim2DWalk
        dataset_config = {'num_steps': num_steps, 'num_states': num_states}
        input_size = num_states
        dset_path = save_path + f'onehot/{gamma}/'
        net = run_rnn(
            dset_path, iters, lr_range, dataset, dataset_config, gamma,
            input_size
            )
        T_error, M_error, T_row_norm, T_col_norm =\
            eval(net, [dataset(**dataset_config)]*iters)
        results = {
            'T_error': T_error, 'M_error': M_error,
            'T_row_norm': T_row_norm, 'T_col_norm': T_col_norm
            }
        with open(f'{save_path}onehot/results_{gamma}.p', 'wb') as f:
            pickle.dump(results, f)
   
def run_rnn(
    save_path, iters, lr_range, dataset, dataset_config, gamma, input_size
    ):

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

    # Manually load
#    ca3_kwargs = {
#        'use_dynamic_lr':False, 'parameterize': False, 'lr': lr_range[0]
#        }
#    net = AnalyticSR( # Initialize network
#        num_states=input_size, gamma=gamma, ca3_kwargs=ca3_kwargs
#        )

    for _iter in range(iters):
        net.reset()
        rnn_save_path = save_path + f'{_iter}'
        if os.path.isfile(f'{rnn_save_path}/results.p'):
            print(f'{rnn_save_path} already calculated. Skipping...')
            continue
        try:
            outputs, _, dset, _ = _run_rnn(
                rnn_save_path, net, dataset, dataset_config, gamma=gamma,
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
        with open(f'{rnn_save_path}/results.p', 'wb') as f:
            pickle.dump(results, f)
    net.reset()
    return net

if __name__ == "__main__":
    main()

