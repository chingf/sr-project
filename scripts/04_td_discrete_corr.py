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
from td_utils import run_models

def main(delete_dir=False):
    save_path = '../../engram/Ching/03_td_discrete_corr/'
    if delete_dir:
        rmtree(save_path, ignore_errors=True)

    iters = 3
    gammas = [0.75, 0.6, 0.8] #[0.75, 0.6, 0.8, 0.85, 0.4]
    spatial_sigmas = [0.0, 1.0, 2.0, 3.0]
    sparsity_range = [[0.001, 0.2], [0.001, 0.1], [0.001, 0.04], [0.001, 0.023]]
    lr_range = [5E-3, 1E-3, 5E-4, 1E-4] # Only used for Linear model
    num_states = 14*14
    num_steps = 5401

    def grid_train(arg):
        gamma, spatial_sigma, sparsity_p = arg
        dataset = sf_inputs_discrete.Sim2DWalk
        feature_maker_kwargs = {
            'feature_dim': num_states, 'feature_type': 'correlated_distributed',
            'feature_vals_p': [1-sparsity_p, sparsity_p],
            'spatial_sigma': spatial_sigma
            }
        dataset_config = {
            'num_steps': num_steps, 'feature_maker_kwargs': feature_maker_kwargs,
            'num_states': num_states
            }
        input_size = num_states
        dset_path = save_path + f'sparsity{sparsity_p}/sigma{spatial_sigma}/{gamma}/'
        run_hopfield = gamma==0.6 # Arbitrary
        run_models(
            dset_path, iters, lr_range, dataset, dataset_config, gamma,
            input_size, save_outputs=True, test_over_all=False,
            print_file=open('dummy.txt', 'w'), run_hopfield=run_hopfield
            )

    args = []
    for gamma in gammas:
        for idx, spatial_sigma in enumerate(spatial_sigmas):
            _range = sparsity_range[idx]
            sparsity_ps = np.linspace(_range[0], _range[1], num=20, endpoint=True)
            for sparsity_p in sparsity_ps:
                args.append([gamma, spatial_sigma, sparsity_p])
    Parallel(n_jobs=56)(delayed(grid_train)(arg) for arg in args)

if __name__ == "__main__":
    main()

