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
from sr_model.models.models import AnalyticSR, STDP_SR, Linear
from td_utils import run_models

def main(delete_dir=False):
    save_path = '../../engram/Ching/03_hannah_dset_revisions/'
    load_path = '../../engram/Ching/03_td_discrete_corr_revisions/'
    if delete_dir:
        rmtree(save_path, ignore_errors=True)

    iters = 10
    n_jobs = 56
    gammas = [0.4, 0.5, 0.6, 0.75, 0.8]

    # Integer sigmas
    spatial_sigmas = [0.0, 1.0, 2.0, 3.0]
    sparsity_range = [[0.001, 0.2], [0.001, 0.1], [0.001, 0.04], [0.001, 0.023]]

    # Other sigmas
    spatial_sigmas.extend([
        0.25,
        0.5,
        1.25,
        1.5,
        1.75,
        2.25,
        2.5,
        2.75,
        3.25
        ])
    sparsity_range.extend([
        [0.001, 0.19], # 0.25
        [0.001, 0.15], # 0.5
        [0.001, 0.09], # 1.25
        [0.001, 0.05], # 1.5
        [0.001, 0.045], # 1.75
        [0.001, 0.037], # 2.25
        [0.001, 0.03], # 2.5
        [0.001, 0.025], # 2.75
        [0.001, 0.021], # 3.25
        ])

    lr_range = [5E-3, 1E-3, 5E-4, 1E-4] # Only used for Linear model
    num_states = 14*14
    num_steps = 5001

    def grid_train(arg):
        gamma, spatial_sigma, sparsity_p = arg
        dataset = sf_inputs_discrete.TitmouseWalk
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
        load_from_path = load_path + f'sparsity{sparsity_p}/sigma{spatial_sigma}/{gamma}/'
        run_hopfield = gamma==0.6 # Arbitrary
        run_models(
            dset_path, iters, lr_range, dataset, dataset_config, gamma,
            input_size, save_outputs=True, test_over_all=False,
            print_file=open('dummy.txt', 'w'), run_hopfield=run_hopfield,
            load_from_dir=load_from_path
            )

    args = []
    for gamma in gammas:
        for idx, spatial_sigma in enumerate(spatial_sigmas):
            _range = sparsity_range[idx]
            sparsity_ps = np.linspace(_range[0], _range[1], num=20, endpoint=True)
            for sparsity_p in sparsity_ps:
                args.append([gamma, spatial_sigma, sparsity_p])
    Parallel(n_jobs=n_jobs)(delayed(grid_train)(arg) for arg in args)

if __name__ == "__main__":
    main()
