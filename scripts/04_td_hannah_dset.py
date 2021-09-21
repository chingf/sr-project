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
    save_path = '../trained_models/03_hannah_dset/'
    if delete_dir:
        rmtree(save_path, ignore_errors=True)

    iters = 7
    gammas = [0.4, 0.6, 0.75, 0.8, 0.95]
    spatial_sigmas = [1.0, 1.25, 1.5, 1.75, 2.0]
    lr_range = [1E-2, 1E-3]
    num_states = 14*14

    # Correlated, with 80/20 sampling
    for gamma in gammas:
        for spatial_sigma in spatial_sigmas:
            dataset = sf_inputs_discrete.TitmouseWalk
            feature_maker_kwargs = {
                'feature_dim': num_states, 'feature_type': 'correlated_distributed',
                'feature_vals_p': [0.8, 0.2], 'spatial_sigma': spatial_sigma
                }
            dataset_config = {
                'feature_maker_kwargs': feature_maker_kwargs,
                'num_steps': np.inf, 'num_states': num_states
                }
            input_size = num_states
            dset_path = save_path + f'pval80_sigma{spatial_sigma}/{gamma}/'
            run_models(
                dset_path, iters, lr_range, dataset, dataset_config, gamma,
                input_size, save_outputs=True, test_over_all=False
                )

    # Correlated, with 95/5 sampling
    for gamma in gammas:
        for spatial_sigma in spatial_sigmas:
            dataset = sf_inputs_discrete.TitmouseWalk
            feature_maker_kwargs = {
                'feature_dim': num_states, 'feature_type': 'correlated_distributed',
                'feature_vals_p': [0.95, 0.05], 'spatial_sigma': spatial_sigma
                }
            dataset_config = {
                'feature_maker_kwargs': feature_maker_kwargs,
                'num_steps': np.inf, 'num_states': num_states
                }
            input_size = num_states
            dset_path = save_path + f'pval95_sigma{spatial_sigma}/{gamma}/'
            run_models(
                dset_path, iters, lr_range, dataset, dataset_config, gamma,
                input_size, save_outputs=True, test_over_all=False
                )

if __name__ == "__main__":
    main()

