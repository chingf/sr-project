import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import pickle
import numpy as np
import torch.nn as nn
from joblib import Parallel, delayed
import argparse

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR
from train import train

experiment_dir = '../../engram/Ching/01_tau_gridsearch/'
#experiment_dir = '../trained_models/01_tau_gridsearch/'
os.makedirs(experiment_dir, exist_ok=True)
n_jobs = 56
n_iters = 20
n_train_steps = 401

tau_negs = np.arange(0.2, 2.5, 0.2)
tau_poses = np.arange(0.2, 3.5, 0.2)
A_signs = [1, -1]

datasets = [inputs.Sim1DWalk]
datasets_config_ranges = [{
    'num_steps': [3, 10, 20, 30],
    'left_right_stay_prob': [[1, 1, 1], [7, 1, 1], [1, 1, 5], [1, 7, 0]],
    'num_states': [5, 10, 15, 25]
    }]

grid_params = [] # 640 total
for tau_neg in tau_negs:
    for tau_pos in tau_poses:
        for A_pos_sign in A_signs:
            for A_neg_sign in A_signs:
                if A_pos_sign*tau_pos < 0: continue # Only pre-post potentiation
                if A_neg_sign*tau_neg > 1: continue # Post-pre potentiation limit
                grid_params.append((tau_neg, tau_pos, A_pos_sign, A_neg_sign))
print(len(grid_params))

def main():
    Parallel(n_jobs=n_jobs)(delayed(grid_train)(param) for param in grid_params)

def slurm_main(idx):
    param = grid_params[idx]
    grid_train(param)

def grid_train(arg):
    tau_neg, tau_pos, A_pos_sign, A_neg_sign = arg
    tau_pos_ax = tau_pos*A_pos_sign
    tau_neg_ax = tau_neg*A_neg_sign
    save_path = experiment_dir + f'pos{tau_pos_ax}_neg{tau_neg_ax}/'

    # Initialize network
    net_configs = {
        'gamma': 0.4, 'ca3_kwargs':
        {'A_pos_sign':A_pos_sign, 'A_neg_sign':A_neg_sign,
        'use_kernels_in_update':True, 'approx_B': False, 'use_B_norm': False
        }
        }
    net = STDP_SR(
        num_states=2, gamma=net_configs['gamma'],
        ca3_kwargs=net_configs['ca3_kwargs']
        )

    # Load grid parameters
    nn.init.constant_(net.ca3.tau_pos, tau_pos)
    nn.init.constant_(net.ca3.tau_neg, tau_neg)
    net.ca3.tau_pos.requires_grad = False
    net.ca3.tau_neg.requires_grad = False

    # Train
    losses = []
    for _ in range(n_iters):
        try:
            net, loss = train(
                save_path, net, datasets, datasets_config_ranges,
                train_steps=n_train_steps, early_stop=True,
                )
            losses.append(loss)
            if loss < 1E-5: break # No need to run more iterations
        except RuntimeError:
            losses.append(np.nan)
    running_loss = np.nanmin(losses)
    
    # Save vals
    with open(save_path + 'net_configs.p', 'wb') as f:
        pickle.dump(net_configs, f)
    return tau_neg_ax, tau_pos_ax, running_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run gridsearch indices.')
    parser.add_argument('--index', type=int, nargs='?',
                        help='Index of arg', default=-1)
    args = parser.parse_args()
    if args.index == -1:
        main()
    else:
        slurm_main(args.index)

