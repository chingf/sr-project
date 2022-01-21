import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
import numpy as np
import torch.nn as nn
from joblib import Parallel, delayed

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR
from train import train

experiment_dir = '../../engram/Ching/02_gamma_v_rsteps/'
n_jobs = 56

datasets = [
    inputs.Sim1DWalk,
    inputs.Sim1DWalk,
    ]
datasets_config_ranges = [
    {
    'num_steps': [3, 10, 20, 30],
    'left_right_stay_prob': [[1, 1, 1], [1, 1, 5], [7, 1, 0], [1, 7, 0]],
    'num_states': [5, 10, 15, 25]
    },
    {
    'num_steps': [100, 200],
    'num_states': [36, 64]
    },
    ]

gammas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rsteps = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
nonlinearities = [None, 'clamp']

args = []
for gamma in gammas:
    for rstep in rsteps:
        for nonlinearity in nonlinearities:
            args.append((gamma, rstep, nonlinearity))

def main():
    gamma_axes = []
    rstep_axes = []
    vals = []
    nonlinearities = []
    results = {
        'gamma_axes': gamma_axes, 'rstep_axes': rstep_axes,
        'nonlinearities': nonlinearities, 'vals': vals
        }
    job_results = Parallel(n_jobs=n_jobs)(delayed(grid_train)(arg) for arg in args)
    for res in job_results:
        gamma_axes.append(res[0])
        rstep_axes.append(res[1])
        nonlinearities.append(res[2])
        vals.append(res[3])
    with open(experiment_dir + 'results.p', 'wb') as f:
        pickle.dump(results, f)

def grid_train(arg):
    gamma, rstep, nonlinearity = arg
    gamma_exp_dir = experiment_dir + f'{gamma}/'
    rstep_dir = gamma_exp_dir + f'{rstep}/'
    save_path = rstep_dir + f'{nonlinearity}/'

    losses = []
    output_params = {
        'num_iterations':rstep, 'input_clamp':rstep, 'nonlinearity': nonlinearity
        }
    net_configs = {
        'gamma':gamma,
        'ca3_kwargs':{
            'A_pos_sign':1, 'A_neg_sign':-1,
            'output_params': output_params
            }
        }
    for idx in range(3):
        net = STDP_SR(
            num_states=2, gamma=gamma, ca3_kwargs=net_configs['ca3_kwargs']
            )
        net, loss = train(
            save_path, net, datasets, datasets_config_ranges, train_steps=801,
            early_stop=True, return_test_error=True, train_M=True
            )
        losses.append(loss)
        if loss < 1E-4: break # No need to run more iterations
    val = np.nanmin(losses)
    with open(save_path + 'net_configs.p', 'wb') as f:
        pickle.dump(net_configs, f)
    return gamma, rstep, nonlinearity, val

if __name__ == "__main__":
    main()

