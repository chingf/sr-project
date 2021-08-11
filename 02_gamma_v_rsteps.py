import pickle
import numpy as np
import torch.nn as nn
from joblib import Parallel, delayed
import os

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR
from train_cma import train

experiment_dir = './trained_models/02_gamma_v_rsteps/'

datasets = [inputs.Sim1DWalk]
datasets_config_ranges = [{
    'num_steps': [3, 10, 20, 30],
    'left_right_stay_prob': [[1, 1, 1], [7, 1, 1], [1, 4, 1]],
    'num_states': [5, 10, 15, 25, 36]
    }]


gammas = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
rsteps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

args = []
for gamma in gammas:
    for rstep in rsteps:
        args.append((gamma, rstep))

def main():
    gamma_axes = []
    rstep_axes = []
    vals = []
    results = {'gamma_axes': gamma_axes, 'rstep_axes': rstep_axes, 'vals': vals}
    job_results = Parallel(n_jobs=7)(delayed(grid_train)(arg) for arg in args)
    for res in job_results:
        gamma_axes.append(res[0])
        rstep_axes.append(res[1])
        vals.append(res[2])
    with open(experiment_dir + 'results.p', 'wb') as f:
        pickle.dump(results, f)

def grid_train(arg):
    gamma, rstep = arg
    gamma_exp_dir = experiment_dir + f'{gamma}/'
    save_path = gamma_exp_dir + f'{rstep}/'
    losses = []
    net_configs = {
        'gamma':gamma,
        'ca3_kwargs': {'output_params':{'num_iterations':rstep, 'input_clamp':rstep}}
        }
    for idx in range(5):
        net = STDP_SR(
            num_states=2, gamma=gamma, ca3_kwargs=net_configs['ca3_kwargs']
            )
        net, loss = train(
            save_path, net, datasets, datasets_config_ranges, train_steps=301
            )
        losses.append(loss)
        if loss == 0.0: break # No need to run more iterations
    val = np.nanmin(losses)
    with open(save_path + 'net_configs.p', 'wb') as f:
        pickle.dump(net_configs, f)
    return gamma, rstep, val

if __name__ == "__main__":
    main()

