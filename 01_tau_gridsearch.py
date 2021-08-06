import pickle
import numpy as np
import torch.nn as nn
from joblib import Parallel, delayed

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR
from train_cma import train

experiment_dir = './trained_models/01_tau_gridsearch/'
tau_negs = np.linspace(0.8, 1.6, num=10, endpoint=True)
tau_poses = np.linspace(0.8, 1.6, num=10, endpoint=True)
A_signs = [1, -1]

datasets = [inputs.Sim1DWalk]
datasets_config_ranges = [{
    'num_steps': [3, 10, 15],
    'left_right_stay_prob': [[1, 1, 1], [7, 1, 1], [1, 4, 1]],
    'num_states': [5, 10, 15]
    }]

args = []
for tau_neg in tau_negs:
    for tau_pos in tau_poses:
        for A_pos_sign in A_signs:
            for A_neg_sign in A_signs:
                args.append((tau_neg, tau_pos, A_pos_sign, A_neg_sign))

def main():
    tau_neg_axes = []
    tau_pos_axes = []
    vals = []
    results = {
        'tau_neg_axes': tau_neg_axes, 'tau_pos_axes': tau_pos_axes,
        'vals': vals
        }
    job_results = Parallel(n_jobs=7)(delayed(grid_train)(arg) for arg in args)
    for res in job_results:
        tau_neg_axes.append(res[0])
        tau_pos_axes.append(res[1])
        vals.append(res[2])
    with open(experiment_dir + 'results.p', 'wb') as f:
        pickle.dump(results, f)

def grid_train(arg):
    tau_neg, tau_pos, A_pos_sign, A_neg_sign = arg
    tau_pos_ax = tau_pos*A_pos_sign
    tau_neg_ax = tau_neg*A_neg_sign
    save_path = experiment_dir + f'pos{tau_pos_ax}_neg{tau_neg_ax}/'

    # Initialize network
    net_configs = {
        'gamma': 0.4, 'ca3_kwargs':
        {'A_pos_sign':A_pos_sign, 'A_neg_sign':A_neg_sign}
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
    try:
        net, running_loss = train(
            save_path, net, datasets, datasets_config_ranges
            )
    except RuntimeError:
        running_loss = np.nan
    
    # Save vals
    with open(save_path + 'net_configs.p', 'wb') as f:
        pickle.dump(net_configs, f)
    return tau_neg_ax, tau_pos_ax, running_loss

if __name__ == "__main__":
    main()

