import numpy as np
import argparse
import time
import os
import pickle
from itertools import chain
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR, OjaRNN

device = 'cpu'

def eval(path_or_model, datasets):
    """
    Evaluates the performance of a model on three datasets. Looks for a model.pt
    and net_configs.p file in SAVE_PATH.

    Returns (1) error from true T over time and (2) deviation from ideally
    estimated T over time.
    """

    if type(path_or_model) == str:
        save_path = path_or_model
        model_path = save_path + 'model.pt'
        net_configs_path = save_path + 'net_configs.p'
        if os.path.isfile(net_configs_path):
            with open(net_configs_path, 'rb') as f:
                net_configs = pickle.load(f)
            net_configs.pop('num_states')
            net = STDP_SR(num_states=64, **net_configs)
        else:
            print("Loading default configs")
            net = STDP_SR(num_states=64, gamma=0.4)
        net.load_state_dict(torch.load(model_path))
        net.ca3.set_differentiability(False)
    else:
        net = path_or_model
    
    results_true_v_rnn = []
    results_est_v_rnn = []
    results_true_v_est = []
    results_T_row_norm = [] # Rows in a proper T sum to 1
    results_T_col_norm = []
    
    with torch.no_grad():
        for dset in datasets:
            res_true_v_rnn = []
            res_est_v_rnn = []
            res_true_v_est = []
            res_t_row_norm = []
            res_t_col_norm = []
            net.ca3.set_num_states(dset.num_states)
            net.ca3.reset()

            dg_inputs = torch.from_numpy(dset.dg_inputs.T).float().to(device).unsqueeze(1)
            dg_modes = torch.from_numpy(dset.dg_modes.T).float().to(device).unsqueeze(1)
            for step in range(dset.num_steps):
                curr_dg_input = dg_inputs[step].unsqueeze(0)
                curr_dg_mode = dg_modes[step].unsqueeze(0)
                reset = True if step == 0 else False
                _, outputs = net(curr_dg_input, curr_dg_mode, reset=reset)
                rnn_T = net.ca3.get_T().detach().numpy()

                if step==0: continue

                # How well the RNN estimates the true T
                true_T = dset.get_true_T()
                true_error = np.mean(np.abs(true_T - rnn_T))
                res_true_v_rnn.append(true_error)

                # How well the RNN follows the observed T
                est_T = net.ca3.get_ideal_T_estimate()
                valid_counts = net.ca3.real_T_count > 0
                est_error = np.mean(np.abs(
                    est_T[valid_counts,:] - rnn_T[valid_counts,:]
                    ))
                res_est_v_rnn.append(est_error)

                # How well the observed T estimates the true T
                true_v_est_error = np.mean(np.abs(
                    est_T[valid_counts,:] - true_T[valid_counts,:]
                    ))
                res_true_v_est.append(true_v_est_error)

                # Check normalization of T (both row and col)
                res_t_row_norm.append(np.mean(np.sum(rnn_T[valid_counts,:], axis=1)))
                res_t_col_norm.append(np.mean(np.sum(rnn_T[:, valid_counts], axis=0)))

            results_true_v_rnn.append(np.array(res_true_v_rnn))
            results_est_v_rnn.append(np.array(res_est_v_rnn))
            results_true_v_est.append(np.array(res_true_v_est))
            results_T_row_norm.append(np.array(res_t_row_norm))
            results_T_col_norm.append(np.array(res_t_col_norm))

    return results_true_v_rnn, results_est_v_rnn, results_true_v_est,\
        results_T_row_norm, results_T_col_norm

if __name__ == "__main__":
    save_path = './trained_models/0.95_2/'
    net = OjaRNN(2, 0.4)

    datasets = [ 
        inputs.Sim1DWalk(num_steps=1000, left_right_stay_prob=[1,1,1], num_states=64),
        inputs.Sim1DWalk(num_steps=1000, left_right_stay_prob=[5,1,1], num_states=64),
        inputs.Sim2DWalk(num_steps=1000, num_states=64),
        inputs.Sim2DLevyFlight(num_steps=1000, walls=7)
        ]

    results_true_v_rnn, results_est_v_rnn, results_true_v_est,\
        results_T_row_norm, results_T_col_norm = eval(
            net, datasets
            )

    plt.figure();
    plt.plot(results_true_v_rnn[0], 'r-', label='1D Walk (RNN)')
    plt.plot(results_true_v_rnn[2], 'b-', label='2D Walk (RNN)')
    plt.plot(results_true_v_est[0], 'r--', label='1D Walk (original)')
    plt.plot(results_true_v_est[2], 'b--', label='2D Walk (original)')
    plt.title("Estimation of True T")
    plt.ylabel("MAE")
    plt.xlabel("Timestep of Simulation")
    plt.legend()
    plt.tight_layout()
#    plt.savefig('eval_true_v_rnn.png', dpi=300)
    plt.show()

    plt.figure();
    plt.plot(results_est_v_rnn[0], label='1D Random Walk')
    plt.plot(results_est_v_rnn[2], label='2D Random Walk')
    plt.plot(results_est_v_rnn[3], label='2D Levy Flight')
    plt.title("Deviation from Ideal Estimator")
    plt.ylabel("MAE")
    plt.xlabel("Timestep of Simulation")
    plt.legend()
    plt.tight_layout()
#    plt.savefig('eval_est_v_rnn.png', dpi=300)
    plt.show()

