import numpy as np
import argparse
import time
import os
from itertools import chain
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR

device = 'cpu'

def eval(save_path):
    """
    Evaluates the performance of a model on three datasets. Looks for a model.pt
    and net_configs.p file in SAVE_PATH.

    Returns (1) error from true T over time and (2) deviation from ideally
    estimated T over time.
    """

    model_path = save_path + 'model.pt'
    net_configs_path = save_path + 'net_configs.p'
    if os.path.isfile(net_configs_path):
        net = STDP_SR(num_states=64, gamma=net_configs['gamma'],
            ca3_kwargs=net_configs['ca3_kwargs']
            )
    else:
        net = STDP_SR(num_states=64, gamma=0.4)
    net.load_state_dict(torch.load(model_path))
    net.ca3.set_differentiability(False)
    
    # Dataset Configs
    datasets = [ 
        inputs.Sim1DWalk(num_steps=8000, left_right_stay_prob=[5,1,1], num_states=64),
        inputs.Sim2DWalk(num_steps=8000, num_states=64),
        inputs.Sim2DLevyFlight(num_steps=8000, walls=7)
        ]
    results_true_v_rnn = []
    results_est_v_rnn = []
    results_true_v_est = []
    
    with torch.no_grad():
        for dset in datasets:
            res_true_v_rnn = []
            res_est_v_rnn = []
            res_true_v_est = []

            dg_inputs = torch.from_numpy(dset.dg_inputs.T).float().to(device).unsqueeze(1)
            dg_modes = torch.from_numpy(dset.dg_modes.T).float().to(device).unsqueeze(1)
            for step in range(dset.num_steps):
                curr_dg_input = dg_inputs[step].unsqueeze(0)
                curr_dg_mode = dg_modes[step].unsqueeze(0)
                _, outputs = net(curr_dg_input, curr_dg_mode, reset=False)
                rnn_T = net.ca3.get_T().detach().numpy()

                true_T = dset.get_true_T()
                true_error = np.mean(np.abs(true_T - rnn_T))
                res_true_v_rnn.append(true_error)

                est_T = net.ca3.get_ideal_T_estimate()
                est_error = np.mean(np.abs(est_T - rnn_T))
                res_est_v_rnn.append(est_error)

                true_v_est_error = np.mean(np.abs(est_T - true_T))
                res_true_v_est.append(true_v_est_error)

            results_true_v_rnn.append(np.array(res_true_v_rnn))
            results_est_v_rnn.append(np.array(res_est_v_rnn))
            results_true_v_est.append(np.array(res_true_v_est))

    return results_true_v_rnn, results_est_v_rnn, results_true_v_est

if __name__ == "__main__":
    save_path = './trained_models/'
    results_true_v_rnn, results_est_v_rnn, results_true_v_est = eval(save_path)

    plt.figure();
    plt.plot(results_true_v_rnn[0], 'r-', label='1D Walk (RNN)')
    plt.plot(results_true_v_rnn[1], 'b-', label='2D Walk (RNN)')
    plt.plot(results_true_v_est[0], 'r--', label='1D Walk (original)')
    plt.plot(results_true_v_est[1], 'b--', label='2D Walk (original)')
    plt.title("Estimation of True T")
    plt.ylabel("MAE")
    plt.xlabel("Timestep of Simulation")
    plt.legend()
    plt.show()

    plt.figure();
    plt.plot(results_est_v_rnn[0], label='1D Random Walk')
    plt.plot(results_est_v_rnn[1], label='2D Random Walk')
    plt.plot(results_est_v_rnn[2], label='2D Levy Flight')
    plt.title("Deviation from Ideal Estimator")
    plt.ylabel("MAE")
    plt.xlabel("Timestep of Simulation")
    plt.legend()
    plt.show()
