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
from sr_model.models.models import AnalyticSR, STDP_SR, OjaRNN, Linear

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
    else:
        net = path_or_model
    
    results_T_error = [] # From underlying T
    results_M_error = [] # From SR of underlying T
    results_T_row_norm = [] # Rows in a proper T sum to 1
    results_T_col_norm = []
    
    with torch.no_grad():
        for dset in datasets:
            res_t_error = []
            res_m_error = []
            res_t_row_norm = []
            res_t_col_norm = []
            net.set_num_states(dset.num_states)
            net.reset()

            dg_inputs = torch.from_numpy(dset.dg_inputs.T).float().to(device).unsqueeze(1)
            dg_modes = torch.from_numpy(dset.dg_modes.T).float().to(device).unsqueeze(1)
            for step in range(dset.num_steps):
                curr_dg_input = dg_inputs[step].unsqueeze(0)
                curr_dg_mode = dg_modes[step].unsqueeze(0)
                reset = True if step == 0 else False

                if isinstance(net, Linear):
                    outputs = net(curr_dg_input, reset=reset)
                else:
                    _, outputs = net(curr_dg_input, reset=reset)

                if step==0: continue

                # How well the model estimates the true T/M
                net_T = net.get_T().detach().numpy()
                net_M = net.get_M().detach().numpy()
                true_T = dset.get_true_T()
                true_M = np.linalg.pinv(np.eye(true_T.shape[0]) - net.gamma*true_T)
                t_error = np.mean(np.abs(true_T - net_T))
                m_error = np.mean(np.abs(true_M - net_M))
                res_t_error.append(t_error)
                res_m_error.append(m_error)

                # Check normalization of T (both row and col)
                res_t_row_norm.append(np.mean(np.sum(net_T, axis=1)))
                res_t_col_norm.append(np.mean(np.sum(net_T, axis=0)))

            results_T_error.append(np.array(res_t_error))
            results_M_error.append(np.array(res_m_error))
            results_T_row_norm.append(np.array(res_t_row_norm))
            results_T_col_norm.append(np.array(res_t_col_norm))

    return results_T_error, results_M_error, results_T_row_norm, results_T_col_norm

if __name__ == "__main__":
    save_path = './trained_models/0.95_2/'
    net = OjaRNN(2, 0.4)

    datasets = [ 
        inputs.Sim1DWalk(num_steps=1000, left_right_stay_prob=[1,1,1], num_states=64),
        inputs.Sim1DWalk(num_steps=1000, left_right_stay_prob=[5,1,1], num_states=64),
        inputs.Sim2DWalk(num_steps=1000, num_states=64),
        inputs.Sim2DLevyFlight(num_steps=1000, walls=7)
        ]

    results_T_error, results_M_error, results_T_row_norm, results_T_col_norm =\
        eval(net, datasets)

    plt.figure();
    plt.plot(results_T_error[0], 'r-', label='1D Walk (RNN)')
    plt.plot(results_T_error[2], 'b-', label='2D Walk (RNN)')
    plt.title("Estimation of True T")
    plt.ylabel("MAE")
    plt.xlabel("Timestep of Simulation")
    plt.legend()
    plt.tight_layout()
#    plt.savefig('eval_true_v_rnn.png', dpi=300)
    plt.show()

