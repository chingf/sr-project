import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
import numpy as np
import torch.nn as nn
import torch
from joblib import Parallel, delayed
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR
from train import train

experiment_dir = '../../engram/Ching/02_gamma_v_ss/'
n_jobs = 14
n_test_iters = 40
dataset = inputs.Sim1DWalk
lrs_probs = [[1,1,1], [5,1,1], [1,1,5]]
num_states = 25
num_steps = 400

gammas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
nonlinearity_args = [None] #, 1.0, 1.5, 2.0]
args = []
for gamma in gammas:
    for nonlinearity_arg in nonlinearity_args:
        args.append((gamma, nonlinearity_arg))

def grid_train(arg):
    _T_maes = []
    _M_maes = []
    gamma, nonlinearity_arg = arg
    gamma_dir = f'{experiment_dir}{gamma}/{nonlinearity_arg}/'
    best_iter_val = np.inf
    best_iter = None
    for _iter in os.listdir(gamma_dir):
        iter_dir = f'{gamma_dir}{_iter}/'
        if not os.path.isfile(iter_dir + 'net_configs.p'):
            continue
        iter_val = []
        for idx in len(lrs_probs):
            _, _M_mae = run(iter_dir, idx)
            iter_val.append(_M_mae)
        iter_val = np.mean(iter_val)
        if iter_val < best_iter_val:
            best_iter_val = iter_val
            best_iter = _iter
    for _ in range(n_test_iters):
        T_mae, M_mae = run(f'{gamma_dir}{best_iter}/')
        _T_maes.append(T_mae)
        _M_maes.append(M_mae)
    return nonlinearity_arg, gamma, _T_maes, _M_maes

def run(exp_dir, lrs_prob_idx=None):
    if lrs_prob_idx is None:
        lrs_prob_idx = np.random.choice(len(lrs_probs))
    lrs_prob = lrs_probs[lrs_prob_idx]
    dset = inputs.Sim1DWalk(
        num_steps=num_steps, num_states=num_states,
        left_right_stay_prob=lrs_prob
        )
    dg_inputs = torch.from_numpy(dset.dg_inputs.T).float().unsqueeze(1)
    with open(exp_dir + 'net_configs.p', 'rb') as f:
        net_configs = pickle.load(f)
        net_configs['num_states'] = num_states
    net = STDP_SR(**net_configs)
    net.load_state_dict(torch.load(exp_dir + 'model.pt'))
    with torch.no_grad():
        _, outputs = net(dg_inputs, reset=True)
    rnn_T = net.get_T().numpy()
    rnn_M = net.get_M()
    if type(rnn_M) is not np.ndarray:
        rnn_M = rnn_M.numpy()
    est_T = dset.est_T
    est_M = np.linalg.pinv(np.eye(est_T.shape[0]) - net.gamma*est_T)
    return np.mean(np.abs(rnn_T - est_T)), np.mean(np.abs(rnn_M - est_M))

nonlinearities = []
gammas = []
T_maes = []
M_maes = []
job_results = Parallel(n_jobs=n_jobs)(delayed(grid_train)(arg) for arg in args)
for res in job_results:
    nonlinearity, gamma, _T_maes, _M_maes = res
    nonlinearities.extend([nonlinearity]*len(_T_maes))
    gammas.extend([gamma]*len(_T_maes))
    T_maes.extend(_T_maes)
    M_maes.extend(_M_maes)
results = {
    'nonlinearities': nonlinearities, 'gammas': gammas,
    'T_maes': T_maes, 'M_maes': M_maes
    }
with open(f'{experiment_dir}test_results.p', 'wb') as f:
    pickle.dump(results, f)
