import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import torch
from joblib import Parallel, delayed

from datasets import inputs
from sr_model import configs
from sr_model.models.models import AnalyticSR, STDP_SR
from eval import eval
import pickle

experiment_dir = f'{configs.engram_dir}02_gamma_v_eigenvals/'
experiment_dir = '../../engram/Ching/02_gamma_v_eigenvals/'
os.makedirs(experiment_dir, exist_ok=True)
n_jobs = 56
n_iters = 24

num_steps = 1501
num_states = 25
dataset = inputs.Sim1DWalk
lrs_probs = [[1,1,1], [4,1,1], [1,1,4]]

gammas = np.arange(0.01, 0.86, 0.01)
nonlinearity_args = [None, 1.0]
args = []
for gamma in gammas:
    for nonlinearity_arg in nonlinearity_args:
        for _iter in range(n_iters):
            args.append((gamma, nonlinearity_arg, _iter))

def grid(arg):
    gamma, nonlinearity_arg, _iter = arg
    eigenvals = []
    steps = []
    lrs_prob = lrs_probs[_iter%3]
    dataset_params = {
        'num_steps':num_steps, 'num_states':num_states,
        'left_right_stay_prob': lrs_prob
        }
    dset = dataset(**dataset_params)
    dg_inputs = torch.from_numpy(dset.dg_inputs.T).float().unsqueeze(1)
    if nonlinearity_arg is None:
        nonlin_str = 'Linear'
        output_params = {}
    else:
        nonlin_str = 'Tanh'
        rstep = int(np.log(1E-5)/np.log(gamma))
        output_params = {
            'num_iterations':rstep,
            'nonlinearity': 'tanh', 'nonlinearity_args': nonlinearity_arg
            }
    net_configs = {
        'gamma':gamma, 'ca3_kwargs':{'output_params': output_params}
        }
    net = STDP_SR(
        num_states=num_states, gamma=gamma, ca3_kwargs=net_configs['ca3_kwargs']
        )
    for step in range(dset.num_steps):
        curr_dg_input = dg_inputs[step].unsqueeze(0)
        reset = True if step == 0 else False
        _, outputs = net(curr_dg_input, reset=reset)
        if step % 100 == 0:
            T = net.get_T().detach().numpy()
            eigenval, _ = np.linalg.eig(T)
            max_eigenval = np.max(np.real(eigenval))
            eigenvals.append(gamma*max_eigenval)
            steps.append(step)
    return gamma, nonlin_str, eigenvals, steps

results = {}
results['gammas'] = []
results['nonlinearity_args'] = []
results['eigenvals'] = []
results['steps'] = []
job_results = Parallel(n_jobs=n_jobs)(delayed(grid)(arg) for arg in args)
for res in job_results:
    gamma, nonlinearity_arg, eigenvals, steps = res
    results['gammas'].extend([gamma]*len(steps))
    results['nonlinearity_args'].extend([nonlinearity_arg]*len(steps))
    results['eigenvals'].extend(eigenvals)
    results['steps'].extend(steps)
with open(f'{experiment_dir}results.p', 'wb') as f:
    pickle.dump(results, f)

