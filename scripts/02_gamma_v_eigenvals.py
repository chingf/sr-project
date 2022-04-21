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
n_jobs = 1
n_iters = 10

num_steps = 3001
num_states = 25
dataset = inputs.Sim1DWalk
dataset_params = {
    'num_steps':num_steps, 'num_states':num_states,
    'left_right_stay_prob':[4,1,1]
    }

gammas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
args = []
for gamma in gammas:
    for _iter in range(n_iters):
        args.append((gamma, _iter))

def grid(arg):
    gamma, _iter = arg
    eigenvals = []
    steps = []
    output_params = {}
    net_configs = {
        'gamma':gamma, 'ca3_kwargs':{'output_params': output_params}
        }
    net = STDP_SR(
        num_states=num_states, gamma=gamma, ca3_kwargs=net_configs['ca3_kwargs']
        )
    dset = dataset(**dataset_params)
    dg_inputs = torch.from_numpy(dset.dg_inputs.T).float().unsqueeze(1)
    for step in range(dset.num_steps):
        curr_dg_input = dg_inputs[step].unsqueeze(0)
        reset = True if step == 0 else False
        _, outputs = net(curr_dg_input, reset=reset)
        if step % 100 == 0:
            M = net.get_M().detach().numpy()
            eigenval, _ = np.linalg.eig(M)
            max_eigenval = np.max(np.abs(eigenval))
            eigenvals.append(max_eigenval)
            steps.append(step)
    return gamma, eigenvals, steps

results = {}
results['gammas'] = []
results['eigenvals'] = []
results['steps'] = []
job_results = Parallel(n_jobs=n_jobs)(delayed(grid)(arg) for arg in args)
for res in job_results:
    gamma, eigenvals, steps = res
    results['gammas'].extend([gamma]*len(steps))
    results['eigenvals'].extend(eigenvals)
    results['steps'].extend(steps)
with open(f'{experiment_dir}results.p', 'wb') as f:
    pickle.dump(results, f)

