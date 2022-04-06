import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import torch.nn as nn
from joblib import Parallel, delayed

from datasets import inputs
from sr_model import configs
from sr_model.models.models import AnalyticSR, STDP_SR
from eval import eval
import pickle

experiment_dir = f'{configs.engram_dir}02_gamma_v_rstep/'
experiment_dir = '../../engram/Ching/02_gamma_v_rstep/'
os.makedirs(experiment_dir, exist_ok=True)
n_jobs = 16

num_steps = 3001
num_states = 25
datasets = [
    inputs.Sim1DWalk(num_steps=num_steps, num_states=num_states),
    inputs.Sim1DWalk(num_steps=num_steps, num_states=num_states, left_right_stay_prob=[4,1,1]),
    inputs.Sim1DWalk(num_steps=num_steps, num_states=num_states, left_right_stay_prob=[1,1,4]),
    ]

gammas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rsteps = [2, 3, 4, 5, 6, 7, 8, 9, 10]
args = []
for gamma in gammas:
    for nonlinearity_arg in nonlinearity_args:
        args.append((gamma, nonlinearity_arg))

def grid(arg):
    gamma, rstep = arg
    output_params = {'num_iterations':rstep}
    net_configs = {'gamma':gamma, 'ca3_kwargs':{'output_params': output_params}}
    net = STDP_SR(
        num_states=2, gamma=gamma, ca3_kwargs=net_configs['ca3_kwargs']
        )
    t_error, m_error, row_norm, _ = eval(net, datasets, print_every_steps=100)
    return gamma, rstep, np.mean(m_error, axis=0)[-1]

results = {}
results['gammas'] = []
results['rsteps'] = []
results['vals'] = []
job_results = Parallel(n_jobs=n_jobs)(delayed(grid)(arg) for arg in args)
for res in job_results:
    gamma, rstep, val = res
    results['gammas'].append(gamma)
    results['rsteps'].append(rstep)
    results['vals'].append(val)
with open(f'{experiment_dir}results.p', 'wb') as f:
    pickle.dump(results, f)

