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

experiment_dir = f'{configs.engram_dir}02_learn_retrieve/'
experiment_dir = '../../engram/Ching/02_learn_retrieve/'
os.makedirs(experiment_dir, exist_ok=True)
n_jobs = 56
n_iters = 5

num_steps = 3001
num_states = 25
lr_probs = [[4,1,1]]
datasets = []
for _iter in range(n_iters):
    for lr_prob in lr_probs:
        dset = inputs.Sim1DWalk(
            num_steps=num_steps, num_states=num_states,
            left_right_stay_prob=lr_prob
            )
        datasets.append(dset)

learn_gammas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
retrieve_gammas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
nonlinearities = ['None', 'Tanh']
args = []
for learn_gamma in learn_gammas:
    for retrieve_gamma in retrieve_gammas:
        for nonlinearity in nonlinearities:
            args.append((learn_gamma, retrieve_gamma, nonlinearity))

def grid(arg):
    learn_gamma, retrieve_gamma, nonlinearity = arg

    # As Tanh
    rstep = int(np.log(1E-5)/np.log(retrieve_gamma))
    if nonlinearity == 'None':
        output_params = {}
    else:
        output_params = {
            'num_iterations':rstep,
            'nonlinearity': 'tanh', 'nonlinearity_args': 1.0
            }
    net_configs = {'gamma':learn_gamma, 'ca3_kwargs':{'output_params': output_params}}
    net = STDP_SR(
        num_states=2, gamma=learn_gamma, ca3_kwargs=net_configs['ca3_kwargs']
        )
    t_error, m_error, row_norm, _ = eval(
        net, datasets, print_every_steps=100, eval_gamma=retrieve_gamma
        )
    return learn_gamma, retrieve_gamma, np.mean(m_error, axis=0)[-1], nonlinearity

results = {}
results['learn_gammas'] = []
results['retrieve_gammas'] = []
results['vals'] = []
results['nonlinearity'] = []
job_results = Parallel(n_jobs=n_jobs)(delayed(grid)(arg) for arg in args)
for res in job_results:
    learn_gamma, retrieve_gamma, val, nonlinearity = res
    results['learn_gammas'].append(learn_gamma)
    results['retrieve_gammas'].append(retrieve_gamma)
    results['vals'].append(val)
    results['nonlinearity'].append(nonlinearity)
with open(f'{experiment_dir}results.p', 'wb') as f:
    pickle.dump(results, f)

