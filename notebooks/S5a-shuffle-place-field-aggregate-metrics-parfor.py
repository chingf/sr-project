import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import re
import sys
from copy import deepcopy
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import re
import pandas as pd
import seaborn as sns
from math import ceil

root = os.path.dirname(os.path.abspath(os.curdir))
sys.path.append(root)
from datasets import inputs, sf_inputs_discrete
from run_td_rnn import run as run_rnn
from run_td_linear import run as run_linear
from utils import get_field_metrics

device = 'cpu'

# PARAMETERS FOR SCRIPT
save_field_info = True
reload_field_info = True
nshuffles = 40
n_jobs = 56
iters = 3
num_states = 14*14
num_steps = 5001
root_dir = "../../engram/Ching/03_hannah_dset_revisions/"
arena_length = 14
model = 'shuffle'
gammas = [0.4, 0.6, 0.8]

def get_sparsity(key):
    p = re.compile('.*sparsity(.+?)\/.*')
    if 'sigma' in key:
        m = p.match(key)
        return m.group(1)
    else:
        return '0'

def get_sigma(key):
    p = re.compile('.*sigma(.+?)\/.*')
    if 'sigma' in key:
        m = p.match(key)
        return m.group(1)
    else:
        return '0'

def collect_metrics(args):
    sigma, sparsity, model = args
    init_sparsities = []
    sigmas = []
    final_sparsities = []
    fieldsizes = []
    nfields = []
    onefields = []
    zerofields = []
    fieldsizekls = []
    nfieldkls = []

    gamma_dir = f'{root_dir}sparsity{sparsity}/sigma{sigma}/{gamma}/'
    linear_model_dir = f'{gamma_dir}linear/'
    for _iter in os.listdir(linear_model_dir):
        linear_iter_dir = f'{linear_model_dir}/{_iter}/'
        model_dir = f'{gamma_dir}{model}/{_iter}/'
        os.makedirs(model_dir, exist_ok=True)
        linear_iter_dir = linear_model_dir + _iter + '/'
        iter_dir = model_dir + _iter + '/'
        results_path = linear_iter_dir + 'results.p'
        if not os.path.isfile(results_path): continue
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        dset = results['dset']
        linear_net = results['net']
        linear_M = linear_net.M.numpy().squeeze()
        linear_M = np.random.choice(linear_M.flatten(), size=linear_M.shape)
        outputs = dset.dg_inputs.T @ linear_M
        _fieldsize, _nfield, _onefield, _zerofield, _fieldsizekl, _nfieldkl =\
            get_field_metrics(
                outputs, dset, arena_length,
                nshuffles=nshuffles,
                save_field_info=save_field_info, reload_field_info=reload_field_info,
                save_path=iter_dir
                )
        init_sparsities.append(sparsity)
        sigmas.append(sigma)
        final_sparsities.append(dset.feature_maker.post_smooth_sparsity)
        fieldsizes.append(_fieldsize)
        nfields.append(_nfield)
        onefields.append(_onefield)
        zerofields.append(_zerofield)
        fieldsizekls.append(_fieldsizekl)
        nfieldkls.append(_nfieldkl)
    return init_sparsities, sigmas, final_sparsities,\
        fieldsizes, nfields, onefields, zerofields, fieldsizekls, nfieldkls

from joblib import Parallel, delayed

# Arguments
spatial_sigmas = [0.0, 1.0, 2.0, 3.0]
sparsity_range = [[0.001, 0.2], [0.001, 0.1], [0.001, 0.04], [0.001, 0.023]]
spatial_sigmas.extend([0.25, 0.5, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75, 3.25])
sparsity_range.extend([
    [0.001, 0.19], # 0.25
    [0.001, 0.15], # 0.5
    [0.001, 0.09], # 1.25
    [0.001, 0.05], # 1.5
    [0.001, 0.045], # 1.75
    [0.001, 0.037], # 2.25
    [0.001, 0.03], # 2.5
    [0.001, 0.025], # 2.75
    [0.001, 0.021], # 3.25
    ])

# Run arguments through each gamma
for gamma in gammas:
    args = []
    for idx, spatial_sigma in enumerate(spatial_sigmas):
        _range = sparsity_range[idx]
        sparsity_ps = np.linspace(_range[0], _range[1], num=20, endpoint=True)
        for sparsity_p in sparsity_ps:
            args.append([spatial_sigma, sparsity_p, gamma])
    init_sparsities = []
    sigmas = []
    final_sparsities = []
    fieldsizes = []
    nfields = []
    onefields = []
    zerofields = []
    fieldsizekls = []
    nfieldkls = []

    job_results = Parallel(n_jobs=n_jobs)(delayed(collect_metrics)(arg) for arg in args)
    for res in job_results:
        if res is None: continue
        init_sparsities.extend(res[0])
        sigmas.extend(res[1])
        final_sparsities.extend(res[2])
        fieldsizes.extend(res[3])
        nfields.extend(res[4])
        onefields.extend(res[5])
        zerofields.extend(res[6])
        fieldsizekls.extend(res[7])
        nfieldkls.extend(res[8])
    
    results = {
        'gamma': gamma, 'arena_length': arena_length,
        'init_sparsities': init_sparsities,
        'sigmas': sigmas, 'final_sparsities': final_sparsities,
        'fieldsizes': fieldsizes,
        'nfields': nfields, 'onefields': onefields,
        'zerofields': zerofields, 'fieldsizekls': fieldsizekls,
        'nfieldkls': nfieldkls
        }
    with open(root_dir + f'5a_{model}_results_gamma{gamma}.p', 'wb') as f:
        pickle.dump(results, f)

