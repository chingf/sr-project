import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import re
import sys
from joblib import Parallel, delayed
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.abspath(os.curdir))
sys.path.append(root)

from sr_model.models.models import AnalyticSR, STDP_SR
from datasets import inputs
import seaborn as sns
import pandas as pd

root_dir = '../../engram/Ching/03_hannah_dset/'
device = 'cpu'
n_jobs = 56

def format_model_name(key): #TODO
    if key == 'rnn_tanh':
        return 'RNN-Tanh'
    elif key == 'rnn_none':
        return 'RNN-None'
    elif key == 'rnn':
        return 'RNN-SF'
    elif key == 'linear':
        return 'Linear'
    else:
        raise ValueError("Invalid key.")

def get_sparsity(key):
    p = re.compile('.*sparsity(.+?)\/.*')
    if 'sparsity' in key:
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

def grid_train(args):
    sparsity_dir, sigma_dir, gamma_dir = args
    path = f'{root_dir}{sparsity_dir}/{sigma_dir}/{gamma_dir}/'
    print(f'Processing {sparsity_dir}/{sigma_dir}/{gamma_dir}/')
    init_sparsities = []
    final_sparsities = []
    sigmas = []
    gammas = []
    models = []
    means = []
    stds = []

    # Get dataset parameters
    init_sparsity = get_sparsity(path)
    sigma = get_sigma(path)
    gamma = float(gamma_dir)

    for model_dir in os.listdir(path):
        try:
            model = format_model_name(model_dir)
        except:
            continue
        model_path = f'{path}{model_dir}/'

        for iter_dir in os.listdir(model_path):
            iter_path = model_path + iter_dir + '/'
            if not os.path.isfile(iter_path + 'results.p'):
                continue
            results = pickle.load(open(iter_path + 'results.p', 'rb'))
            feature_maker = results['dset'].feature_maker
            feature_map = feature_maker.feature_map # (feature dim, num states)
            init_sparsities.append(init_sparsity)
            final_sparsities.append(feature_maker.post_smooth_sparsity)
            sigmas.append(sigma)
            gammas.append(gamma)
            models.append(model)
            mean_feature = np.mean(feature_map, axis=1)
            std_feature = np.mean(np.abs(feature_map - mean_feature[:,None]), axis=1)
            means.append(np.linalg.norm(mean_feature))
            stds.append(np.linalg.norm(std_feature))

    return init_sparsities, final_sparsities,\
        sigmas, gammas, models, means, stds

args = []
for sparsity_dir in os.listdir(root_dir):
    if 'sparsity' not in sparsity_dir: continue
    for sigma_dir in os.listdir(f'{root_dir}{sparsity_dir}/'):
        if 'sigma' not in sigma_dir: continue
        for gamma_dir in os.listdir(f'{root_dir}{sparsity_dir}/{sigma_dir}/'):
            if 'DS' in gamma_dir: continue 
            if gamma_dir != '0.75': continue
            args.append((sparsity_dir, sigma_dir, gamma_dir))
init_sparsities = []
final_sparsities = []
sigmas = []
gammas = []
models = []
means = []
stds = []
job_results = Parallel(n_jobs=n_jobs)(delayed(grid_train)(arg) for arg in args)
for res in job_results:
    _init_sparsities, _final_sparsities,\
        _sigmas, _gammas, _models, _means, _stds = res
    init_sparsities.extend(_init_sparsities)
    final_sparsities.extend(_final_sparsities)
    sigmas.extend(_sigmas)
    gammas.extend(_gammas)
    models.extend(_models)
    means.extend(_means)
    stds.extend(_stds)
results = {
    'init_sparsities': init_sparsities,
    'final_sparsities': final_sparsities,
    'sigmas': sigmas,
    'gammas': gammas,
    'models': models,
    'means': means,
    'stds': stds,
    }
with open(f'{root_dir}td_chance.p', 'wb') as f: #TODO
    pickle.dump(results, f)
