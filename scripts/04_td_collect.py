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

def format_model_name(key):
    if key == 'rnn':
        return 'RNN-SF'
    elif key == 'rnn_oja':
        return 'RNN-Oja'
    elif key == 'linear':
        return 'Linear'
    elif key == 'mlp':
        return 'MLP with Replay Buffer'
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
    final_losses = []
    chance_losses = []

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
            for file in os.listdir(iter_path):
                if 'tfevents' not in file: continue
                tfevents_file = iter_path + '/' + file
                event_acc = EventAccumulator(tfevents_file)
                event_acc.Reload()
                try:
                    scalar_events = event_acc.Scalars('loss_train')
                except:
                    continue
                values = np.array([event.value for event in scalar_events])
                if np.any(np.isnan(values)): continue
                results = pickle.load(open(iter_path + 'results.p', 'rb'))
                init_sparsities.append(init_sparsity)
                final_sparsities.append(
                    results['dset'].feature_maker.post_smooth_sparsity
                    )
                sigmas.append(sigma)
                gammas.append(gamma)
                models.append(model)
                final_losses.append(values[-1])
                if 'mlp' in model_dir:
                    chance_losses.append(np.nan)
                else:
                    chance_losses.append(
                        event_acc.Scalars('chance_loss')[-1].value
                        )
                break

    return init_sparsities, final_sparsities,\
        sigmas, gammas, models, final_losses, chance_losses

args = []
for sparsity_dir in os.listdir(root_dir):
    if 'sparsity' not in sparsity_dir: continue
    for sigma_dir in os.listdir(f'{root_dir}{sparsity_dir}/'):
        for gamma_dir in os.listdir(f'{root_dir}{sparsity_dir}/{sigma_dir}/'):
            args.append((sparsity_dir, sigma_dir, gamma_dir))
init_sparsities = []
final_sparsities = []
sigmas = []
gammas = []
models = []
final_losses = []
chance_losses = []
job_results = Parallel(n_jobs=n_jobs)(delayed(grid_train)(arg) for arg in args)
for res in job_results:
    _init_sparsities, _final_sparsities,\
        _sigmas, _gammas, _models, _final_losses, _chance_losses = res
    init_sparsities.extend(_init_sparsities)
    final_sparsities.extend(_final_sparsities)
    sigmas.extend(_sigmas)
    gammas.extend(_gammas)
    models.extend(_models)
    final_losses.extend(_final_losses)
    chance_losses.extend(_chance_losses)
results = {
    'init_sparsities': init_sparsities,
    'final_sparsities': final_sparsities,
    'sigmas': sigmas,
    'gammas': gammas,
    'models': models,
    'final_losses': final_losses,
    'chance_losses': chance_losses
    }
with open(f'{root_dir}td_results.p', 'wb') as f:
    pickle.dump(results, f)
