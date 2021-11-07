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
from sr_model.models.models import AnalyticSR, STDP_SR, Linear, MLP
from datasets import inputs, sf_inputs_discrete
from run_td_rnn import run as run_rnn
from run_td_mlp import run as run_mlp
from run_td_linear import run as run_linear
from utils import get_field_metrics

device = 'cpu'

# PARAMETERS FOR SCRIPT
save_field_info = True
nshuffles = 40
n_jobs = 30

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
    sparsity, sigma, model = args
    init_sparsities = []
    sigmas = []
    final_sparsities = []
    fieldsizes = []
    nfields = []
    onefields = []
    zerofields = []
    nfieldkls = []

    gamma_dir = f'{root_dir}{sparsity}/{sigma}/{gamma}/'
    if 'hopfield' in model:
        model_dir = f'{gamma_dir}hopfield/'
    else:
        model_dir = f'{gamma_dir}rnn_fixedlr_alpha/'
    if not os.path.isdir(model_dir): return
    print(f"Processing: {model_dir}\n")
    for _iter in os.listdir(model_dir):
        iter_dir = model_dir + _iter + '/'
        results_path = iter_dir + 'results.p'
        if not os.path.isfile(results_path): continue
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        if 'hopfield' in model_dir:
            outputs = results['outputs'].detach().numpy().squeeze()
        else:
            outputs = results['outputs']
        dset = results['dset']
        _fieldsize, _nfield, _onefield, _zerofield, _nfieldkl = get_field_metrics(
            outputs, dset, arena_length,
            nshuffles=nshuffles, save_field_info=save_field_info,
            save_path=iter_dir
            )

        init_sparsities.append(float(get_sparsity(model_dir)))
        sigmas.append(float(get_sigma(model_dir)))
        final_sparsities.append(dset.feature_maker.post_smooth_sparsity)
        fieldsizes.append(_fieldsize)
        nfields.append(_nfield)
        onefields.append(_onefield)
        zerofields.append(_zerofield)
        nfieldkls.append(_nfieldkl)
    return init_sparsities, sigmas, final_sparsities, fieldsizes,\
        nfields, onefields, zerofields, nfieldkls

from joblib import Parallel, delayed

root_dir = "../trained_models/03_td_discrete_corr/"
root_dir = "../../engram/Ching/03_td_discrete_corr/"

arena_length = 20

for model in ['rnn']:
    for gamma in [0.75, 0.8, 0.6, 0.85]:
        init_sparsities = []
        sigmas = []
        final_sparsities = []
        fieldsizes = []
        nfields = []
        onefields = []
        zerofields = []
        nfieldkls = []
        
        args = []
        for sparsity in os.listdir(root_dir):
            for sigma in os.listdir(f'{root_dir}{sparsity}/'):
                args.append([sparsity, sigma, model])
        job_results = Parallel(n_jobs=n_jobs)(delayed(collect_metrics)(arg) for arg in args)
        for res in job_results:
            init_sparsities.extend(res[0])
            sigmas.extend(res[1])
            final_sparsities.extend(res[2])
            fieldsizes.extend(res[3])
            nfields.extend(res[4])
            onefields.extend(res[5])
            zerofields.extend(res[6])
            nfieldkls.extend(res[7])
        
        results = {
            'gamma': gamma, 'arena_length': arena_length,
            'init_sparsities': init_sparsities,
            'sigmas': sigmas, 'final_sparsities': final_sparsities,
            'fieldsizes': fieldsizes,
            'nfields': nfields, 'onefields': onefields,
            'zerofields': zerofields, 'nfieldkls': nfieldkls
            }
        with open(root_dir + '5a_{model}_results_gamma{gamma}.p', 'wb') as f:
            pickle.dump(results, f)

