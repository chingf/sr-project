import numpy as np
import argparse
import time
import os
from itertools import chain
import matplotlib.pyplot as plt
import multiprocessing as mp

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR

def get_kernel_params(tau_0, A_0, tau_offset):
    area = A_0*tau_0
    new_tau = tau_0 + tau_offset
    new_A = area/new_tau
    return new_A, new_tau

def get_symm_kernel_params(tau_0, A_0, tau_offset):
    area = A_0*tau_0
    new_tau = tau_0 + tau_offset
    new_A = (area/2)/new_tau
    return new_A, new_tau

def get_stdp_kernel(
    A_pos, tau_pos, A_neg, tau_neg, kernel_len
    ):
    """ Returns plasticity kernel for plotting or debugging. """

    k = np.zeros(kernel_len)
    half_len = kernel_len//2
    scaling = 1
    k[:half_len] = scaling*A_neg * np.exp(
        np.arange(-half_len, 0)/tau_neg
        )
    k[-half_len-1:] = scaling*A_pos * np.exp(
        -1*np.arange(half_len+1)/tau_pos
        )
    return k

# Order: [A_pos, tau_pos, A_neg, tau_neg]
A_0 = 0.5
tau_0 = 1.15
set0 = get_kernel_params(tau_0, A_0, -0.5)
set1 = get_kernel_params(tau_0, A_0, -0.25)
set2 = get_kernel_params(tau_0, A_0, 0.5)
set3 = get_kernel_params(tau_0, A_0, 0.75)
set4 = get_symm_kernel_params(tau_0, A_0, -0)

#params = [
#    [A_0, tau_0, 0, 1],
#    [set0[0], set0[1], 0, 1],
#    [set1[0], set1[1], 0, 1],
#    [set2[0], set2[1], 0, 1],
#    [set3[0], set3[1], 0, 1],
#    [set4[0], set4[1], set4[0], set4[1]],
#    ]

#params = [
#    [A_0, tau_0, 0, 1],
#    [A_0, set0[1], 0, 1],
#    [A_0, set1[1], 0, 1],
#    [A_0, set2[1], 0, 1],
#    [A_0, set3[1], 0, 1],
#    [A_0, set4[1], A_0, set4[1]],
#    ]
A_0 = 0.35
sets = [0.2, 0.4, 0.6, 0.8]
params = [
    [A_0, sets[0], A_0, sets[0]],
    [A_0, sets[1], A_0, sets[1]],
    [A_0, sets[2], A_0, sets[2]],
    [A_0, sets[3], A_0, sets[3]]
    ]


errs = []

device = 'cpu'

# Dataset Configs
dset = inputs.RBYCacheWalk(
    num_spatial_states=25*25,
    downsample_factor=None,
    skip_frame=0.7
    )

def eval_model(args):
    param = args

    net = STDP_SR(num_states=692, gamma=0.4)
    net.ca3.reset_trainable_ideal()
    net.ca3.set_differentiability(False)

    nn.init.constant_(net.ca3.A_pos, param[0])
    nn.init.constant_(net.ca3.tau_pos, param[1])
    nn.init.constant_(net.ca3.A_neg, param[2])
    nn.init.constant_(net.ca3.tau_neg, param[3])

    with torch.no_grad():
        dg_inputs = torch.from_numpy(dset.dg_inputs.T).float().to(device).unsqueeze(1)
        dg_modes = torch.from_numpy(dset.dg_modes.T).float().to(device).unsqueeze(1)
        _, outputs = net(dg_inputs, dg_modes, reset=True)
        est_T = net.ca3.get_T().detach().numpy()
        real_T = net.ca3.get_real_T()
        err = np.mean(np.abs(est_T - real_T))

    results = [param, est_T, err]
    return results

result = []
for args in params:
    print(args)
    result.append(eval_model(args))

import pickle
with open("evaled3.p","wb") as f:
    pickle.dump(result, f)

import pdb; pdb.set_trace()
