import numpy as np
import argparse
import time
import os
from itertools import chain
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR

model_file = 'model.pt'
save_path = './trained_models/'
model_path = save_path + model_file

device = 'cpu'

# Dataset Configs
num_steps = 100
num_states = 16
dataset = inputs.Sim1DWalk
dataset_config = {
    'num_steps': num_steps, 'left_right_stay_prob': [1, 1, 1],
    'num_states': num_states
    }

# Init net
net = STDP_SR(num_states=num_states, gamma=0.5)
net.load_state_dict(torch.load(model_path))
net.ca3.set_leaky_slope(0)
net.ca3._init_ideal()
net.ca3.debug_print = True
net.ca3.gamma_M0 = 0.1

# Make input
input = dataset(**dataset_config)
dg_inputs = torch.from_numpy(input.dg_inputs.T).float().to(device).unsqueeze(1)
dg_modes = torch.from_numpy(input.dg_modes.T).float().to(device).unsqueeze(1)

with torch.no_grad():
    _, outputs = net(dg_inputs, dg_modes, reset=True)

plt.figure()
plt.plot(net.ca3.get_stdp_kernel())
plt.show()
fig, axs = plt.subplots(1, 2)
im0 = axs[0].imshow(net.ca3.get_real_T())
axs[0].set_title("Real T")
im1 = axs[1].imshow(net.ca3.get_T().detach().numpy())
axs[1].set_title("Estimated T")
plt.colorbar(mappable=im0, ax=axs[0])
plt.colorbar(mappable=im1, ax=axs[1])
plt.show()

import pdb; pdb.set_trace()
