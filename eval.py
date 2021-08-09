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

model_file = 'model_0.pt'
save_path = './trained_models/'
model_path = save_path + model_file

device = 'cpu'

# Dataset Configs
datasets = [ 
    inputs.Sim1DWalk(num_steps=2000, left_right_stay_prob=[5,1,1], num_states=64),
    inputs.Sim2DWalk(num_steps=2000, num_states=64),
    inputs.Sim2DLevyFlight(num_steps=2000, walls=7)
    ]

# Init net
net = STDP_SR(num_states=64, gamma=0.4)
net.load_state_dict(torch.load(model_path))
net.ca3.set_differentiability(False)

criterion = nn.MSELoss(reduction='none')

with torch.no_grad():
    for dset in datasets:
        dg_inputs = torch.from_numpy(dset.dg_inputs.T).float().to(device).unsqueeze(1)
        dg_modes = torch.from_numpy(dset.dg_modes.T).float().to(device).unsqueeze(1)
        _, outputs = net(dg_inputs, dg_modes, reset=True)
        est_T = net.ca3.get_T().detach().numpy()
        real_T = net.ca3.get_real_T()
        err = np.mean(np.abs(est_T - real_T))
        print(err)

