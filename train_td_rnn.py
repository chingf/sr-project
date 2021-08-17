import numpy as np
import argparse
import random
import time
import os
from itertools import chain
from copy import deepcopy
from collections import namedtuple, deque
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import cma

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR, MLP

device = 'cpu'

def train(
    save_path, net, dataset, dataset_config, print_file=None,
    print_every_steps=50, buffer_batch_size=32, buffer_size=5000, gamma=0.4
    ):

    # Initializations
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    buffer = ReplayMemory(buffer_size)
    writer = SummaryWriter(save_path)
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    lr=1E-3
    weight_decay = 0

    # Loss reporting
    running_loss = 0.0
    prev_running_loss = np.inf
    time_step = 0
    time_net = 0
    grad_avg = 0
    
    dset = dataset(**dataset_config)
    dg_inputs = torch.from_numpy(dset.dg_inputs.T).float().to(device).unsqueeze(1)
    dg_modes = torch.from_numpy(dset.dg_modes.T).float().to(device).unsqueeze(1)
    prev_input = None

    for step in np.arange(dg_inputs.shape[0]):
        start_time = time.time()
        dg_input = dg_inputs[step]
        dg_input = dg_input.unsqueeze(0)

        net(dg_input, reset=False)

        if step == 0:
            prev_input = dg_input.detach()
            continue

        buffer.push((prev_input.detach(), dg_input.detach()))
        with torch.no_grad():
            all_transitions = buffer.memory
            all_s = torch.stack([t[0] for t in all_transitions]).squeeze(1)
            all_next_s = torch.stack([t[1] for t in all_transitions]).squeeze(1)
            test_phi = all_s.squeeze(1)
            M = net.get_M()
            test_psi_s = torch.stack([s.squeeze() @ M for s in all_s])
            test_psi_s_prime = torch.stack([next_s.squeeze() @ M for next_s in all_next_s])
            test_value_function = test_psi_s
            test_exp_value_function = test_phi + gamma*test_psi_s_prime
            test_loss = criterion(test_value_function, test_exp_value_function)

        prev_input = dg_input.detach()

        # Print statistics
        elapsed_time = time.time() - start_time
        time_step += elapsed_time
        time_net += elapsed_time
        running_loss += test_loss
    
        if step % print_every_steps == 0:
            time_step /= print_every_steps
            if step > 0:
                running_loss /= print_every_steps
    
            for name, param in chain(net.named_parameters(), net.named_buffers()):
                if torch.numel(param) > 1:
                    std, mean = torch.std_mean(param)
                    writer.add_scalar(name+'_std', std, step)
                else:
                    mean = torch.mean(param)
                writer.add_scalar(name, mean, step)
    
            writer.add_scalar('loss_train', running_loss, step)
   
            print('', flush=True, file=print_file)
            print(
                '[{:5d}] loss: {:0.3f}'.format(step + 1, running_loss),
                file=print_file
                )
            print(
                'Time per step {:0.3f}s, net {:0.3f}s'.format(time_step, time_net),
                 file=print_file
                )
            model_path = os.path.join(save_path, 'model.pt')
            torch.save(net.state_dict(), model_path)
            time_step = 0
            prev_running_loss = running_loss
            running_loss = 0.0
            grad_avg = 0

    
    writer.close()
    print('Finished Training\n', file=print_file)
    return net, prev_running_loss

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, _list):
        """Save a transition _list is (state, next_state)"""
        self.memory.append(_list)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__ == "__main__":
    save_path = '../trained_models/rnn_td_loss/'
    dataset = inputs.Sim1DWalk
    dataset_config = {
        'num_steps': 8000, 'left_right_stay_prob': [1,1,1], 'num_states': 10
        }
    net = STDP_SR(
        num_states=dataset_config['num_states'], gamma=0.4,
        ca3_kwargs={'gamma_T':1}
        )
    net.ca3.set_differentiability(False)
    state_dict_path = './trained_models/baseline/model.pt'
    net.load_state_dict(torch.load(state_dict_path))

    net = AnalyticSR(num_states=dataset_config['num_states'], gamma=0.4)
    train(save_path, net, dataset, dataset_config)
