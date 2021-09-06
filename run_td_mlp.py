import numpy as np
import argparse
import random
import time
import os
from itertools import chain
from copy import deepcopy
from collections import namedtuple, deque

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import cma

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR, MLP

device = 'cpu'

def run(
    save_path, net, dataset, dataset_config, print_file=None,
    print_every_steps=50, buffer_batch_size=32, buffer_size=5000, gamma=0.4
    ):

    # Initializations
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    buffer = ReplayMemory(buffer_size)
    writer = SummaryWriter(save_path)
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    lr=1E-3
    weight_decay = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay) #Adam

    # Loss reporting
    running_loss = 0.0
    prev_running_loss = np.inf
    time_step = 0
    time_net = 0
    grad_avg = 0
    
    dset = dataset(**dataset_config)
    inputs = torch.from_numpy(dset.dg_inputs.T).float().to(device).unsqueeze(1)
    prev_input = None
    outputs = []

    for step in np.arange(inputs.shape[0]):
        start_time = time.time()
        input = inputs[step]
        input = input.unsqueeze(0)

        # Get net response
        with torch.no_grad():
            out = net(input)
            outputs.append(out.detach().numpy().squeeze())

        if step == 0:
            prev_input = input.detach()
            continue

        buffer.push((prev_input.detach(), input.detach()))

        # Update Model
        transitions = buffer.sample(min(step, buffer_batch_size))
        states = torch.stack([t[0] for t in transitions], dim=2).squeeze(0)
        next_states = torch.stack([t[1] for t in transitions], dim=2).squeeze(0)

        phi = states
        psi_s = net(states)
        psi_s_prime = net(next_states)
        value_function = psi_s
        expected_value_function = phi + gamma*psi_s_prime
    
        loss = criterion(value_function, expected_value_function)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate error
        with torch.no_grad():
            all_transitions = buffer.memory
            all_s = torch.stack([t[0] for t in all_transitions], dim=2).squeeze(0)
            all_next_s = torch.stack([t[1] for t in all_transitions], dim=2).squeeze(0)
            test_phi = all_s
            test_psi_s = net(all_s)
            test_psi_s_prime = net(all_next_s)
            test_value_function = test_psi_s
            test_exp_value_function = test_phi + gamma*test_psi_s_prime
            test_loss = criterion(test_value_function, test_exp_value_function)
    
        prev_input = input.detach()

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
    return np.array(outputs), prev_running_loss

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
    save_path = './trained_models/mlp/'
    dataset = inputs.Sim1DWalk
    dataset_config = {
        'num_steps': 8000, 'left_right_stay_prob': [1,1,1], 'num_states': 10
        }
    net = MLP(input_size=10, hidden_size=32)
    run(save_path, net, dataset, dataset_config)
