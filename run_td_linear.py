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

from datasets import inputs, sf_inputs_discrete
from sr_model.models.models import AnalyticSR, STDP_SR, Linear

device = 'cpu'

def run(
    save_path, net, dataset, dataset_config, print_file=None,
    print_every_steps=500, buffer_batch_size=1, buffer_size=5000, gamma=0.4,
    lr=1E-3, test_over_all=True, test_batch_size=32
    ):

    # Initializations
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    buffer = ReplayMemory(buffer_size)
    writer = SummaryWriter(save_path)
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()
    weight_decay = 0

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

    with torch.no_grad():
        for step in np.arange(inputs.shape[0]):
            start_time = time.time()
            input = inputs[step]
            input = input.unsqueeze(0)
    
            # Get net response
            out = net(input, update=False)
            outputs.append(out.detach().numpy().squeeze())
    
            if step == 0:
                prev_input = input.detach()
                continue
    
            buffer.push((prev_input.detach(), input.detach()))
    
            # Update Model
            if buffer_batch_size == 1: # Don't sample-- use current observation
                transitions = [buffer.memory[-1]]
            else:
                transitions = buffer.sample(min(step, buffer_batch_size))
    
            states = torch.stack([t[0] for t in transitions]).squeeze(1)
            next_states = torch.stack([t[1] for t in transitions]).squeeze(1)
        
            phi = states
            psi_s = net(states, update=False)
            psi_s_prime = net(next_states, update=False)
            value_function = psi_s
            expected_value_function = phi + gamma*psi_s_prime
            errors = expected_value_function - value_function
            for idx, error in enumerate(errors):
                net.M[0,:,:] = net.M[0,:,:] + lr *error*states[idx].t()
    
            # Calculate error 
            if test_over_all:
                all_transitions = buffer.memory
                all_s = torch.stack([t[0] for t in all_transitions]).squeeze(1)
                all_next_s = torch.stack([t[1] for t in all_transitions]).squeeze(1)
            else:
                transitions = buffer.sample(min(step, test_batch_size))
                all_s = torch.stack([t[0] for t in transitions], dim=2).squeeze(0)
                all_next_s = torch.stack([t[1] for t in transitions], dim=2).squeeze(0)
            test_phi = all_s
            test_psi_s = net(all_s, update=False)
            test_psi_s_prime = net(all_next_s, update=False)
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
   
        # Get chance-level performance at the end of the walk
        transitions = buffer.sample(min(step, 96))
        phi = torch.stack([t[0] for t in transitions], dim=2).squeeze(0)
        transitions = buffer.sample(min(step, 96))
        phi_prime = torch.stack([t[1] for t in transitions], dim=2).squeeze(0)
        psi = net(phi, update=False)
        psi_prime = net(phi_prime, update=False)
        chance_mse = criterion(psi, phi + gamma*psi_prime)
        writer.add_scalar('chance_loss', chance_mse, step)
        writer.add_scalar('model_lr', lr, step)
    
    writer.close()
    print('Finished Training\n', file=print_file)
    return np.array(outputs), prev_running_loss, dset, net

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
    save_path = './trained_models/linear/'

    sparsity_p = 0.125
    spatial_sigma = 1.5
    num_states = input_size = 20*20
    num_steps = 4000
    dataset = sf_inputs_discrete.Sim2DWalk
    feature_maker_kwargs = {
        'feature_dim': num_states, 'feature_type': 'correlated_distributed',
        'feature_vals_p': [1-sparsity_p, sparsity_p], 'feature_vals': None,
        'spatial_sigma': spatial_sigma
        }
    dataset_config = {
        'num_steps': num_steps, 'feature_maker_kwargs': feature_maker_kwargs,
        'num_states': num_states
        }
    net = Linear(input_size=input_size)
    run(
        save_path, net, dataset, dataset_config, gamma=0.6,
        test_over_all=False, lr=1E-3
        )

