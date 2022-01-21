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

from datasets import inputs, sf_inputs_discrete
from sr_model.models.models import AnalyticSR, STDP_SR, MLP
from train_td import train

device = 'cpu'

def run(
    save_path, net, dataset, dataset_config, print_file=None,
    print_every_steps=500, buffer_batch_size=32, buffer_size=5000, gamma=0.4,
    train_net=False, summ_write=True, test_over_all=True,
    test_batch_size=32
    ):

    # Initializations
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    buffer = ReplayMemory(buffer_size)
    if summ_write:
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

    if train_net: # Not very clean, but trains and returns the net
        datasets = [dataset]
        dc = deepcopy(dataset_config)
        dc['num_steps'] = min(2201, dc['num_steps'])
        for key in dc.keys():
            dc[key] = [dc[key]]
        datasets_config_ranges = [dc]
        net, return_error = train(
            save_path + 'training/', net, datasets, datasets_config_ranges,
            train_steps=6, early_stop=False, print_every_steps=1 
            )
        return None, None, None, net
    
    dset = dataset(**dataset_config)
    dg_inputs = torch.from_numpy(dset.dg_inputs.T).float().to(device).unsqueeze(1)
    dg_modes = torch.from_numpy(dset.dg_modes.T).float().to(device).unsqueeze(1)
    prev_input = None
    outputs = []

    with torch.no_grad():
        for step in np.arange(dg_inputs.shape[0]):
            start_time = time.time()
            dg_input = dg_inputs[step]
            dg_input = dg_input.unsqueeze(0)
    
            # Get net response and update model
            if step ==  0:
                _, out = net(dg_input, reset=True)
            else:
                _, out = net(dg_input, reset=False)
            outputs.append(out.detach().numpy().squeeze())
    
            if step == 0:
                prev_input = dg_input.detach()
                continue
    
            buffer.push((prev_input.detach(), dg_input.detach()))
    
            # Calculate error
            if test_over_all:
                all_transitions = buffer.memory
                all_s = torch.stack([t[0] for t in all_transitions]).squeeze(1)
                all_next_s = torch.stack([t[1] for t in all_transitions]).squeeze(1)
            else:
                transitions = buffer.sample(min(step, test_batch_size))
                all_s = torch.stack([t[0] for t in transitions], dim=2).squeeze(0)
                all_next_s = torch.stack([t[1] for t in transitions], dim=2).squeeze(0)

            test_phi = all_s.squeeze(1)
            test_psi_s = torch.stack(
                [net(s.view(1,1,-1), reset=False, update=False)[1].squeeze()\
                    for s in all_s.squeeze(0)]
                ).unsqueeze(0)
            test_psi_s_prime = torch.stack(
                [net(s.view(1,1,-1), reset=False, update=False)[1].squeeze()\
                    for s in all_next_s.squeeze(0)]
                ).unsqueeze(0)
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
                        if summ_write: writer.add_scalar(name+'_std', std, step)
                    else:
                        mean = torch.mean(param)
                    if summ_write: writer.add_scalar(name, mean, step)
        
                if summ_write: writer.add_scalar('loss_train', running_loss, step)
       
                print('', flush=True, file=print_file)
                print(
                    '[{:5d}] loss: {:0.4f}'.format(step + 1, running_loss),
                    file=print_file
                    )
                print(
                    'Time per step {:0.3f}s, net {:0.3f}s'.format(time_step, time_net),
                     file=print_file
                    )
                model_path = os.path.join(save_path, 'model.pt')
                if summ_write: torch.save(net.state_dict(), model_path)
                time_step = 0
                prev_running_loss = running_loss
                running_loss = 0.0
                grad_avg = 0

        # Get chance-level performance at the end of the walk
        try:
            feature_map = dset.feature_maker.feature_map
        except:
            feature_map = np.eye(dset.num_states)
        num_features = feature_map.shape[1]
        phi = feature_map[:,np.random.choice(num_features, size=96)]
        phi_prime = feature_map[:,np.random.choice(num_features, size=96)]
        psi = net(
            torch.tensor(phi.T).float().unsqueeze(1),
            reset=False, update=False
            )[1]
        psi_prime = net(
            torch.tensor(phi_prime.T).float().unsqueeze(1),
            reset=False, update=False
            )[1]
        td_error = (torch.tensor(phi.T) + gamma*psi_prime - psi).numpy()
        chance_mse = np.mean(np.mean(np.square(td_error), axis=1))
        if summ_write: writer.add_scalar('chance_loss', chance_mse, step)
        print(f'Chance: {chance_mse}')

    if summ_write: writer.close()
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
    save_path = '../trained_models/rnn_td_loss/'

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
    
    net = AnalyticSR(
        num_states=input_size, gamma=0.6,
        ca3_kwargs={
            'use_dynamic_lr':False, 'lr': -0.002046,
            'output_params':{
                'num_iterations':25, 'nonlinearity': 'clamp',
                'nonlinearity_args': [-0.055744, 2.045]
                }
            }
        )
    _ = run(
        'test/', net, dataset, dataset_config,
        gamma=0.6,
        train_net=False, test_over_all=False
        )
