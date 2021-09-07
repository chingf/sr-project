import numpy as np
import argparse
import time
import os
from itertools import chain
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import cma

from datasets import inputs, sf_inputs_discrete
from sr_model.models.models import AnalyticSR, STDP_SR

device = 'cpu'

def train(
    save_path, net, datasets, datasets_config_ranges, print_file=None,
    train_steps=201, print_every_steps=50, early_stop=False
    ):

    # Initializations
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_datasets = len(datasets)
    writer = SummaryWriter(save_path)
    criterion = nn.MSELoss()
    lr=1E-3
    weight_decay = 0
    
    # Initialize CMA-ES optimizer
    parameter_names, parameter_init = flatten_parameters(net)
    es_optimizer = cma.CMAEvolutionStrategy(parameter_init, sigma0=0.3)
    
    # Loss reporting
    running_loss = 0.0
    prev_running_loss = np.inf
    time_step = 0
    time_net = 0
    grad_avg = 0
    end_training = False
    
    for step in range(train_steps):
        start_time = time.time()
    
        # Select dataset parameters and load dataset
        dataset = datasets[step % num_datasets]
        dataset_config_rang = datasets_config_ranges[step % num_datasets]
        dataset_config = {}
        for key in dataset_config_rang:
            num_samples = len(dataset_config_rang[key])
            sample_idx = np.random.choice(num_samples)
            dataset_config[key] = dataset_config_rang[key][sample_idx]
        input = dataset(**dataset_config)
        dg_inputs = torch.from_numpy(input.dg_inputs.T).float().to(device).unsqueeze(1)
        dg_modes = torch.from_numpy(input.dg_modes.T).float().to(device).unsqueeze(1)
        net.ca3.set_num_states(input.feature_maker.feature_dim)

        # Generate candidate solutions
        candidate_params = es_optimizer.ask()
    
        # Evaluate loss on each candidate
        losses = []
        for params in candidate_params:
            set_parameters(net, parameter_names, params)
            _, outputs = net(dg_inputs, dg_modes, reset=True)
            with torch.no_grad():
                all_s = dg_inputs[:-1]
                all_next_s = dg_inputs[1:]
                phi = all_s.squeeze(1)
                M = net.get_M(net.gamma)
                psi_s = torch.stack([s.squeeze() @ M for s in all_s])
                psi_s_prime = torch.stack([next_s.squeeze() @ M for next_s in all_next_s])
                value_function = psi_s
                exp_value_function = phi + net.gamma*psi_s_prime
                loss = criterion(value_function, exp_value_function)
            losses.append(loss.item())
    
        # Take optimization step based on losses
        es_optimizer.tell(candidate_params, losses)
    
        # Print statistics
        elapsed_time = time.time() - start_time
        time_step += elapsed_time
        time_net += elapsed_time
        running_loss += min(losses)
    
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
            print(net.ca3.state_count_init)
            print(net.ca3.update_term_scale)
            model_path = os.path.join(save_path, 'model.pt')
            torch.save(net.state_dict(), model_path)
            time_step = 0
            if (prev_running_loss < 1E-4) and (running_loss < 1E-4) and early_stop:
                end_training = True
            prev_running_loss = running_loss
            running_loss = 0.0
            grad_avg = 0
        if end_training:
            break
    
    writer.close()
    print('Finished Training\n', file=print_file)
    import pdb; pdb.set_trace()
    return net, prev_running_loss

def flatten_parameters(net):
    parameters = []
    names = []
    for name, param in net.named_parameters():
        if param.numel() != 1:
            raise NotImplementedError("Can't flatten parameter {}: numel={}"
                                      .format(name, param))
        if param.requires_grad:
            parameters.append(param.item())
            names.append(name)
    return names, parameters

def set_parameters(net, names, flattened_params):
    with torch.no_grad():
        for name, param in zip(names, flattened_params):
            eval('net.{}.fill_(param)'.format(name)) #TODO: do this better
    return net

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
    save_path = './trained_models/td_trained/'
    datasets = [sf_inputs_discrete.Sim2DLevyFlight]
    datasets_config_ranges = [{
        'num_steps': [500], 'walls': [7],
        'feature_maker_kwargs': [{'feature_dim': 64, 'feature_vals':[0,0.5,1]}]
        }]
    output_params = {
        'num_iterations':15, 'input_clamp':15, 'nonlinearity': None,
        'transform_activity': True, 'clamp_activity': False
        }
    net = AnalyticSR(
        num_states=2, gamma=0.95,
        ca3_kwargs={'use_dynamic_lr':True, 'lr': 1., 'parameterize': True}
        )
    train(
        save_path, net, datasets, datasets_config_ranges, train_steps=801,
        early_stop=False
        )