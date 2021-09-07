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

from datasets import inputs
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
    net.ca3.set_differentiability(False)
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

        # Generate candidate solutions
        candidate_params = es_optimizer.ask()
    
        # Evaluate loss on each candidate
        losses = [[] for _ in candidate_params]

        for _ in range(5):
            # Select dataset parameters and load dataset
            dataset = datasets[step % num_datasets]
            dataset_config_rang = datasets_config_ranges[step % num_datasets]
            dataset_config = {}
            for key in dataset_config_rang:
                num_samples = len(dataset_config_rang[key])
                sample_idx = np.random.choice(num_samples)
                dataset_config[key] = dataset_config_rang[key][sample_idx]
            input = dataset(**dataset_config)
            dg_inputs = torch.from_numpy(input.dg_inputs.T).float().to(device)
            dg_inputs = dg_inputs.unsqueeze(1)
            dg_modes = torch.from_numpy(input.dg_modes.T).float().to(device)
            dg_modes = dg_modes.unsqueeze(1)
            net.ca3.set_num_states(input.num_states)

            for idx, params in enumerate(candidate_params):
                set_parameters(net, parameter_names, params)

                # Feed inputs into network
                with torch.no_grad():
                    _, outputs = net(dg_inputs, dg_modes, reset=True)
                rnn_T = net.ca3.get_T().detach().numpy()
                est_T = net.ca3.get_ideal_T_estimate()
                error = np.mean(np.square(est_T - rnn_T))
                losses[idx].append(error)
        test = [l for l in losses]
        losses = [np.mean(loss) for loss in losses]
    
        # Take optimization step based on losses
        es_optimizer.tell(candidate_params, losses)
        set_parameters(net, parameter_names, candidate_params[np.argmin(losses)])
    
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
                '[{:5d}] loss: {:0.4f}'.format(step + 1, running_loss),
                file=print_file
                )
            print(
                'Time per step {:0.3f}s, net {:0.3f}s'.format(time_step, time_net),
                 file=print_file
                )
            print(net.ca3.output_param_scale)
            print(net.ca3.output_param_bias)
            print(net.ca3.update_clamp_a)
            print(net.ca3.update_clamp_b)
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

if __name__ == "__main__":
    save_path = './trained_models/0.9_short/'
    datasets = [
        inputs.Sim1DWalk,
        ]
    datasets_config_ranges = [
        {
        'num_steps': [3, 10, 20, 30, 40],
        'left_right_stay_prob': [[1, 1, 1], [7, 1, 1], [1, 4, 1]],
        'num_states': [5, 10, 15, 25]
        },
        ]
    output_params = {
        'num_iterations':10, 'input_clamp':100, 'nonlinearity': 'clamp',
        'transform_input': False
        }
    net_params = {
        'num_states':2, 'gamma':0.9,
        'ca3_kwargs':{'output_params':output_params}
        }
    net = STDP_SR(**net_params)
    train(
        save_path, net, datasets, datasets_config_ranges, train_steps=801,
        early_stop=False, print_every_steps=25
        )
    with open(save_path + "net_configs.p", 'wb') as f:
        import pickle
        pickle.dump(net_params, f)
