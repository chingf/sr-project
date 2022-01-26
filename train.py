import numpy as np
import argparse
import time
import shutil
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
    train_steps=201, print_every_steps=50, early_stop=False,
    return_test_error=False, train_M=False
    ):

    # Initializations
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_datasets = len(datasets)
    writer = SummaryWriter(save_path)
    lr=1E-3
    weight_decay = 0
    
    # Initialize CMA-ES optimizer
    parameter_names, parameter_init = flatten_parameters(net)
    es_optimizer = cma.CMAEvolutionStrategy(parameter_init, sigma0=0.3)
    
    # Loss reporting
    running_loss = 0.0
    running_mae_loss = 0.0
    prev_running_loss = np.inf
    prev_running_mae_loss = np.inf
    time_step = 0
    time_net = 0
    grad_avg = 0
    end_training = False
    
    for step in range(train_steps):
        start_time = time.time()

        # Generate candidate solutions
        candidate_params = es_optimizer.ask()
    
        # Evaluate loss on each candidate
        losses = [[] for _ in candidate_params] # Actually used for optimization
        mae_losses = [[] for _ in candidate_params] # For plotting only

        for _ in range(5):
            # Select dataset parameters and load dataset
            if step < 100:
                dset_idx = 0
            else:
                dset_idx = step % num_datasets
            dataset = datasets[dset_idx]
            dataset_config_rang = datasets_config_ranges[dset_idx]
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
                est_T = input.est_T
                rnn_M = net.get_M()
                try:
                    rnn_M = rnn_M.numpy()
                except:
                    pass
                est_M = np.linalg.pinv(np.eye(est_T.shape[0]) - net.gamma*est_T)
                if train_M:
                    diff = est_M - rnn_M
                else:
                    diff = est_T - rnn_T
                mse_error = np.mean(np.square(diff))
                mae_error = np.mean(np.abs(diff))
                losses[idx].append(mse_error)
                mae_losses[idx].append(mae_error)
        losses = [np.mean(loss) for loss in losses]
        mae_losses = [np.mean(loss) for loss in mae_losses]
    
        # Take optimization step based on losses
        es_optimizer.tell(candidate_params, losses)
        set_parameters(net, parameter_names, candidate_params[np.argmin(losses)])
    
        # Print statistics
        elapsed_time = time.time() - start_time
        time_step += elapsed_time
        time_net += elapsed_time
        running_loss += min(losses)
        running_mae_loss += min(mae_losses)
    
        if step % print_every_steps == 0:
            time_step /= print_every_steps
            if step > 0:
                running_loss /= print_every_steps
                running_mae_loss /= print_every_steps
    
            for name, param in chain(net.named_parameters(), net.named_buffers()):
                if torch.numel(param) > 1:
                    std, mean = torch.std_mean(param)
                    writer.add_scalar(name+'_std', std, step)
                else:
                    mean = torch.mean(param)
                writer.add_scalar(name, mean, step)
    
            writer.add_scalar('loss_train', running_loss, step)
            writer.add_scalar('mae_loss_train', running_mae_loss, step)
   
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
            torch.save(net.state_dict(), model_path)
            time_step = 0
            if (prev_running_loss < 1E-5) and (running_loss < 1E-5) and early_stop:
                end_training = True
            prev_running_loss = running_loss
            prev_running_mae_loss = running_mae_loss
            running_loss = 0.0
            running_mae_loss = 0.0
            grad_avg = 0
        if end_training:
            break
    
    if return_test_error:
        return_errors = []
        for _ in range(5):
            input = inputs.Sim2DWalk(num_steps=1000, num_states=8*8)
            dg_inputs = torch.from_numpy(input.dg_inputs.T).float().to(device)
            dg_inputs = dg_inputs.unsqueeze(1)
            dg_modes = torch.from_numpy(input.dg_modes.T).float().to(device)
            dg_modes = dg_modes.unsqueeze(1)
            net.ca3.set_num_states(input.num_states)
            with torch.no_grad():
                _, outputs = net(dg_inputs, dg_modes, reset=True)
            rnn_T = net.ca3.get_T().detach().numpy()
            est_T = input.est_T
            rnn_M = net.get_M()
            est_M = np.linalg.pinv(np.eye(est_T.shape[0]) - net.gamma*est_T)
            if train_M:
                return_errors.append(np.mean(np.abs(est_M - rnn_M)))
            else:
                return_errors.append(np.mean(np.abs(est_T - rnn_T)))
        return_error = np.mean(return_errors)
    else:
        return_error = prev_running_mae_loss
   
    error_type = 'M' if train_M else 'T'
    writer.add_scalar(f'return_error_{error_type}', return_error, step)
    writer.close()
    print('Finished Training\n', file=print_file)
    return net, return_error

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
    save_path = './trained_models/test6/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    datasets = [
        inputs.Sim1DWalk,
        inputs.Sim1DWalk,
        ]
    datasets_config_ranges = [
        {
        'num_steps': [3, 10, 20, 30],
        'left_right_stay_prob': [[1, 1, 1], [1, 1, 5], [7, 1, 0], [1, 7, 0]],
        'num_states': [5, 10, 15, 25]
        },
        {
        'num_steps': [100, 200],
        'num_states': [25, 36]
        },
        ]
    output_params = {
        'num_iterations': 20, 'input_clamp':np.inf,
        'nonlinearity': None
        }
    net_params = {
        'num_states':2, 'gamma':0.8,
        'ca3_kwargs':{
            'A_pos_sign':1, 'A_neg_sign':-1,
            'output_params':output_params
            }
        }

    with open(save_path + "net_configs.p", 'wb') as f:
        import pickle
        pickle.dump(net_params, f)

    net = STDP_SR(**net_params)
    net, return_error = train(
        save_path, net, datasets, datasets_config_ranges, train_steps=701,
        early_stop=True, print_every_steps=25, return_test_error=True,
        train_M=True
        )
    print(f'Final Error: {return_error}')
