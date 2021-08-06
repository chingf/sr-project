import numpy as np
import argparse
import time
import os
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR

device = 'cpu'

def train(
    save_path, net, datasets, datasets_config_ranges,
    p=None, print_file=None
    ):

    # Initializations
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_datasets = len(datasets)
    writer = SummaryWriter(save_path)
    criterion = nn.MSELoss(reduction='none')
    lr=1E-3
    weight_decay = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay) #Adam
    
    # Loss reporting
    running_loss = 0.0
    print_every_steps = 50
    train_steps = 50000
    time_step = 0
    time_net = 0
    grad_avg = 0
    
    for step in range(train_steps):
        start_time = time.time()
    
        # Select dataset parameters and load dataset
        dataset = datasets[step % num_datasets]
        dataset_config_rang = datasets_config_ranges[step % num_datasets]
        dataset_config = {}
        for key in dataset_config_rang:
            num_samples = len(dataset_config_rang[key])
            p_key = p[key] if key in p else None
            sample_idx = np.random.choice(num_samples, p=p_key)
            dataset_config[key] = dataset_config_rang[key][sample_idx]
        input = dataset(**dataset_config)
        dg_inputs = torch.from_numpy(input.dg_inputs.T).float().to(device).unsqueeze(1)
        dg_modes = torch.from_numpy(input.dg_modes.T).float().to(device).unsqueeze(1)
        net.ca3.set_num_states(input.num_states)
    
        # Zero grad and run network
        optimizer.zero_grad()
        _, outputs = net(dg_inputs, dg_modes, reset=True)
    
        # Backprop the loss
        loss = criterion(
            net.ca3.get_T(),
            torch.tensor(net.ca3.get_real_T()).float()
            )
        loss = torch.sum(torch.sum(loss, dim=1))
        loss.backward()
        grad_avg += net.ca3.tau_pos.grad.item()
        optimizer.step()
    
        # Print statistics
        elapsed_time = time.time() - start_time
        time_step += elapsed_time
        time_net += elapsed_time
        running_loss += loss.item()
    
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
            print(f'A_pos: {net.ca3.A_pos.data.item()}', file=print_file )
            print(f'tau_pos: {net.ca3.tau_pos.data.item()}', file=print_file )
            print(f'A_neg: {net.ca3.A_neg.data.item()}', file=print_file )
            print(f'tau_neg: {net.ca3.tau_neg.data.item()}', file=print_file )
            print(f'alpha_self: {net.ca3.alpha_self.data.item()}', file=print_file )
            print(f'alpha_other: {net.ca3.alpha_other.data.item()}', file=print_file )
            print(f'Update clamp: {net.ca3.update_clamp.x0.data.item()}', file=print_file)
            print(
                f'Update activity clamp: {net.ca3.update_activity_clamp.x0.data.item()}',
                file=print_file
                )

            model_path = os.path.join(save_path, 'model.pt')
            torch.save(net.state_dict(), model_path)
            time_step = 0
            running_loss = 0.0
            grad_avg = 0

    writer.close()
    print('Finished Training\n', file=print_file)
    return net, running_loss

if __name__ == "__main__":
    save_path = './trained_models'
    datasets = [inputs.Sim1DWalk]
    datasets_config_ranges = [{
        'num_steps': [3, 10, 15],
        'left_right_stay_prob': [[1, 1, 1], [7, 1, 1], [1, 4, 1]],
        'num_states': [5, 10, 15]
        }]
    net = STDP_SR(num_states=2, gamma=0.4)
    p = {
        'num_steps': [0.7, 0.2, 0.1],
        }
    train(save_path, net, datasets, datasets_config_ranges, p=p)

