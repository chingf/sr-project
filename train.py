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

# Dataset Configs
datasets = [inputs.Sim1DWalk]
datasets_config_ranges = [{
    'num_steps': [3, 5, 15, 30],
    'left_right_stay_prob': [[1, 1, 1], [7, 1, 1], [1, 4, 1]],
    'num_states': [5, 10, 15]
    }]
num_datasets = len(datasets)
p = {
    'num_steps': [0.8, 0.2, 0, 0],
    }

# Init net
writer = SummaryWriter('./trained_models')
net = STDP_SR(num_states=2, gamma=0.5)
criterion = nn.MSELoss(reduction='none')
lr=1E-3
weight_decay = 1E-3
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay) #Adam

# Loss reporting
running_loss = 0.0
running_loss_reg = 0.0
print_every_steps = 50
train_steps = 50000
time_task = 0
time_net = 0
save_path = './trained_models'

grad_avg = 0

for step in range(train_steps):
    dataset = datasets[step % num_datasets]
    dataset_config_rang = datasets_config_ranges[step % num_datasets]
    dataset_config = {}
    for key in dataset_config_rang:
        num_samples = len(dataset_config_rang[key])
        p_key = p[key] if key in p else None
        sample_idx = np.random.choice(num_samples, p=p_key)
        dataset_config[key] = dataset_config_rang[key][sample_idx]
    start_time = time.time()
    input = dataset(**dataset_config)
    dg_inputs = torch.from_numpy(input.dg_inputs.T).float().to(device).unsqueeze(1)
    dg_modes = torch.from_numpy(input.dg_modes.T).float().to(device).unsqueeze(1)
    net.ca3.set_num_states(input.num_states)

    time_task += time.time() - start_time
    start_time = time.time()

    # zero the parameter gradients
    optimizer.zero_grad()
    _, outputs = net(dg_inputs, dg_modes, reset=True)

    loss = criterion(
        net.ca3.get_T(),
        torch.tensor(net.ca3.get_real_T()).float()
        )
    loss = torch.sum(torch.sum(loss, dim=1))
    loss.backward(retain_graph=True)
    nn.utils.clip_grad_norm_(net.parameters(), 1)

    for param in net.parameters():
        if torch.isnan(param):
            import pdb; pdb.set_trace()

    grad_avg += net.ca3.tau_pos.grad.item()
    optimizer.step()

    time_net += time.time() - start_time

    # print statistics
    running_loss += loss.item()

    if step % print_every_steps == 0:
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

        print('', flush=True)
        print('[{:5d}] loss: {:0.3f}'.format(step + 1, running_loss))
        print('Time per step {:0.3f}ms'.format(1e3*(time_task+time_net)/(step+1)))
        print('Total time on task {:0.3f}s, net {:0.3f}s'.format(time_task,
                                                                 time_net))
        print(f'A_pos: {net.ca3.A_pos.data.item()}')
        print(f'tau_pos: {net.ca3.tau_pos.data.item()}')
        print(f'A_neg: {net.ca3.A_neg.data.item()}')
        print(f'tau_neg: {net.ca3.tau_neg.data.item()}')
        print(f'alpha_self: {net.ca3.alpha_self}')
        print(f'alpha_other: {net.ca3.alpha_other.data.item()}')
        print(f'Update clamp: {net.ca3.update_clamp.x0.data.item()}')
        print(f'Update activity clamp: {net.ca3.update_activity_clamp.x0.data.item()}')
        model_path = os.path.join(save_path, 'model.pt')
        torch.save(net.state_dict(), model_path)
        running_loss = 0.0
        grad_avg = 0

    if step == 50*print_every_steps:
        for key in p:
            p[key] = [0.3, 0.6, 0.1, 0]
    elif step == 80*print_every_steps:
        for key in p:
            p[key] = [0, 0.5, 0.5, 0]
    elif step == 110*print_every_steps:
        for key in p:
            p[key] = [0, 0.4, 0.4, 0.2]
    elif step == 140*print_every_steps:
        for key in p:
            p[key] = None

writer.close()
print('Finished Training\n')

