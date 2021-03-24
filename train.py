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
num_steps = 100
num_test_steps = 20
min_start_step = 0
max_start_step = 75
num_states = 5
datasets = [
    inputs.Sim1DWalk,
    inputs.Sim1DWalk,
    inputs.Sim1DWalk
    ]
datasets_configs = [
    {'num_steps': num_steps, 'left_right_stay_prob': [1, 1, 1], 'num_states': num_states},
    {'num_steps': num_steps, 'left_right_stay_prob': [7, 1, 1], 'num_states': num_states},
    {'num_steps': num_steps, 'left_right_stay_prob': [1, 4, 1], 'num_states': num_states}
    ]
#datasets = [
#    inputs.Sim1DWalk,
#    inputs.Sim1DWalk,
#    inputs.Sim1DWalk,
#    inputs.Sim1DWalk,
#    inputs.Sim2DLevyFlight
#    ]
#datasets_configs = [
#    {'num_steps': num_steps, 'left_right_stay_prob': [1, 1, 1], 'num_states': num_states},
#    {'num_steps': num_steps, 'left_right_stay_prob': [5, 1, 1], 'num_states': num_states},
#    {'num_steps': num_steps, 'left_right_stay_prob': [1, 5, 1], 'num_states': num_states},
#    {'num_steps': num_steps, 'left_right_stay_prob': [9, 1, 1], 'num_states': num_states},
#    {'num_steps': num_steps, 'walls': int(np.sqrt(num_states) - 1)}
#    ]
num_datasets = len(datasets)

# Init net
writer = SummaryWriter('./trained_models')
net = STDP_SR(num_states=num_states, gamma=0.5)
criterion = nn.MSELoss(reduction='none')
lr=1E-3
optimizer = torch.optim.Adam(net.parameters(), lr=lr) #Adam

# Loss reporting
running_loss = 0.0
running_loss_reg = 0.0
print_every_steps = 50
train_steps = 10000
time_task = 0
time_net = 0
save_path = './trained_models'

grad_avg = 0

for step in range(train_steps):
    dataset = datasets[step % num_datasets]
    dataset_config = datasets_configs[step % num_datasets]
    start_time = time.time()
    input = dataset(**dataset_config)
    dg_inputs = torch.from_numpy(input.dg_inputs.T).float().to(device).unsqueeze(1)
    dg_modes = torch.from_numpy(input.dg_modes.T).float().to(device).unsqueeze(1)

    time_task += time.time() - start_time
    start_time = time.time()

    # Start randomly in the middle
    start_step = np.random.choice(np.arange(min_start_step, max_start_step))
    if start_step > num_test_steps:
        _ = net(dg_inputs[:start_step,:,:], dg_modes[:start_step,:])
        net.ca3.set_J_to_real_T()
    else:
        start_step = 0

    # zero the parameter gradients
    optimizer.zero_grad()
    _, outputs = net(
        dg_inputs[start_step:start_step+num_test_steps, :, :],
        dg_modes[start_step:start_step+num_test_steps, :],
        reset=start_step==0
        )

    loss = criterion(
        net.ca3.get_T(),
        torch.tensor(net.ca3.get_real_T()).float()
        )
    loss = torch.sum(torch.sum(loss, dim=1))
    loss.backward()
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
        print(f'{net.ca3.tau_pos.data.item()}')
        print(f'{net.ca3._ceil.data.item()}')
        model_path = os.path.join(save_path, 'model.pt')
        torch.save(net.state_dict(), model_path)
        running_loss = 0.0
        grad_avg = 0

writer.close()
print('Finished Training\n')

