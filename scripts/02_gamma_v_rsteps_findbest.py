import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pickle
import numpy as np
import torch.nn as nn
import torch
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from datasets import inputs
from sr_model.models.models import AnalyticSR, STDP_SR
from train import train

def find_best(gamma, rstep, nonlinearity):
    experiment_dir = '../../engram/Ching/02_gamma_v_rsteps/'
    nonlinearity_dir = f'{experiment_dir}{gamma}/{rstep}/{nonlinearity}/'
    best_iter_val = np.inf
    best_iter = None
    for _iter in os.listdir(nonlinearity_dir):
        iter_dir = f'{nonlinearity_dir}{_iter}/'
        if not os.path.isfile(iter_dir + 'net_configs.p'):
            continue
        for file in os.listdir(iter_dir):
            if 'tfevents' not in file: continue
            tfevents_file = iter_dir + file
            event_acc = EventAccumulator(tfevents_file)
            event_acc.Reload()
            iter_val = [
                event_acc.Scalars('loss_train')[-i].value for i in range(1)
                ]
            iter_val = np.mean(iter_val)
            print(f'Iteration {_iter}: {iter_val}')
            if iter_val < best_iter_val:
                best_iter_val = iter_val
                best_iter = _iter
            break
    print(f'Best iteration: {best_iter}, value={best_iter_val}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gamma")
    parser.add_argument("rstep")
    parser.add_argument("nonlinearity")
    args = parser.parse_args()
    find_best(args.gamma, args.rstep, args.nonlinearity)
