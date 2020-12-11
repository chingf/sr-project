import numpy as np
import inputs
from math import ceil, sqrt

def random_walk():
    input = inputs.RandomWalk(walls=40, num_steps=10000)
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(input.num_states))
        }
    plot_params = {
        'num_cells_to_plot': 25, 'history_size': input.num_steps, 'fps': 3,
        'save_filename': 'random_walk.mp4', 'select_cells': None,
        'num_frames': 300
        }
    return input, sr_params, plot_params

def rby_xywalk():
    input = inputs.RBYXYWalk(num_states=25*25, downsample_factor=None)
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(input.num_states))
        }
    plot_params = {
        'num_cells_to_plot': 25, 'history_size': input.num_steps, 'fps': 3,
        'save_filename': 'rby_xywalk.mp4',
        'select_cells': input.sorted_states[:25], 'num_frames': 300
        }
    return input, sr_params, plot_params

def lmn_xywalk():
    input = inputs.LMNXYWalk(
        num_states=25*25, downsample_factor=None, speed_thresh=8
        )
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(input.num_states))
        }
    plot_params = {
        'num_cells_to_plot': 25, 'history_size': input.num_steps, 'fps': 10,
        'save_filename': 'lmn_xywalk_fast.mp4',
        'select_cells': input.sorted_states[:25], 'num_frames': 300
        }
    return input, sr_params, plot_params

def lmn_xyhop():
    input = inputs.LMNXYHop(
        num_states=25*25, downsample_factor=None
        )
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(input.num_states))
        }
    plot_params = {
        'num_cells_to_plot': 25, 'history_size': input.num_steps, 'fps': 10,
        'save_filename': 'lmn_xyhop.mp4',
        'select_cells': input.sorted_states[:25], 'num_frames': 300
        }
    return input, sr_params, plot_params

def fakecache_walk():
    input = inputs.FakeCacheWalk(num_states=25*25, downsample_factor=None)
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(input.num_states))
        }
    plot_params = {
        'num_cells_to_plot': 25, 'history_size': input.num_steps, 'fps': 10,
        'save_filename': 'fakecache_walk.mp4',
        'select_cells': input.sorted_states[:25], 'num_frames"': 300
        }
    return input, sr_params, plot_params

