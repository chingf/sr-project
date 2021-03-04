import numpy as np
import inputs
import models
import plotting
from math import ceil, sqrt


def sim_walk():
    input = inputs.SimWalk(num_steps=250, stay_to_hop_ratio=6)
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 16, 'history_size': input.num_steps, 'fps': 10,
        'save_filename': 'simwalk.mp4',
        'select_cells': np.arange(16), 'num_frames': 300,
        'plot_frames': np.array([1e4])
        }
    model = models.STDP_LR_SR(sr_params['gamma'], input.num_states)
    plotter = None
    return input, model, plotter, sr_params, plot_params

def sim_walk2():
    input = inputs.SimWalk2(num_steps=1000, left_right_bias=0.5)
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 16, 'history_size': input.num_steps, 'fps': 10,
        'save_filename': 'simwalk.mp4',
        'select_cells': np.arange(16), 'num_frames': 300,
        'plot_frames': np.array([1e4])
        }
    model = models.STDP_LR_SR(sr_params['gamma'], input.num_states)
    plotter = None
    return input, model, plotter, sr_params, plot_params

def sim_walk3():
    input = inputs.SimWalk3(num_steps=1000, num_states=100)
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 16, 'history_size': input.num_steps, 'fps': 30,
        'save_filename': 'simwalk.mp4',
        'select_cells': input.sorted_states[:16], 'num_frames': 249,
        'plot_frames': np.arange(999)
        }
    model = models.STDP_LR_SR(sr_params['gamma'], input.num_states)
    plotter = plotting.SpatialPlot(
        input, plot_params['num_cells_to_plot'],
        plot_params['fps'], plot_params['select_cells'], plot_params['save_filename']
        )
    return input, model, plotter, sr_params, plot_params

def sim_walk4():
    input = inputs.SimWalk4(num_steps=1000, walls=25)
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 16, 'history_size': input.num_steps, 'fps': 30,
        'save_filename': 'simwalk.mp4',
        'select_cells': input.sorted_states[:16], 'num_frames': 249,
        'plot_frames': np.arange(999)
        }
    model = models.STDP_LR_SR(sr_params['gamma'], input.num_states)
    plotter = plotting.SpatialPlot(
        input, plot_params['num_cells_to_plot'],
        plot_params['fps'], plot_params['select_cells'], plot_params['save_filename']
        )
    return input, model, plotter, sr_params, plot_params

def sim_walk5():
    input = inputs.SimWalk5(num_steps=1000)
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 16, 'history_size': input.num_steps, 'fps': 30,
        'save_filename': 'simwalk.mp4',
        'select_cells': np.arange(16), 'num_frames': 249,
        'plot_frames': np.arange(999)
        }
    model = models.STDP_LR_SR(sr_params['gamma'], input.num_states)
    plotter = None
    return input, model, plotter, sr_params, plot_params


def rby_xywalk():
    input = inputs.RBYXYWalk(num_states=25*25, downsample_factor=None)
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 25, 'history_size': input.num_steps, 'fps': 10,
        'save_filename': 'rby_xywalk.mp4',
        'select_cells': input.sorted_states[:25], 'num_frames': 300,
        'plot_frames': np.round(np.linspace(2, input.num_steps, 300))
        }
    model = models.AnalyticSR(sr_params['gamma'], input.num_states)
    plotter = plotting.SpatialPlot(
        input, plot_params['num_cells_to_plot'],
        plot_params['fps'], plot_params['select_cells'], plot_params['save_filename']
        )
    return input, model, plotter, sr_params, plot_params


def rby_cachewalk():
    input = inputs.RBYCacheWalk(num_spatial_states=25*25, downsample_factor=None)
    sr_params = {
        'gamma': 0.9, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 25, 'history_size': input.num_steps, 'fps': 10,
        'save_filename': 'rby_xycache.mp4',
        'select_cells': input.sorted_states[:25], 'num_frames': 300,
        'plot_frames': np.round(np.linspace(2, input.num_steps, 300))
        }
    model = models.AnalyticSR(sr_params['gamma'], input.num_states)
    plotter = plotting.SpatialPlot(
        input, plot_params['num_cells_to_plot'],
        plot_params['fps'], plot_params['select_cells'], plot_params['save_filename']
        )
    return input, model, plotter, sr_params, plot_params
