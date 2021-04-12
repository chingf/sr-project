import numpy as np
import inputs
import models
import plotting
from math import ceil, sqrt


## Artificial Simulations

def sim_1dwalk():
    input = inputs.Sim1DWalk(
        num_steps=4000, left_right_stay_prob=[8, 1, 1], num_states=64
        )
    sr_params = {
        'gamma': 0.5, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 16, 'history_size': input.num_steps, 'fps': 30,
        'save_filename': 'simwalk.mp4',
        'select_cells': np.arange(16),
        'plot_frames': np.arange(1)
        }
    model = models.STDP_SR(
        input.num_states, sr_params['gamma'], ca3_kwargs={'debug_print':False}
        )
    plotter = None
    return input, model, plotter, sr_params, plot_params

def sim_2dwalk():
    input = inputs.Sim2DWalk(num_steps=4000, num_states=64)
    sr_params = {
        'gamma': 0.5, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 16, 'history_size': input.num_steps, 'fps': 30,
        'save_filename': 'simwalk.mp4',
        'select_cells': input.sorted_states[:16],
        'plot_frames': np.arange(1)
        }
    model = models.STDP_SR(input.num_states, sr_params['gamma'])
    plotter = plotting.SpatialPlot(
        input, plot_params['num_cells_to_plot'],
        plot_params['fps'], plot_params['select_cells'], plot_params['save_filename']
        )
    plotter = None
    return input, model, plotter, sr_params, plot_params

def sim_2dlevy():
    input = inputs.Sim2DLevyFlight(num_steps=4000, walls=7)
    sr_params = {
       'gamma': 0.5, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 16, 'history_size': input.num_steps, 'fps': 30,
        'save_filename': 'simwalk.mp4',
        'select_cells': input.sorted_states[:16],
        'plot_frames': np.linspace(0, input.num_steps, 30*8).astype(int)
        }
    model = models.STDP_SR(input.num_states, sr_params['gamma'])
    plotter = plotting.SpatialPlot(
        input, plot_params['num_cells_to_plot'],
        plot_params['fps'], plot_params['select_cells'], plot_params['save_filename']
        )
    plotter = None
    return input, model, plotter, sr_params, plot_params


## Experiment Simulations

def rby_xywalk():
    input = inputs.RBYXYWalk(num_states=25*25, downsample_factor=10)
    sr_params = {
        'gamma': 0.5, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 25, 'history_size': input.num_steps, 'fps': 30,
        'save_filename': 'rby_xywalk.mp4',
        'select_cells': input.sorted_states[:25],
        'plot_frames': np.round(np.linspace(0, input.num_steps, 30*8))
        }
    model = models.STDP_SR(input.num_states, sr_params['gamma'])
    plotter = plotting.SpatialPlot(
        input, plot_params['num_cells_to_plot'],
        plot_params['fps'], plot_params['select_cells'], plot_params['save_filename']
        )
    return input, model, plotter, sr_params, plot_params

def rby_cachewalk_analytic():
    input = inputs.RBYCacheWalk(
        num_spatial_states=25*25, downsample_factor=None, skip_frame=0.7
        )
    print(f'input of size {input.xs.size}')
    sr_params = {
        'gamma': 0.5, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 25, 'history_size': input.num_steps, 'fps': 30,
        'save_filename': 'rby_xywalk.mp4',
        'select_cells': input.sorted_states[:25],
        'plot_frames': np.round(np.linspace(0, input.num_steps, 30*8))
        }
    model = models.AnalyticSR(
        input.num_states, sr_params['gamma'], ca3_kwargs={'gamma_T': 0.99}
        )
    plotter = plotting.SpatialPlot(
        input, plot_params['num_cells_to_plot'],
        plot_params['fps'], plot_params['select_cells'], plot_params['save_filename']
        )
    return input, model, plotter, sr_params, plot_params

def rby_cachewalk_learned():
    input = inputs.RBYCacheWalk(
        num_spatial_states=25*25, downsample_factor=None, skip_frame=0.7
        )
    sr_params = {
        'gamma': 0.5, 'recon_dim': ceil(sqrt(sqrt(input.num_states)))
        }
    plot_params = {
        'num_cells_to_plot': 25, 'history_size': input.num_steps, 'fps': 30,
        'save_filename': 'rby_xycache.mp4',
        'select_cells': input.sorted_states[:25],
        'plot_frames': np.round(np.linspace(0, input.num_steps, 30*8))
        }
    model = models.STDP_SR(
        input.num_states, sr_params['gamma'], ca3_kwargs={'debug_print':False}
        )
    plotter = plotting.SpatialPlot(
        input, plot_params['num_cells_to_plot'],
        plot_params['fps'], plot_params['select_cells'], plot_params['save_filename']
        )
    return input, model, plotter, sr_params, plot_params

