import numpy as np
import time
import experiments
import plotting
from utils import get_sr_features

input, sr_params, plot_params = experiments.fakecache_walk()

A = 0.001*np.random.rand(input.num_states, input.num_states)
plot_data = []
plot_Ms = []
plot_Us = []
plot_Mhats = []
plot_xs = []
plot_ys = []
for step in np.arange(1, input.num_steps):
    A[input.states[step-1], input.states[step]] += 1
    if step in np.round(np.linspace(2, input.num_steps, 300)): #np.round(np.logspace(2, np.log10(input.num_steps), 100)):
        M, U, M_hat = get_sr_features(A, sr_params)
        history_start = max(0, step - plot_params['history_size'])
        plot_data.append(
            [M, U, M_hat, input.xs[history_start:step], input.ys[history_start:step]]
            )

import matplotlib.pyplot as plt
print("animating")
plotter = plotting.BasicPlot(
    input, plot_data, plot_params['num_cells_to_plot'],
    plot_params['fps'], plot_params['select_cells'], plot_params['save_filename']
    )
plotter.animate()

