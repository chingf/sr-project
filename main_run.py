import numpy as np
import time
import argparse
import pickle
import experiments
import plotting
from utils import get_sr_features, debug_plot

experiments_dict = { # Possible experiments to run
    "rbyxywalk": experiments.rby_xywalk,
    "rbycachewalk": experiments.rby_cachewalk
    }

parser = argparse.ArgumentParser() # Parse user-provided arguments
parser.add_argument('-e', '--experiment', help='Experiment Name', type=str)
parser.add_argument('-s', '--save', help='Save model filename', type=str)
args = parser.parse_args()
experiment_loader = experiments_dict[args.experiment]
model_filename = args.save
input, model, plotter, sr_params, plot_params = experiment_loader()

plot_data = []
plot_Ms = []
plot_Us = []
plot_Mhats = []
plot_xs = []
plot_ys = []
for step in np.arange(input.num_steps):
    dg_input = inputs.dg_inputs[:, step]
    dg_mode = input.dg_modes[step]
    prev_dg_mode = input.dg_modes[step-1] if step != 0 else np.nan

    if dg_mode == 0: # Predictive mode
        dg_out, ca3_out = model.forward(dg_input)
    elif (dg_mode == 1) and (prev_dg_mode == 0): # Query Mode
        dg_out, ca3_out = model.query(dg_input)

    if step in plot_params['plot_frames']: # If frame is included in animation
        M, U, M_hat = get_sr_features(model.ca3.T, sr_params)
        history_start = max(0, step - plot_params['history_size'])
        for _ in range(plot_count):
            plot_data.append([
                dg_out, input.dg_modes[step], M, U, M_hat,
                input.xs[:step], input.ys[:step], input.zs[:step]
                ])

print("Saving model...")
pkl_objects = {
    'model': model, 'input': input, 'plotter':plotter,
    'sr_params': sr_params, 'plot_params': plot_params
    }
with open('pickles/' + model_filename, 'wb') as f:
    pickle.dump(pkl_objects, f)

print("Animating SR development through experiment...")
plotter.set_data(plot_data)
plotter.animate()

