import numpy as np
import time
import argparse
import pickle
import experiments
import plotting
from utils import get_sr_features, debug_plot

experiments_dict = {
    "rbyxywalk": experiments.rby_xywalk,
    "rbycachewalk": experiments.rby_cachewalk
    }

parser = argparse.ArgumentParser()
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
for step in np.arange(1, input.num_steps):
    if input.dg_modes[step] == 0:
        dg_out, ca3_out = model.forward(input.dg_inputs[:,:step])
        #model.update(input.dg_inputs[:,step], input.dg_inputs[:,:step])
    elif (input.dg_modes[step] == 1) and (input.dg_modes[step-1] == 0):
        dg_out, ca3_out = model.query(
            input.dg_inputs[:,step], input.dg_inputs[:,step-1] #TODO: consider
            )
    plot_count = np.sum(plot_params['plot_frames'] == step)
    if plot_count > 0:
        M, U, M_hat = get_sr_features(model.ca3.T, sr_params)
        history_start = max(0, step - plot_params['history_size'])
        for _ in range(plot_count):
            plot_data.append([
                dg_out, input.dg_modes[step], M, U, M_hat,
                input.xs[:step], input.ys[:step], input.zs[:step]
                ])
with open('pickles/' + model_filename, 'wb') as f:
    pickle.dump({'model': model}, f)
print("animating")
plotter.set_data(plot_data)
plotter.animate()

