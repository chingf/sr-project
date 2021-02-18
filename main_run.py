import numpy as np
import time
import argparse
import pickle
import experiments
import plotting
from utils import get_sr_features, debug_plot

import matplotlib.pyplot as plt

experiments_dict = { # Possible experiments to run
    "rbyxywalk": experiments.rby_xywalk,
    "rbycachewalk": experiments.rby_cachewalk,
    "sim_walk":experiments.sim_walk
    }

parser = argparse.ArgumentParser() # Parse user-provided arguments
parser.add_argument('-e', '--experiment', help='Experiment Name', type=str)
parser.add_argument('-s', '--save', help='Save model filename', type=str)
args = parser.parse_args()
experiment_loader = experiments_dict[args.experiment]
model_filename = args.save
input, model, plotter, sr_params, plot_params = experiment_loader()

# DEBUGGING PLOT
plt.figure()
plt.plot(model.ca3.get_stdp_kernel(), linewidth=2)
plt.title("STDP Kernel")
plt.show()

plot_data = []
plot_Ms = []
plot_Us = []
plot_Mhats = []
plot_xs = []
plot_ys = []
for step in np.arange(input.num_steps):
    dg_input = input.dg_inputs[:, step]
    dg_mode = input.dg_modes[step]
    prev_dg_mode = input.dg_modes[step-1] if step != 0 else np.nan

    if dg_mode == 0: # Predictive mode
        dg_out, ca3_out = model.forward(dg_input)
    elif (dg_mode == 1) and (prev_dg_mode == 0): # Query Mode
        dg_out, ca3_out = model.query(dg_input)

    # DEBUGGING PLOTS
    if step % int(input.stay_to_hop_ratio*16) == 0:
        plt.figure(); plt.imshow(model.ca3.allX); plt.show()
        plt.figure(); plt.imshow(model.ca3.get_T()); plt.colorbar(); plt.title("Est T"); plt.show()
        plt.figure(); plt.imshow(model.ca3.get_real_T()); plt.colorbar();plt.title("Real T"); plt.show()
        plt.figure(); plt.imshow(model.ca3.J); plt.colorbar();plt.title("J, no scaling");plt.show();
        plt.figure(); plt.imshow(model.ca3.get_M_hat()); plt.colorbar();plt.title("Est M");plt.show();
        plt.figure(); plt.imshow(model.ca3.last_update); plt.colorbar();plt.title("Update matrix");plt.show();
        model.ca3.last_update = np.zeros(model.ca3.J.shape)
        import pdb; pdb.set_trace()

    if step in plot_params['plot_frames']: # If frame is included in animation
        M, U, M_hat = get_sr_features(model.ca3.get_T(), sr_params)
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

