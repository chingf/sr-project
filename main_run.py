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
    "sim_walk":experiments.sim_walk,
    "sim_walk2":experiments.sim_walk2,
    "sim_walk3":experiments.sim_walk3,
    "sim_walk4":experiments.sim_walk4,
    "sim_walk5":experiments.sim_walk5
    }

parser = argparse.ArgumentParser() # Parse user-provided arguments
parser.add_argument('-e', '--experiment', help='Experiment Name', type=str)
parser.add_argument('-s', '--save', help='Save model filename', type=str)
args = parser.parse_args()
experiment_loader = experiments_dict[args.experiment]
model_filename = args.save
input, model, plotter, sr_params, plot_params = experiment_loader()

# Show STDP Kernel
plt.figure()
plt.plot(model.ca3.get_stdp_kernel(), linewidth=2)
plt.title("STDP Kernel")
plt.show()

# Buffer to store variables throughout simulation
plot_data = []

# Relevant performance tracking if model is estimating T
T_probabilities = [np.mean(np.sum(model.ca3.get_T(), axis=1))]
T_error = [0]

# Go through simulation step by step
for step in np.arange(input.num_steps):
    dg_input = input.dg_inputs[:, step]
    dg_mode = input.dg_modes[step]
    prev_dg_mode = input.dg_modes[step-1] if step != 0 else np.nan

    if dg_mode == 0: # Predictive mode
        dg_out, ca3_out = model.forward(dg_input)
    elif (dg_mode == 1) and (prev_dg_mode == 0): # Query Mode
        dg_out, ca3_out = model.query(dg_input)

    # DEBUGGING PLOTS
    if False: #step == input.num_steps-2: #step % 30 == 0:
        plt.figure(); plt.imshow(model.ca3.allX); plt.show()

        T = model.ca3.get_T()
        fig, ax = plt.subplots()
        im = ax.imshow(T)
        fig.colorbar(im, ax=ax); plt.title("Est T");
        for i in range(T.shape[0]):
            text = ax.text(i, i, "{:.2f}".format(T[i, i]),
                ha="center", va="center", color="k", fontsize=6)
            text = ax.text((i+1)%16, i, "{:.2f}".format(T[i, (i+1)%16]),
                ha="center", va="center", color="w", fontsize=5)
        plt.show()
        print(np.sum(T, axis=1))

        plt.figure(); plt.imshow(model.ca3.get_real_T()); plt.colorbar();
        plt.title("Real T"); plt.show()

        plt.figure(); plt.imshow(model.ca3.J); plt.colorbar();
        plt.title("J, no scaling");plt.show();

        plt.figure(); plt.imshow(model.ca3.get_M_hat()); plt.colorbar();
        plt.title("Est M");plt.show();

        plt.figure(); plt.imshow(model.ca3.last_update); plt.colorbar();
        plt.title("Update matrix");plt.show();

        plt.figure(); plt.imshow(model.ca3.allBpos); plt.colorbar();
        plt.title("Plasticity");plt.show();

        model.ca3.last_update = np.zeros(model.ca3.J.shape)

        import pdb; pdb.set_trace()

    # If this frame will be animated, save the relevant variables
    if step in plot_params['plot_frames']:
        M, U, M_hat = get_sr_features(model.ca3.get_T(), sr_params)
        history_start = max(0, step - plot_params['history_size'])
        plot_data.append([
            dg_out, input.dg_modes[step], M, U, M_hat,
            input.xs[:step], input.ys[:step], input.zs[:step]
            ])

    # If model is estimating T, track the performance
    if model.estimates_T:
        T_probabilities.append(np.mean(np.sum(model.ca3.get_T(), axis=1)))
        T_error.append(
            np.mean(np.abs(model.ca3.get_T() - model.ca3.get_real_T()))
            )

print("Saving model...")
pkl_objects = {
    'model': model, 'input': input, 'plotter':plotter,
    'sr_params': sr_params, 'plot_params': plot_params,
    'T_probabilities': T_probabilities, 'T_error': T_error
    }
with open('pickles/' + model_filename + ".p", 'wb') as f:
    pickle.dump(pkl_objects, f)
print("Saved.")

print("Animating SR development through experiment...")
plotter.set_data(plot_data)
plotter.set_save_filename(model_filename + ".mp4")
plotter.animate()

