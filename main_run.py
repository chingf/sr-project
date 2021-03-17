import numpy as np
import time
import sys
import os
import argparse
import pickle
import experiments
import plotting
from utils import get_sr, get_sr_features, debug_plot

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser() # Parse user-provided arguments
parser.add_argument('-e', '--experiment', help='Experiment Name', type=str)
parser.add_argument('-s', '--save', help='Save model filename', type=str)
args = parser.parse_args()
experiment_loader = getattr(experiments, args.experiment)
model_filename = args.save
input, model, plotter, sr_params, plot_params = experiment_loader()

# Buffer to store variables throughout simulation
plot_data = []

# Relevant performance tracking if model is estimating T
debug = False
T_probabilities = [np.mean(np.sum(model.ca3.get_T(), axis=1))]
T_error = [0]
test_gammas = [0.1, 0.25, 0.5, 0.99]
M_error = [[0] for test_gamma in test_gammas]
M_mean = [[0.001] for test_gamma in test_gammas]

# Test PCA
xs = []
#sys.stdout = open(os.devnull, 'w')

# Go through simulation step by step
print("Beginning walk")
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
    if model.estimates_T and debug:
        T_hat = model.ca3.get_T()
        real_T = model.ca3.get_real_T()
        T_probabilities.append(np.mean(np.sum(model.ca3.get_T(), axis=1)))
        T_error.append(np.mean(np.abs(T_hat - real_T)))
        for gamma_idx, test_gamma in enumerate(test_gammas):
            M_hat = get_sr(T_hat, test_gamma)
            M = get_sr(real_T, test_gamma)
            M_error[gamma_idx].append(np.mean(np.abs(M_hat - M)))
            M_mean[gamma_idx].append(np.mean(M))

    # Test PCA
    #xs.append(ca3_out)

# Test PCA
#xs = np.array(xs) # (timesteps, states)
#plt.figure()
#plt.imshow(xs.T)
#plt.show()
#from statsmodels.tsa.api import ExponentialSmoothing
#from sklearn.decomposition import PCA
#pca = PCA(1)
#M, U, M_hat = get_sr_features(model.ca3.get_T(), sr_params)
#plt.figure();
#plt.imshow(U);
#plt.show()
#plt.figure()
#for alpha in [0.001, 0.005, 0.01, 0.1, 1]:
#    smoothed_xs = []
#    for state in range(xs.shape[1]):
#        exp = ExponentialSmoothing(xs[:,state])
#        exp_model = exp.fit(smoothing_level=alpha)
#        result = exp_model.fittedvalues
#        smoothed_xs.append(result)
#    smoothed_xs = np.array(smoothed_xs) # (states, timesteps)
#    pca.fit(smoothed_xs.T)
#    y = pca.components_.squeeze()
#    plt.plot(y + pca.mean_, label=str(alpha))
#plt.legend()
#plt.show()

print("Saving model...")
pkl_objects = { 
    'model': model, 'input': input, 'plotter':plotter,
    'sr_params': sr_params, 'plot_params': plot_params,
    'T_probabilities': T_probabilities, 'T_error': T_error,
    'test_gammas': test_gammas, 'M_error': M_error, 'M_mean': M_mean
    }
with open('pickles/' + model_filename + ".p", 'wb') as f:
    pickle.dump(pkl_objects, f)
print("Saved.")

print("Animating SR development through experiment...")
plotter.set_data(plot_data)
plotter.set_save_filename(model_filename + ".mp4")
plotter.animate()

