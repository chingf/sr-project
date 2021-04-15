import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
import pickle

import experiments
import plotting
from utils import get_sr_features, debug_plot

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', help='Experiment Name', type=str)
parser.add_argument('-s', '--save', help='Model filename to load', type=str)
args = parser.parse_args()
model_filename = args.save
experiment_loader = getattr(experiments, args.experiment)
input, model, plotter, sr_params, plot_params = experiment_loader()

with open("pickles/" + model_filename + ".p", "rb") as f:
    model = pickle.load(f)['model']
plt.show()
M, U, M_hat = get_sr_features(model.ca3.get_real_T(), sr_params)

wedge_states, cache_interactions = input.get_rel_vars()
dg_allcache_query = np.zeros(input.num_states)
dg_allcache_query[-16:] = 1
debug_plot(np.zeros(input.num_states), input, "$0$")
debug_plot(dg_allcache_query, input, "$i_c$")
for cache in [0, 1, 2]:
    dg_cache_query = np.zeros(input.num_states)
    dg_cache_query[-(16-cache)] = 1
    Mi_cache = M.T @ dg_cache_query
for wedge, wedge_state in enumerate(wedge_states):
    if wedge not in [7, 13]: continue
    dg_spatial_query = np.zeros(input.num_states)
    dg_spatial_query[wedge_state] = 1
    debug_plot(dg_spatial_query, input, "$i_s$")
    Mi_sp = M.T@dg_spatial_query
    debug_plot(Mi_sp, input, "Likely states from Wedge %s: $M^Ti_{spatial}$"%wedge)
    #Mi_c = M@dg_allcache_query
    #debug_plot(Mi_c, input, "Likely states to all wedges: $Mi_{caches}$")
    #dg_input = M.T@dg_spatial_query * M@dg_allcache_query
    #debug_plot(dg_input, input, "Query Result for Wedge %s:\n$M^Ti_s \odot Mi_c$"%wedge)
    dg_input = M.T@dg_spatial_query * dg_allcache_query
    debug_plot(dg_input, input, "Query Result for Wedge %s:\n$M^Ti_s \odot i_c$"%wedge)


