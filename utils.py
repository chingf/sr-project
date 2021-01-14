import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def pol2cart(thetas, rhos):
    xs = rhos * np.cos(np.radians(thetas))
    ys = rhos * np.sin(np.radians(thetas))
    return xs, ys

def downsample(vector, factor):
    pad_size = ceil(float(vector.size)/factor)*factor - vector.size
    padded_vec = np.append(vector, np.zeros(pad_size)*np.nan)
    downsamp_vec = np.nanmean(padded_vec.reshape((-1, factor)), axis=1)
    return downsamp_vec   

def get_sr(T, gamma):
    D = np.diag(T @ np.ones(T.shape[0]))
    P = np.linalg.inv(D) @ T
    M = np.linalg.pinv(np.eye(P.shape[0]) - gamma*P)
    return M

def get_sr_features(T, sr_params):
    gamma = sr_params['gamma']
    recon_dim = sr_params['recon_dim']
    D = np.diag(T @ np.ones(T.shape[0]))
    P = np.linalg.inv(D) @ T
    M = np.linalg.pinv(np.eye(P.shape[0]) - sr_params['gamma']*P)
    U, S, V = np.linalg.svd(M)
    M_hat = np.dot(U[:,:recon_dim]*S[:recon_dim], V[:recon_dim,:])
    return M, U, M_hat

def debug_plot(state_vector, input, title):
    spatial_mat, context_vec = input.unravel_state_vector(state_vector)
    context_vec = context_vec.squeeze()
    vmax = np.max([spatial_mat.max(), context_vec.max()])
    context_mat = np.zeros(input.num_spatial_states)*np.nan
    wedge_states, cache_interactions = input.get_rel_vars()
    context_mat[wedge_states] = context_vec
    context_mat = context_mat.reshape(spatial_mat.shape)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    plt.suptitle(title)

    ax1.imshow(spatial_mat, vmin=0, vmax=vmax)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_title("Spatial Dim")

    im2 = ax2.imshow(context_mat, vmin=0, vmax=vmax)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_title("Cache Dim")
    for cache, interaction_amt in enumerate(cache_interactions):
        if interaction_amt == 0: continue
        anchor_point = np.unravel_index([wedge_states[cache]], context_mat.shape)
        anchor_point = [anchor_point[1] - 0.5, anchor_point[0] - 0.5]
        rect = patches.Rectangle(
            anchor_point, 1, 1, linewidth=1.5, edgecolor='red', facecolor='none'
            )
        ax2.add_patch(rect)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    plt.show()

