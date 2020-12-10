import numpy as np
from math import ceil

def pol2cart(thetas, rhos):
    xs = rhos * np.cos(np.radians(thetas))
    ys = rhos * np.sin(np.radians(thetas))
    return xs, ys

def downsample(vector, factor):
    pad_size = ceil(float(vector.size)/factor)*factor - vector.size
    padded_vec = np.append(vector, np.zeros(pad_size)*np.nan)
    downsamp_vec = np.nanmean(padded_vec.reshape((-1, factor)), axis=1)
    return downsamp_vec   

def get_sr_features(A, sr_params):
    gamma = sr_params['gamma']
    recon_dim = sr_params['recon_dim']
    D = np.diag(A @ np.ones(A.shape[0]))
    P = np.linalg.inv(D) @ A
    M = np.linalg.pinv(np.eye(P.shape[0]) - sr_params['gamma']*P)
    U, S, V = np.linalg.svd(M)
    M_hat = np.dot(U[:,:recon_dim]*S[:recon_dim], V[:recon_dim,:])
    return M, U, M_hat

