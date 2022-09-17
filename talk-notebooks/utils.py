import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import re
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from astropy.convolution import convolve
from scipy.signal.windows import hamming
from scipy.stats import gamma
from scipy.special import digamma, loggamma

root = os.path.dirname(os.path.abspath(os.curdir))
sys.path.append(root)

from sr_model.models.models import AnalyticSR, STDP_SR, Linear
from datasets import inputs, sf_inputs_discrete
from scipy.ndimage.measurements import label
import configs

device = 'cpu'

def get_firing_field(
    xs, ys, activity, arena_length, window_ratio=0.12, normalize=False
    ):
    """
    The window size (area*0.12) is a ratio taken from Payne & Aronov 2021.

    Args:
        xs: (frames,) array of x locations
        ys: (frames,) array of y locations
        activity: (frames,) array of activity of one neuron
        window_ratio: Hamming window size; default ratio from Payne 2021
        arena_length: length of square arena
    """

    # Collect mean firing rates
    firing_field = np.zeros((arena_length, arena_length))*np.nan
    for x in range(arena_length):
        for y in range(arena_length):
            frame_idxs = np.logical_and(xs==x, ys==y)
            fr = np.nanmean(activity[frame_idxs])
            firing_field[x,y] = fr
    nan_idxs = np.isnan(firing_field)
    
    # Smooth field
    window = int(np.sqrt(arena_length*arena_length*window_ratio))
    window = window + 1 if window%2 == 0 else window # Must be odd number
    kernel = np.outer(hamming(window), hamming(window))
    firing_field = convolve(firing_field, kernel)

    # Normalize, if desired
    if normalize:
        max_rate = firing_field.max()
        if max_rate != 0:
            firing_field = firing_field/firing_field.max()
            
    return firing_field, nan_idxs

def get_field_metrics(
        activity, dset, arena_length,
        nshuffles=50, save_field_info=False, reload_field_info=False,
        save_path=None
        ):
    """
    Args:
        activity: (frames, neurs) array of activity of all neurons
        arena_length: length of square arena
    """

    xs = dset.xs.astype(int)
    ys = dset.ys.astype(int)
    
    fieldsizes = []
    nfields = []

    # If you want to reload fields and there is a field info file available
    field_info_path = save_path + 'field_infos.p'
    if reload_field_info and os.path.exists(field_info_path):
        fields_reloaded = True
        with open(field_info_path, 'rb') as f:
            field_infos = pickle.load(f)
        print(f"Reloading: {save_path}\n")
    else:
        fields_reloaded = False
        field_infos = [] # Neuron-size list of (field, field_mask, nan_idxs)
        print(f"Recalculating: {save_path}\n")

    # Go over each neuron
    for neur in np.arange(activity.shape[1]):
        if fields_reloaded:
            field, field_mask, nan_idxs = field_infos[neur]
        else:
            field, nan_idxs = get_firing_field(xs, ys, activity[:, neur], arena_length)
            field_mask = np.zeros(field.shape)
            for _ in range(nshuffles):
                shuffled_field, _ = get_firing_field(
                    xs, ys, circular(activity[:, neur]), arena_length
                    )
                field_mask += (field > shuffled_field)
            zz = np.copy(field_mask)
            field_mask = field_mask > 0.99*nshuffles
            if save_field_info:
                field_infos.append((field, field_mask, nan_idxs))

        sizes, nfield = get_area_and_peaks(field, field_mask, nan_idxs)
        fieldsizes.extend(sizes)
        nfields.append(nfield)

    # If you want to save field info and it was not already loaded from file 
    if save_field_info and not fields_reloaded:
        with open(field_info_path, 'wb') as f:
            pickle.dump(field_infos, f)

    fieldsizes = np.array(fieldsizes)/(arena_length**2)
    nfields = np.array(nfields)
    onefield = np.sum(nfields==1)/nfields.size
    zerofield = np.sum(nfields==0)/nfields.size # For quality checks
    fieldsize_kl = get_kl_fieldsizes(fieldsizes)
    nfield_kl = get_kl_nfields(nfields)
    return np.mean(fieldsizes), np.mean(nfields), onefield,\
        zerofield, fieldsize_kl, nfield_kl

def get_area_and_peaks(field, field_mask, nan_idxs, ignore_nans=False):
    """
    Gets connected components of max-normalized firing field to calculate area
    and number of fields. Area threshold (0.00716) is a ratio calculated from
    the Henrikson & Mosers 2010 area threshold.
    """

    labeled_array, ncomponents = label(field_mask, np.ones((3,3)))
    area_thresh = 0.00716*field_mask.size
    areas = []
    for label_id in np.unique(labeled_array):
        if label_id == 0: continue
        area = np.sum(labeled_array == label_id)
        if area < area_thresh: continue
        areas.append(area)
    return areas, len(areas)

def get_kl_nfields(nfields):
    """ Get KL divergence of nfields distribution from Payne 2021 distribution. """

    nfields = nfields[nfields != 0].copy()
    nfields[nfields > 5] = 5 # Bins are [1, 2, 3, 4, 5+]
    P = np.array([
        np.sum(nfields==num)/nfields.size for num in np.arange(1, 6)
        ])
    Q = configs.payne2021.nfield_distribution

    kl = 0
    for idx in np.arange(Q.size):
        p_x = P[idx]; q_x = Q[idx]
        if p_x == 0: continue
        kl += p_x * np.log(p_x/q_x)
    return kl

def get_kl_fieldsizes(fieldsizes):
    """
    Get KL divergence of field size distribution from Payne 2021 distribution.
    """

    k_P, _, theta_P = gamma.fit(fieldsizes) # shape, scale
    k_Q, theta_Q = configs.payne2021.fieldsize_distribution # shape, scale
    term1 = (k_P - k_Q) * digamma(k_P)
    term2 = -loggamma(k_P)
    term3 = loggamma(k_Q)
    term4 = k_Q * (np.log(theta_Q) - np.log(theta_P))
    term5 = k_P * (theta_P - theta_Q)/theta_Q
    kl = term1 + term2 + term3 + term4 + term5
    return kl

def circular(fr):
    """
    Circularly shuffles a (neur, frames) array of firing rates, neuron by neuron.
    """

    fr = fr.copy()
    shift = np.random.choice(np.arange(1, fr.size))
    if len(fr.shape) == 2:
        num_neur, num_frames = fr.shape
        for neur in range(num_neur):
            shift = np.random.choice(np.arange(1, num_frames))
            fr[neur,:] = np.roll(fr[neur,:], shift)
        return fr
    else:
        return np.roll(fr, shift)

def get_mutual_info_all(xs, ys, fr, num_shuffles):
    """ Gets the spatial mutual information of each cell."""

    num_neurs, num_frames = fr.shape
    spatial_info = np.zeros(num_neurs)
    significance = np.zeros(num_neurs)
    conditions = flatten_xy(xs, ys)
    spatial_info = get_mutual_info(conditions, fr)

    for _ in range(num_shuffles):
        shuffled_fr = circular(fr)
        shuffled_info = get_mutual_info(conditions, shuffled_fr)
        significance += (shuffled_info < spatial_info)
    significance /= num_shuffles

    return spatial_info, significance
