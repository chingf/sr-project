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

root = os.path.dirname(os.path.abspath(os.curdir))
sys.path.append(root)

from sr_model.models.models import AnalyticSR, STDP_SR, Linear, MLP
from datasets import inputs, sf_inputs_discrete
from run_td_rnn import run as run_rnn
from run_td_mlp import run as run_mlp
from run_td_linear import run as run_linear
from findpeaks import findpeaks
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
        nshuffles=50, save_field_info=False, save_path=None
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


    field_infos = [] # Neuron-size list of (field, field_mask, nan_idxs)

    for neur in np.arange(activity.shape[1]):
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

        # Area?
        sizes, nfield = get_area_and_peaks(field, field_mask, nan_idxs)
        fieldsizes.extend(sizes)
        nfields.append(nfield)

    if save_field_info:
        with open(save_path + 'field_infos.p', 'wb') as f:
            pickle.dump(field_infos, f)

    fieldsizes = np.array(fieldsizes)/(arena_length**2)
    nfields = np.array(nfields)
    onefield = np.sum(nfields==1)/nfields.size,
    nfield_kl = get_kl_payne(nfields)
    return np.mean(fieldsizes), np.mean(nfields), onefield, nfield_kl

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

def get_kl_payne(nfields):
    """ Get KL divergence of nfields distribution from Payne 2021 distribution. """

    P = np.array([
        np.sum(nfields==num)/nfields.size for num in np.arange(nfields.max() + 1)
        ])
    Q = configs.payne2021.nfield_distribution

    kl = 0
    for idx in np.arange(max(P.size, Q.size)):
        p_x = P[idx] if idx < P.size else 0
        q_x = Q[idx] if idx < Q.size else 0
        kl += p_x * np.log(p_x/q_x)
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
