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

device = 'cpu'

def get_firing_field(xs, ys, activity, arena_length):
    """
    The window size (area*0.12) is a ratio taken from Payne & Aronov 2021.

    Args:
        xs: (frames,) array of x locations
        ys: (frames,) array of y locations
        activity: (frames,) array of activity of one neuron
        arena_length: length of square arena
    """

    firing_field = np.zeros((arena_length, arena_length))*np.nan
    for x in range(arena_length):
        for y in range(arena_length):
            frame_idxs = np.logical_and(xs==x, ys==y)
            fr = np.nanmean(activity[frame_idxs])
            firing_field[x,y] = fr
    nan_idxs = np.isnan(firing_field)
    window = int(np.sqrt(arena_length*arena_length*0.12))
    window = window + 1 if window%2 == 0 else window
    kernel = np.outer(hamming(window), hamming(window))
    firing_field = convolve(firing_field, kernel)
    max_rate = firing_field.max()
    #if max_rate != 0:
    #    firing_field = firing_field/firing_field.max()
    return firing_field, nan_idxs

def get_field_metrics(activity, arena_length, get_spatial_info=False):
    """
    Args:
        activity: (frames, neurs) array of activity of all neurons
        arena_length: length of square arena
    """
    walk_xs = dset.xs.astype(int)
    walk_ys = dset.ys.astype(int)
    arena_length = 10
    
    if get_spatial_info:
        spatial_info, significance = get_mutual_info_all(
            walk_xs, walk_ys, outputs.T, 100
            )
    else:
        spatial_info = None

    areas = []
    ncomps = []

    for neur in np.arange(outputs.shape[1]):
        firing_field = np.zeros((arena_length, arena_length))*np.nan
        for x in range(arena_length):
            for y in range(arena_length):
                frame_idxs = np.logical_and(walk_xs == x, walk_ys == y)
                fr = np.nanmean(outputs[frame_idxs, neur])
                firing_field[x,y] = fr
        num_nonnans = firing_field.size - np.sum(np.isnan(firing_field))
        firing_field[np.isnan(firing_field)] = 0

        # Area?
        area, ncomp = get_area_and_peaks(firing_field)
        areas.append(np.sum(area)/num_nonnans)
        ncomps.append(ncomp)
    return np.array(areas), np.array(ncomps), spatial_info

def get_area_and_peaks(firing_field):
    """
    Gets connected components of max-normalized firing field to calculate area
    and number of fields. Area threshold (0.00716) is a ratio calculated from
    the Henrikson & Mosers 2010 area threshold.
    """

    firing_field = firing_field/firing_field.max()
    area_thresh = ceil(0.00716*firing_field.size)
    masked_field = firing_field > 0.8
    labeled_array, ncomponents = label(masked_field, np.ones((3,3)))
    areas = []
    for label_id in np.unique(labeled_array):
        if label_id == 0: continue
        area = np.sum(labeled_array == label_id)
        if area < area_thresh: continue
        areas.append(area)
    return areas, len(areas)

# Spatial Info

def flatten_xy(walk_xs, walk_ys):
    max_col = walk_ys.max()
    new_bins = walk_xs * max_col + walk_ys
    return new_bins

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

def get_mutual_info(conditions, fr):
    """
    Calculates mutual information between firing rate and a set of conditions

    Args:
        conditions: (frames,) array of conditions
        fr: (neurs, frames) array of firing rates
    Returns:
        (neurs,) array of scaler value mutual information per neuron
    """

    num_neurs, _ = fr.shape
    mean_fr = np.mean(fr, axis=1)
    mutual_info = np.zeros(num_neurs)
    for condn in np.unique(conditions):
        prob = np.sum(conditions==condn)/conditions.size
        condn_mean_fr = np.mean(fr[:,conditions==condn], axis=1)
        log_term = np.log2(condn_mean_fr/mean_fr)
        log_term[np.isnan(log_term)] = 0
        log_term[np.isinf(log_term)] = 0
        mutual_info += prob*condn_mean_fr*log_term
    return mutual_info

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
