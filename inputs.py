import numpy as np
import h5py
from scipy.stats import binned_statistic_2d
from math import ceil, sqrt
from utils import pol2cart, downsample
from analysis.config import h5_path_dict
from analysis.ExpData import ExpData

class RBYXYWalk(object):
    def __init__(self, num_states, downsample_factor):
        f = h5py.File(h5_path_dict['RBY45'][3].as_posix(), 'r')
        self.exp_data = ExpData(f)
        self.xmin = 0; self.xmax = 425
        self.ymin = 0; self.ymax = 425
        self.downsample_factor = downsample_factor
        self.num_states = int((ceil(sqrt(num_states)) + 1)**2)
        self.num_xybins = int(sqrt(self.num_states))
        self.dg_inputs, self.dg_modes, self.xs, self.ys, self.zs = self._walk()
        self.num_steps = self.dg_inputs.shape[0]
        self.sorted_states = np.argsort(-np.sum(self.dg_inputs, axis=1)).squeeze()

    def unravel_state_vector(self, state_vector):
        """
        Given (num_states, ) vector of state activation, returns
        (xybins, xybins, 1) size matrix useful for visualization.
        Since this is a spatial only task, the third, context-dependent,
        dimension is irrelevant.
        """

        unraveled_res = np.zeros((self.num_xybins, self.num_xybins, 1))
        xbins, ybins = np.unravel_index(
            np.arange(self.num_states), (self.num_xybins, self.num_xybins)
            )
        unraveled_res[xbins, ybins] = state_vector[:, np.newaxis]
        return unraveled_res

    def get_onehot_states(self, xs, ys, zs):
        """
        Given (T,) size xyz vectors, returns (num_states, T) one-hot encoding
        of the associated state at each time point.
        """

        encoding = np.zeros((self.num_states, xs.size))
        _, _, _, states = binned_statistic_2d(
            xs, ys, xs, bins=self.num_xybins-2
            )
        encoding[states, np.arange(states.size)] = 1
        return encoding

    def _walk(self):
        xs = self.exp_data.x
        ys = self.exp_data.y
        valid_frames = np.zeros(xs.size).astype(bool)
        for idx, hop_end in enumerate(self.exp_data.hop_ends):
            if idx+1 < self.exp_data.hop_starts.size:
                next_hop_start = self.exp_data.hop_starts[idx+1]
            else:
                next_hop_start = self.exp_data.num_frames
            end = min(next_hop_start, hop_end+40)
            valid_frames[hop_end:end] = True
        xs = xs[valid_frames]
        ys = ys[valid_frames]
        if self.downsample_factor is not None:
            xs = downsample(xs, self.downsample_factor)
            ys = downsample(ys, self.downsample_factor)
        zs = np.zeros(xs.size)
        dg_inputs = self.get_onehot_states(xs, ys, zs)
        dg_modes = np.zeros(xs.size)
        return dg_inputs, dg_modes, xs, ys, zs
