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
        self.num_steps = self.dg_inputs.shape[1]
        self.sorted_states = np.argsort(-np.sum(self.dg_inputs, axis=1)).squeeze()

    def unravel_state_vector(self, state_vector):
        """
        Given (num_states, ) vector of state activation, returns
        (xybins, xybins) size matrix useful for visualization.
        Since this is a spatial only task, the second returned matrix (which
        indicates a different contex) is irrelevant.
        """

        unraveled_res = np.zeros((self.num_xybins, self.num_xybins))
        xbins, ybins = np.unravel_index(
            np.arange(self.num_states), (self.num_xybins, self.num_xybins)
            )
        unraveled_res[xbins, ybins] = state_vector
        return unraveled_res, None

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

class RBYCacheWalk(object):
    def __init__(self, num_spatial_states, downsample_factor):
        f = h5py.File(h5_path_dict['RBY45'][3].as_posix(), 'r')
        self.exp_data = ExpData(f)
        self.xmin = 0; self.xmax = 425
        self.ymin = 0; self.ymax = 425
        self.downsample_factor = downsample_factor
        self.num_spatial_states = int((ceil(sqrt(num_spatial_states)) + 1)**2)
        self.num_xybins = int(sqrt(self.num_spatial_states))
        self.num_states = self.num_spatial_states + 16
        self.dg_inputs, self.dg_modes, self.xs, self.ys, self.zs = self._walk()
        self.num_steps = self.dg_inputs.shape[1]
        self.sorted_states = np.argsort(-np.sum(self.dg_inputs, axis=1)).squeeze()

    def unravel_state_vector(self, state_vector):
        """
        Given (num_states, ) vector of state activation, returns
        (xybins, xybins) size matrix useful for spatial visualization. Also
        returns a (16, 1) matrix useful for context visualization.
        """

        unraveled_spatial = np.zeros((self.num_xybins, self.num_xybins))
        xbins, ybins = np.unravel_index(
            np.arange(self.num_spatial_states), (self.num_xybins, self.num_xybins)
            )
        unraveled_spatial[xbins, ybins] = state_vector[:self.num_spatial_states]
        unraveled_context = np.zeros((16, 1))
        unraveled_context = state_vector[self.num_spatial_states:, np.newaxis]
        return unraveled_spatial, unraveled_context

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
        for t in np.argwhere(zs>0).squeeze():
            encoding[:,t] = 0
            state = self.num_spatial_states + (self.exp_data.wedges[t] - 1)
            encoding[state, t] = 1
        return encoding

    def get_rel_vars(self): #TODO: this should go in a different file
        """
        Will use the average xy position of each wedge to get the state associated
        with each wedge. Will also provide the cache interaction amount of each
        wedge.

        Returns:
            wedge_states: (16,) np array of wedge to state mapping (spatial)
            cache_interaction: (16,) np array of cache interactions at each wedge
        """

        wedge_xs = np.zeros(16)
        wedge_ys = np.zeros(16)
        for wedge in range(16):
            wedge_frames = self.exp_data.wedges == wedge+1
            wedge_xs[wedge] = np.mean(self.exp_data.x[wedge_frames])
            wedge_ys[wedge] = np.mean(self.exp_data.y[wedge_frames])
        wedge_states = np.argmax(
            self.get_onehot_states(wedge_xs, wedge_ys, np.zeros(16)), axis=0
            )

        cache_inputs = self.dg_inputs[self.num_spatial_states:,:]
        cache_inputs = cache_inputs[:, np.sum(cache_inputs, axis=0) > 0]
        cache_inputs = np.argmax(cache_inputs, axis=0)
        cache_interactions = np.histogram(cache_inputs, np.arange(17))[0]
        
        return wedge_states, cache_interactions

    def _walk(self):
        xs = self.exp_data.x
        ys = self.exp_data.y
        zs = np.zeros(xs.size)
        dg_modes = np.zeros(xs.size)
        valid_frames = self.exp_data.speeds > 3

        # Add cache/retrieval modes and states
        exp_data = self.exp_data
        c_hops, r_hops, ch_hops, noncrch_hops = exp_data.get_crch_hops()
        event_window = 1 # in frames
        for idx, hop in enumerate(c_hops): # Cache
            poke = exp_data.event_pokes[exp_data.cache_event][idx]
            zs[poke-event_window:poke+event_window] = 1
            dg_modes[poke-event_window:poke+event_window] = 0
            valid_frames[poke-event_window:poke+event_window] = True
        for idx, hop in enumerate(ch_hops):
            poke = exp_data.event_pokes[exp_data.check_event][idx]
            site = exp_data.event_sites[exp_data.check_event][idx] - 1
            if self.exp_data.cache_present[hop, site]: # Full Check
                zs[poke-event_window:poke+event_window] = 1
                dg_modes[poke-event_window:poke+event_window] = 0
                valid_frames[poke-event_window:poke+event_window] = True
        for idx, hop in enumerate(r_hops): # Retrieval
            poke = exp_data.event_pokes[exp_data.retriev_event][idx]
            zs[poke-event_window:poke+event_window] = 1
            dg_modes[poke-event_window:poke+event_window] = 2
            valid_frames[poke-event_window:poke+event_window] = True
        dg_inputs = self.get_onehot_states(xs, ys, zs)

        # Velocity modulation
        dg_inputs = dg_inputs[:,valid_frames]
        dg_modes = dg_modes[valid_frames]
        xs = xs[valid_frames]
        ys = ys[valid_frames]
        zs = zs[valid_frames]

        # Downsample
        if self.downsample_factor is not None:
            downsample_factor = min(self.downsample_factor, event_window)
            dg_inputs = downsample(dg_inputs, downsample_factor)
            dg_modes = downsample(dg_modes, downsample_factor)
            xs = downsample(xs, downsample_factor)
            ys = downsample(ys, downsample_factor)
            zs = downsample(zs, downsample_factor)

        return dg_inputs, dg_modes, xs, ys, zs
