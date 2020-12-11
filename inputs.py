import numpy as np
import h5py
from scipy.stats import binned_statistic_2d
from math import ceil, sqrt
from utils import pol2cart, downsample
from analysis.config import h5_path_dict
from analysis.ExpData import ExpData

class RandomWalk(object):
    def __init__(self, walls, num_steps):
        self.walls = walls
        self.num_steps = num_steps
        self.num_states = (walls+1)**2
        self.xmin = 0; self.xmax = walls
        self.ymin = 0; self.ymax = walls
        self.num_xybins = walls+1
        self.states, self.xs, self.ys = self._walk()

    def get_xybins(self, states):
        xbins, ybins = np.unravel_index(states, (self.num_xybins, self.num_xybins))
        return xbins, ybins

    def get_states(self, xs, ys):
        _, _, _, states = binned_statistic_2d(
            xs, ys, xs, bins=np.arange(self.walls)+1
            )
        return states

    def _walk(self):
        xs = [self.walls/2]; ys = [self.walls/2];
        thetas = 360*np.random.uniform(size=self.num_steps)
        rhos = np.minimum(
            np.random.gamma(shape=2, scale=1, size=self.num_steps),
            np.ones(self.num_steps)*self.walls
            )
        delta_xs, delta_ys = pol2cart(thetas, rhos)
        for step in np.arange(1, self.num_steps):
            xs.append(xs[step-1] + delta_xs[step])
            ys.append(ys[step-1] + delta_ys[step])
            if xs[step] > self.walls:
                xs[step] = self.walls - (xs[step] - self.walls)
            elif xs[step] < 0:
                xs[step] = - xs[step]
            if ys[step] > self.walls:
                ys[step] = self.walls - (ys[step] - self.walls)
            elif ys[step] < 0:
                ys[step] = - ys[step]
        states = self.get_states(xs, ys)
        return states, xs, ys

class RBYXYWalk(object):
    def __init__(self, num_states, downsample_factor):
        f = h5py.File(h5_path_dict['RBY45'][3].as_posix(), 'r')
        self.exp_data = ExpData(f)
        self.xmin = 0; self.xmax = 425
        self.ymin = 0; self.ymax = 425
        self.downsample_factor = downsample_factor
        self.num_states = int((ceil(sqrt(num_states)) + 1)**2)
        self.num_xybins = int(sqrt(self.num_states))
        self.states, self.xs, self.ys = self._walk()
        self.num_steps = self.states.size
        self.sorted_states = np.argsort(-np.bincount(self.states)).squeeze()

    def get_xybins(self, states):
        xbins, ybins = np.unravel_index(states, (self.num_xybins, self.num_xybins))
        return xbins, ybins

    def get_states(self, xs, ys):
        bin_edges = np.linspace(
            self.xmin, self.xmax, self.num_xybins-1, endpoint=True
            )[1:]
        _, _, _, states = binned_statistic_2d(
            xs, ys, xs, bins=bin_edges
            )
        return states

    def _walk(self):
        speed_valid = self.exp_data.speeds > -1
        xs = self.exp_data.x[speed_valid]
        ys = self.exp_data.y[speed_valid]
        if self.downsample_factor is not None:
            xs = downsample(xs, self.downsample_factor)
            ys = downsample(ys, self.downsample_factor)
        states = self.get_states(xs, ys)
        return states, xs, ys

class LMNXYWalk(object):
    def __init__(self, num_states, downsample_factor, speed_thresh):
        f = h5py.File(h5_path_dict['LMN73'][4].as_posix(), 'r')
        self.exp_data = ExpData(f)
        self.xmin = 0; self.xmax = 450
        self.ymin = 0; self.ymax = 450
        self.downsample_factor = downsample_factor
        self.speed_thresh = speed_thresh
        self.num_states = int((ceil(sqrt(num_states)) + 1)**2)
        self.num_xybins = int(sqrt(self.num_states))
        self.states, self.xs, self.ys = self._walk()
        self.num_steps = self.states.size
        self.sorted_states = np.argsort(-np.bincount(self.states)).squeeze()

    def get_xybins(self, states):
        xbins, ybins = np.unravel_index(states, (self.num_xybins, self.num_xybins))
        return xbins, ybins

    def get_states(self, xs, ys):
        _, _, _, states = binned_statistic_2d(
            self.ymax - ys, xs, self.ymax-ys, bins=self.num_xybins-2#bin_edges
            )
        return states

    def _walk(self):
        speed_valid = self.exp_data.speeds > self.speed_thresh
        xs = self.exp_data.x[speed_valid]
        ys = self.exp_data.y[speed_valid]
        if self.downsample_factor is not None:
            xs = downsample(xs, self.downsample_factor)
            ys = downsample(ys, self.downsample_factor)
        states = self.get_states(xs, ys)
        return states, xs, ys

class LMNXYHop(object):
    def __init__(self, num_states, downsample_factor):
        f = h5py.File(h5_path_dict['LMN73'][4].as_posix(), 'r')
        self.exp_data = ExpData(f)
        self.xmin = 0; self.xmax = 450
        self.ymin = 0; self.ymax = 450
        self.downsample_factor = downsample_factor
        self.num_states = int((ceil(sqrt(num_states)) + 1)**2)
        self.num_xybins = int(sqrt(self.num_states))
        self.states, self.xs, self.ys = self._walk()
        self.num_steps = self.states.size
        self.sorted_states = np.argsort(-np.bincount(self.states)).squeeze()

    def get_xybins(self, states):
        xbins, ybins = np.unravel_index(states, (self.num_xybins, self.num_xybins))
        return xbins, ybins

    def get_states(self, xs, ys):
        _, _, _, states = binned_statistic_2d(
            self.ymax - ys, xs, self.ymax-ys, bins=self.num_xybins-2#bin_edges
            )
        return states

    def _walk(self):
        speed_valid = np.zeros(self.exp_data.x.size).astype(bool)
        for i, hop_start in enumerate(self.exp_data.hop_starts):
            hop_end = self.exp_data.hop_ends[i]
            speed_valid[hop_start:hop_end+1] = True
        xs = self.exp_data.x[speed_valid]
        ys = self.exp_data.y[speed_valid]
        if self.downsample_factor is not None:
            xs = downsample(xs, self.downsample_factor)
            ys = downsample(ys, self.downsample_factor)
        states = self.get_states(xs, ys)
        return states, xs, ys

class FakeCacheWalk(object):
    """
    The last 16 states will represent the 16 cache sites. Zs represent a third,
    context-dependent axis
    """

    def __init__(self, num_states, downsample_factor):
        f = h5py.File(h5_path_dict['RBY45'][3].as_posix(), 'r')
        self.exp_data = ExpData(f)
        self.xmin = 0; self.xmax = 450
        self.ymin = 0; self.ymax = 450
        self.downsample_factor = downsample_factor
        self.num_states = int((ceil(sqrt(num_states)) + 1)**2) + 16
        self.num_xybins = int(sqrt(self.num_states))
        self.states, self.xs, self.ys, self.zs = self._walk()
        self.num_steps = self.states.size
        state_bincounts = np.bincount(self.states)
        state_bincounts[fakesite] = state_bincounts.max() + 1
        self.sorted_states = np.argsort(-state_bincounts).squeeze()

    def get_xybins(self, states): # TODO: decide on a convention for this
        xbins, ybins = np.unravel_index(states, (self.num_xybins, self.num_xybins))
        return xbins, ybins

    def get_states(self, xs, ys):
        _, _, _, states = binned_statistic_2d(
            ys, xs, ys, bins=self.num_xybins-2
            )
        return states

    def _walk(self):
        xs = self.exp_data.x
        ys = self.exp_data.y
        zs = np.zeros(xs.size)
        valid_frame = np.zeros(xs.size).astype(bool)
        for idx, hop_end in enumerate(self.exp_data.hop_ends):
            end = min(self.exp_data.hop_starts[idx+1], hop_end+10)
            valid_frame[hop_end:end] = True
        for idx, poke in self.exp_data.event_pokes:
            if self.exp_data.check_event[idx]: continue
            end = self.exp_data.hop_starts[np.argwhere(self.exp_data.hops==start).item()+1]
            valid_frame[start:end] = True
            zs[start:end] = 1
        xs = xs[speed_valid]
        ys = ys[speed_valid]
        if self.downsample_factor is not None:
            xs = downsample(xs, self.downsample_factor)
            ys = downsample(ys, self.downsample_factor)
        states = self.get_states(xs, ys)
        return states, xs, ys

