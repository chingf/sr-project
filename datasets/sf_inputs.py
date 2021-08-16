import numpy as np
import h5py
from scipy.stats import binned_statistic_2d
from math import ceil, sqrt
from itertools import permutations
import warnings

try:
    from analysis.config import h5_path_dict
    from analysis.ExpData import ExpData
except:
    warnings.warn("Emily's experimental data could not be loaded.")

from sr_model.utils import pol2cart, downsample

class Sim1DWalk(object):
    """
    Simulates a walk in a 1D ring, where you can go left/right/stay
    """

    def __init__(
            self, num_steps, left_right_stay_prob=[1, 1, 1], num_states=16,
            feature_dim=32
            ):

        self.num_steps = num_steps
        self.left_right_stay_prob = np.array(left_right_stay_prob)
        self.left_right_stay_prob = self.left_right_stay_prob/np.sum(self.left_right_stay_prob)
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.dg_inputs, self.dg_modes, self.xs, self.ys, self.zs = self._walk()

    def get_true_T(self):
        true_T = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_states):
            true_T[i, i-1] = self.left_right_stay_prob[0]
            true_T[i, i] = self.left_right_stay_prob[1]
            true_T[i, (i+1)%self.num_states] = self.left_right_stay_prob[2]
        return true_T

    def _walk(self):
        curr_state = 0
        dg_inputs = np.zeros((self.num_states, self.num_steps))
        dg_modes = np.zeros((self.num_steps))
        xs = np.zeros(self.num_steps)
        ys = np.zeros(self.num_steps)
        zs = np.zeros(self.num_steps)
        for step in np.arange(self.num_steps):
            action = np.random.choice([-1,0,1], p=self.left_right_stay_prob)
            curr_state = (curr_state + action)%self.num_states
            ys[step] = curr_state
            dg_inputs[curr_state, step] = 1
        dg_inputs = np.repeat(dg_inputs, self.feature_dim//self.num_states, axis=0)
        return dg_inputs, dg_modes, xs, ys, zs
