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
from datasets import inputs

class Sim1DWalk(inputs.Sim1DWalk):
    """
    Simulates a walk in a 1D ring, where you can go left/right/stay
    """

    def __init__(
            self, num_steps, left_right_stay_prob=[1, 1, 1], num_states=16,
            feature_dim=32, feature_vals=[0,1], feature_type='linear'
            ):

        super(Sim1DWalk, self).__init__(
            num_states, left_right_stay_prob, num_states
            )
        self.feature_dim = feature_dim
        self.feature_type = feature_type

        if feature_type == 'nhot':
            self.feature_dim = self.feature_dim//self.num_states * num_states
            self.dg_inputs = np.repeat(
                self.dg_inputs, self.feature_dim//self.num_states, axis=0
                )
        elif feature_type == 'linear':
            self.expansion_mat = np.random.choice(feature_vals, (feature_dim, num_states))
            self.dg_inputs = [self.expansion_mat@x for x in self.dg_inputs.T]
            self.dg_inputs = np.array(self.dg_inputs).T
        else:
            raise ValueError(f'Feature type {feature_type} is not an option.')

    def get_true_T(self):
        raise NotImplementedError(
            "Transition matrix is ill-defined for non one-hot features."
            )

class Sim2DWalk(inputs.Sim2DWalk):
    """
    Simulates a 2D random walk around a 10x10 arena.
    """

    def __init__(
            self, num_steps, num_states,
            feature_dim, feature_vals=[0,1], feature_type='linear'
            ):

        super(Sim2DWalk, self).__init__(num_steps, num_states)
        self.feature_dim = feature_dim
        self.feature_type = feature_type

        if feature_type == 'nhot':
            self.feature_dim = self.feature_dim//self.num_states * num_states
            self.dg_inputs = np.repeat(
                self.dg_inputs, self.feature_dim//self.num_states, axis=0
                )
        elif feature_type == 'linear':
            self.expansion_mat = np.random.choice(feature_vals, (feature_dim, num_states))
            self.dg_inputs = [self.expansion_mat@x for x in self.dg_inputs.T]
            self.dg_inputs = np.array(self.dg_inputs).T
        else:
            raise ValueError(f'Feature type {feature_type} is not an option.')

    def get_true_T(self):
        raise NotImplementedError(
            "Transition matrix is ill-defined for non one-hot features."
            )

