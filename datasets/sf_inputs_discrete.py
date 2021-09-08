import numpy as np
import h5py
from scipy.stats import binned_statistic_2d
from math import ceil, sqrt
from itertools import permutations
import warnings
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize

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
            feature_maker_kwargs=None
            ):

        super(Sim1DWalk, self).__init__(
            num_states, left_right_stay_prob, num_states
            )
        feature_maker_kwargs['spatial_dim'] = 1
        self.feature_maker = FeatureMaker(
            self.num_states, **feature_maker_kwargs
            )
        self.state_inputs = self.dg_inputs.copy()
        self.dg_inputs = self.feature_maker.make_features(self.dg_inputs)

    def get_true_T(self):
        raise NotImplementedError(
            "Transition matrix is ill-defined for non one-hot features."
            )

class Sim2DWalk(inputs.Sim2DWalk):
    """
    Simulates a 2D random walk around a 10x10 arena.
    """

    def __init__(
            self, num_steps, num_states, feature_maker_kwargs=None
            ):

        super(Sim2DWalk, self).__init__(num_steps, num_states)
        feature_maker_kwargs['spatial_dim'] = 2
        self.feature_maker = FeatureMaker(
            self.num_states, **feature_maker_kwargs
            )
        self.state_inputs = self.dg_inputs.copy()
        self.dg_inputs = self.feature_maker.make_features(self.dg_inputs)

    def get_true_T(self):
        raise NotImplementedError(
            "Transition matrix is ill-defined for non one-hot features."
            )

class Sim2DLevyFlight(inputs.Sim2DLevyFlight):
    """
    Simulates a 2D Levy flight around a 10x10 arena.
    """

    def __init__(
            self, num_steps, walls, alpha=2, beta=1, feature_maker_kwargs=None
            ):

        super(Sim2DLevyFlight, self).__init__(num_steps, walls, alpha, beta)
        feature_maker_kwargs['spatial_dim'] = 2
        self.feature_maker = FeatureMaker(
            self.num_states, **feature_maker_kwargs
            )
        self.state_inputs = self.dg_inputs.copy()
        self.dg_inputs = self.feature_maker.make_features(self.dg_inputs)

class FeatureMaker(object):
    def __init__(
            self, num_states, feature_dim=32, feature_type='linear',
            feature_vals=[0,1], spatial_dim=2, spatial_sigma=2,
            feature_vals_p=None
            ):

        self.num_states = num_states
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.feature_type = feature_type
        if feature_vals is not None:
            self.feature_vals = np.array(feature_vals)
            self.feature_vals_p = np.array(feature_vals_p)
        else:
            self.feature_vals = feature_vals
        self.spatial_sigma = spatial_sigma

    def make_features(self, dg_inputs):
        num_states = self.num_states
        feature_dim = self.feature_dim
        feature_type = self.feature_type
        feature_vals = self.feature_vals

        if feature_type == 'nhot':
            feature_dim = feature_dim//num_states * num_states
            self.feature_map = np.repeat(
                np.eye(num_states), feature_dim//num_states, axis=0
                )
        elif feature_type == 'linear':
            self.feature_map = np.random.choice(
                feature_vals, (int(feature_dim), num_states),
                p=self.feature_vals_p
                )
        elif feature_type == 'correlated_sparse':
            self.feature_map = self._generate_sparse_corr_features()
        elif feature_type == 'correlated_distributed':
            self.feature_map = self._generate_distrib_corr_features()
        else:
            raise ValueError(f'Feature type {feature_type} is not an option.')
        dg_inputs = [self.feature_map@x for x in dg_inputs.T]
        dg_inputs = np.array(dg_inputs).T
        return dg_inputs

    def _generate_sparse_corr_features(self):
        from scipy.interpolate import interp1d, interp2d
        num_states = self.num_states
        feature_dim = self.feature_dim
        feature_vals = self.feature_vals

        if self.spatial_dim == 1:
            features = np.eye(num_states)
            f = interp1d(np.arange(num_states), features, 'linear', axis=1)
            features = np.apply_along_axis(
                f, 0, np.linspace(0, num_states-1, num=feature_dim)
                ) # (num_states, feature_dim)
            sigma = [self.spatial_sigma, 0]
        else:
            arena_length = int(np.sqrt(num_states))
            x = y = np.arange(arena_length)
            new_x = new_y = np.linspace(
                0, arena_length-1, num=int(np.sqrt(feature_dim))
                )
            features = []
            for state in range(num_states):
                feature = np.zeros(num_states)
                feature[state] = 1.
                feature = feature.reshape((arena_length, arena_length))
                f = interp2d(x, y, feature, kind='linear')
                feature = f(new_x, new_y)
                features.append(feature.flatten())
            # (arena_length, arena_length, feature_dim)
            features = np.array(features).reshape((arena_length, arena_length, -1))
            sigma = [self.spatial_sigma, self.spatial_sigma, 0]
        blurred_features = gaussian_filter(features, sigma=sigma)
        blurred_features -= np.min(blurred_features, axis=1)[:,None]
        blurred_features = normalize(
            blurred_features.reshape(num_states, feature_dim),
            axis=1, norm='max'
            )

        if feature_vals is not None:
            val_midpoints = (feature_vals[1:] + feature_vals[:-1])/2
            val_bins = np.digitize(blurred_features, val_midpoints)
            blurred_features = feature_vals[val_bins]

        return blurred_features.T # (feature_dim, num_states)

    def _generate_distrib_corr_features(self):
        num_states = self.num_states
        feature_dim = self.feature_dim
        feature_vals = self.feature_vals

        if self.spatial_dim == 1:
            features = np.random.choice(
                [0,1.], size=(num_states, feature_dim)
                )
            sigma = [self.spatial_sigma, 0]
        else:
            arena_length = int(np.sqrt(num_states))
            features = np.random.choice(
                [0,1.], size=(arena_length, arena_length, feature_dim)
                )
            sigma = [self.spatial_sigma, self.spatial_sigma, 0]
        blurred_features = gaussian_filter(features, sigma=sigma)
        blurred_features -= np.min(blurred_features, axis=1)[:,None]
        blurred_features = blurred_features.reshape(num_states, feature_dim)
        blurred_features = normalize(blurred_features, axis=1, norm='max')

        if feature_vals is not None:
            val_midpoints = (feature_vals[1:] + feature_vals[:-1])/2
            val_bins = np.digitize(blurred_features, val_midpoints)
            blurred_features = feature_vals[val_bins]

        return blurred_features.T # (feature_dim, num_states)

