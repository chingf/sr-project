import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import h5py
from scipy.stats import binned_statistic_2d
from math import ceil, sqrt
from itertools import permutations
import warnings
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize
from scipy.io import loadmat

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
            self, num_steps, num_states, barriers=None, feature_maker_kwargs=None
            ):

        super(Sim2DWalk, self).__init__(num_steps, num_states, barriers)
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

class TitmouseWalk(object):

    def __init__(
            self, cm_to_bin=5, fps=3, num_steps=np.inf,
            feature_maker_kwargs=None, num_states=14*14
            ):
        self.cm_to_bin = cm_to_bin
        self.fps = fps
        self.num_steps = num_steps
        self.num_states = num_states

        # Load data
        data_dir = '/home/chingf/Code/Payne2021/payne_et_al_2021_data/'
        titmouse_data = [f for f in os.listdir(data_dir) if f.startswith('HT')]
        sampled_session = np.random.choice(titmouse_data)
        path = data_dir + sampled_session
        data = loadmat(path)['B'][0,0]
        self.exp_fps = data[0][0,0]
        self.exp_xs = data[1].squeeze()
        self.exp_ys = data[2].squeeze()

        # Interpolate and downsample original x/y coordinates
        import pandas as pd
        self.exp_xs = pd.Series(self.exp_xs).interpolate().to_numpy()
        self.exp_ys = pd.Series(self.exp_ys).interpolate().to_numpy()

        # Transform to discrete coordinates
        self.contin_xs = self.exp_xs[::int(self.exp_fps/fps)]
        self.contin_ys = self.exp_ys[::int(self.exp_fps/fps)]
        self.contin_xs -= self.contin_xs.min()
        self.contin_ys -= self.contin_ys.min()
        max_pixel = max(self.contin_xs.max(), self.contin_ys.max())
        self.contin_xs = (self.contin_xs/max_pixel)*69 # Assume 70x70 cm box
        self.contin_ys = (self.contin_ys/max_pixel)*69
        self.num_states = int((70/self.cm_to_bin) ** 2)
        self.dg_inputs = self.get_onehot_states(self.contin_xs, self.contin_ys)
        self.dg_modes = np.zeros(self.xs.shape)
        
        # Get features
        self.state_inputs = self.dg_inputs.copy()
        if feature_maker_kwargs is not None:
            feature_maker_kwargs['spatial_dim'] = 2
            self.feature_maker = FeatureMaker(
                self.num_states, **feature_maker_kwargs
                )
            self.dg_inputs = self.feature_maker.make_features(self.state_inputs)

    def get_onehot_states(self, contin_xs, contin_ys):
        """
        Given (T,) size xyz vectors, returns (num_states, T) one-hot encoding
        of the associated state at each time point.
        """

        self.xs = np.digitize(contin_xs, np.arange(0, 70, self.cm_to_bin)) - 1
        self.ys = np.digitize(contin_ys, np.arange(0, 70, self.cm_to_bin)) - 1
        self.arena_length = 70/self.cm_to_bin
        self.states = (self.xs*self.arena_length + self.ys).astype(int)

        # Find runs
        run_starts = []
        run_ends = []
        prev_state = None
        run_length = 0 # 5 seconds is 5*3 = 15 frames
        for frame, state in enumerate(self.states):
            if prev_state == state:
                run_length += 1
            else:
                if run_length >= 15:
                    run_starts.append(frame - run_length) # Inclusive
                    run_ends.append(frame) # Exclusive
                run_length = 1
                prev_state = state

        # Exclude stationary points
        exclude_points = []
        for r_start, r_end in zip(run_starts, run_ends):
            exclude_points += range(r_start+1, r_end)
        self.xs = np.delete(self.xs, exclude_points)
        self.ys = np.delete(self.ys, exclude_points)
        self.states = np.delete(self.states, exclude_points)
       
        # Truncate if needed
        if self.num_steps != np.inf:
            self.xs = self.xs[:self.num_steps]
            self.ys = self.ys[:self.num_steps]
            self.states = self.states[:self.num_steps]

        # Format into encoding
        encoding = np.zeros((self.num_states, self.xs.size))
        encoding[self.states, np.arange(self.states.size)] = 1
        return encoding

class FeatureMaker(object):
    def __init__(
            self, num_states, feature_dim=32, feature_type='linear',
            feature_vals=None, spatial_dim=2, spatial_sigma=2,
            feature_vals_p=None, seed_generation=None, gaussian_truncate=1.
            ):

        self.num_states = num_states
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.feature_type = feature_type
        if feature_vals is not None:
            self.feature_vals = np.array(feature_vals)
        else:
            self.feature_vals = feature_vals
        if feature_vals_p is not None:
            self.feature_vals_p = np.array(feature_vals_p)
        else:
            self.feature_vals_p = np.array([0.5, 0.5]) # 0, 1
        self.spatial_sigma = spatial_sigma
        self.seed_generation = seed_generation
        self.gaussian_truncate = gaussian_truncate

    def make_features(self, dg_inputs):
        num_states = self.num_states
        feature_dim = self.feature_dim
        feature_type = self.feature_type
        feature_vals = self.feature_vals
        if self.seed_generation is not None:
            np.random.seed(self.seed_generation)

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
        if self.seed_generation is not None:
            np.random.seed()
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
        blurred_features = gaussian_filter(
            features, sigma=sigma, truncate=self.gaussian_truncate
            )
        blurred_features = blurred_features.reshape(num_states, feature_dim)
        blurred_features -= np.min(blurred_features, axis=1)[:,None]
        blurred_features = normalize(blurred_features, axis=1, norm='max')

        if feature_vals is not None:
            val_midpoints = (feature_vals[1:] + feature_vals[:-1])/2
            val_bins = np.digitize(blurred_features, val_midpoint)
            blurred_features = feature_vals[val_bins]

        return blurred_features.T # (feature_dim, num_states)

    def _generate_distrib_corr_features(self):
        num_states = self.num_states
        feature_dim = self.feature_dim
        feature_vals = self.feature_vals

        if self.spatial_dim == 1:
            features = np.random.choice(
                [0,1.], size=(num_states, feature_dim),
                p=self.feature_vals_p
                )
            sigma = [self.spatial_sigma, 0]
        else:
            arena_length = int(np.sqrt(num_states))
            features = np.random.choice(
                [0,1.], size=(num_states, feature_dim),
                p=self.feature_vals_p
                )
            sigma = [self.spatial_sigma, self.spatial_sigma, 0]

        # If correlation is zero, zero out non-unique features
        unique_features, unique_indices = np.unique(
            features, axis=0, return_index=True
            ) # (num_states, n_unique_feat), (n_unique_feat)
        non_unique_indices = np.ones(num_states).astype(bool)
        non_unique_indices[unique_indices] = False
        features[non_unique_indices] = 0

        # In case some states are still not encoded
        no_support = np.sum(features, axis=1) == 0
        n_no_support = np.sum(no_support)
        if n_no_support > 0:
            unused_features = []
            for feature in np.arange(feature_dim):
                one_hot_feature = np.zeros(feature_dim)
                one_hot_feature[feature] = 1.
                if not (one_hot_feature.tolist() in features.tolist()):
                    unused_features.append(feature)
            replace = len(unused_features) < n_no_support
            if replace == True:
                print("Warning: could not uniquely encode all states")
            new_support = np.random.choice(
                unused_features, size=n_no_support, replace=replace)
            features[no_support, new_support] = 1.

        # Shape into 2D if needed
        if self.spatial_dim != 1:
            features = features.reshape((arena_length, arena_length, -1))

        # Blur gaussian
        blurred_features = gaussian_filter(
            features, sigma=sigma, truncate=self.gaussian_truncate
            )
        blurred_features = blurred_features.reshape(num_states, feature_dim)
        blurred_features -= np.min(blurred_features, axis=1)[:,None]
        blurred_features = normalize(blurred_features, axis=1, norm='max')

        if feature_vals is not None:
            val_midpoints = (feature_vals[1:] + feature_vals[:-1])/2
            val_bins = np.digitize(blurred_features, val_midpoints)
            blurred_features = feature_vals[val_bins]

        # Store sparsity calculation
        sparsities = []
        for feat in blurred_features:
            sparsities.append(np.sum(feat)/feat.size)
        self.post_smooth_sparsity = np.median(sparsities)

        return blurred_features.T # (feature_dim, num_states)

