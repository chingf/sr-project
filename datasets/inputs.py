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

## Artificial Simulations

class Sim1DWalk(object):
    """
    Simulates a walk in a 1D ring, where you can go left/right/stay
    """

    def __init__(
            self, num_steps, left_right_stay_prob=[1, 1, 1], num_states=16
            ):

        self.num_steps = num_steps
        self.left_right_stay_prob = np.array(left_right_stay_prob)
        self.left_right_stay_prob = self.left_right_stay_prob/np.sum(self.left_right_stay_prob)
        self.num_states = num_states
        self.dg_inputs, self.dg_modes, self.xs, self.ys, self.zs = self._walk()
        self.est_T = get_est_T(self.dg_inputs)

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
            action = np.random.choice([-1,1,0], p=self.left_right_stay_prob)
            curr_state = (curr_state + action)%self.num_states
            ys[step] = curr_state
            dg_inputs[curr_state, step] = 1
        return dg_inputs, dg_modes, xs, ys, zs

class Sim2DWalk(object):
    """
    Simulates a 2D random walk around a 10x10 arena.
    """

    def __init__(
            self, num_steps, num_states, barriers=None
            ):

        self.num_steps = num_steps
        self.num_states = int((ceil(sqrt(num_states)))**2)
        self.num_xybins = int(sqrt(self.num_states))
        self.xmin = 0; self.xmax = self.num_xybins
        self.ymin = 0; self.ymax = self.num_xybins
        self.barriers = barriers
        self.dg_inputs, self.dg_modes, self.xs, self.ys, self.zs = self._walk()
        self.num_steps = self.dg_inputs.shape[1]
        self.sorted_states = np.argsort(-np.sum(self.dg_inputs, axis=1)).squeeze()
        self.est_T = get_est_T(self.dg_inputs)

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

    def get_onehot_states(self, xs, ys):
        """
        Given (T,) size xy vectors, where xys are already in bin indices,
        returns (num_states, T) one-hot encoding of the associated state at
        each time point.
        """

        encoding = np.zeros((self.num_states, xs.size))
        states = np.ravel_multi_index(
            (xs, ys), (self.num_xybins, self.num_xybins), mode='raise'
            )
        encoding[states, np.arange(states.size)] = 1
        return encoding

    def get_true_T(self):
        true_T = np.zeros((self.num_states, self.num_states))
        _get_flat_idx = lambda x, y: x*self.num_xybins + y
        for x in np.arange(self.num_xybins):
            for y in np.arange(self.num_xybins):
                curr_idx = _get_flat_idx(x, y)
                for action in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
                    new_idx = _get_flat_idx(x+action[0], y+action[1])
                    if self._in_range(x, y, action[0], action[1]):
                        true_T[curr_idx, new_idx] = 1
        true_T = true_T/(np.sum(true_T,axis=1)[:,None])
        return true_T

    def _walk(self):
        curr_state = 0
        dg_inputs = np.zeros((self.num_states, self.num_steps))
        dg_modes = np.zeros((self.num_steps))
        xs = np.zeros(self.num_steps).astype(int)
        xs[0] = curr_x = np.random.choice(self.num_xybins)
        ys = np.zeros(self.num_steps).astype(int)
        ys[0] = curr_y = np.random.choice(self.num_xybins)
        zs = np.zeros(self.num_steps).astype(int)
        actions = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
        for step in np.arange(1, self.num_steps):
            move_set = [
                move for move in actions if\
                    self._in_range(curr_x, curr_y, move[0], move[1])
                ]
            move = move_set[np.random.choice(len(move_set))]
            xs[step] = curr_x + move[0]
            ys[step] = curr_y + move[1]
            curr_x = xs[step]
            curr_y = ys[step]
        dg_inputs = self.get_onehot_states(xs, ys)
        dg_modes = np.zeros(xs.size)
        return dg_inputs, dg_modes, xs, ys, zs
    
    def _in_range(self, x, y, delta_x, delta_y):
        new_x = x + delta_x; new_y = y + delta_y;
        in_range = (0 <= new_x < self.num_xybins) and (0 <= new_y < self.num_xybins)
        if self.barriers == 'states':
            midpoint = self.num_xybins//2
            y_tip = min(6, midpoint)
            if (new_x == midpoint) and (new_y < y_tip):
                in_range = False
        elif self.barriers == 'transitions':
            midpoint = self.num_xybins//2
            y_tip = min(6, midpoint)
            if (x == midpoint-1) and (new_y < y_tip) and (delta_x == 1):
                in_range = False
            elif (x == midpoint) and (new_y < y_tip) and (delta_x == -1):
                in_range = False
        return in_range

class Sim2DLevyFlight(object):
    """
    Simulates a 2D Levy flight around a 10x10 arena.
    """

    def __init__(
            self, num_steps, walls, alpha=2, beta=1
            ):

        self.alpha = alpha
        self.beta = beta
        self.walls = walls
        self.num_steps = num_steps
        self.num_states = (walls+1)**2
        self.xmin = 0; self.xmax = walls
        self.ymin = 0; self.ymax = walls
        self.num_xybins = walls+1
        self.dg_inputs, self.dg_modes, self.xs, self.ys, self.zs = self._walk()
        self.num_steps = self.dg_inputs.shape[1]
        self.sorted_states = np.argsort(-np.sum(self.dg_inputs, axis=1)).squeeze()
        self.est_T = get_est_T(self.dg_inputs)

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

    def get_onehot_states(self, xs, ys):
        """
        Given (T,) size xy vectors, returns (num_states, T) one-hot encoding of
        the associated state at each time point.
        """

        encoding = np.zeros((self.num_states, xs.size))
        _, _, _, states = binned_statistic_2d(
            xs, ys, xs, bins=self.num_xybins-2
            )
        encoding[states, np.arange(states.size)] = 1
        return encoding

    def get_true_T(self):
        warnings.warn("True T estimate not implemented in Levy Flight dataset")
        true_T = np.zeros((self.num_states, self.num_states))
        return true_T

    def _walk(self):
        dg_inputs = self.get_onehot_states(xs, ys, zs)
        dg_modes = np.zeros(xs.size)
        return dg_inputs, dg_modes, xs, ys, zs

    def _walk(self):
        xs = [self.walls/2]; ys = [self.walls/2];
        thetas = 360*np.random.uniform(size=self.num_steps)
        rhos = np.minimum(
            np.random.gamma(shape=self.alpha, scale=self.beta, size=self.num_steps),
            np.ones(self.num_steps)*self.walls
            )
        #rhos = np.maximum( # No stay transitions
        #    rhos,
        #    np.ones(self.num_steps)
        #    )
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
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.zeros(xs.size)
        dg_inputs = self.get_onehot_states(xs, ys)
        dg_modes = np.zeros(xs.shape)
        return dg_inputs, dg_modes, xs, ys, zs
    
    def _in_range(self, x, y):
        return (0 <= x < self.num_xybins) and (0 <= y < self.num_xybins)

class Sim1DFeeder(object):
    """
    Simulates a walk in a 1D ring, where you can go left/right/stay. Agent is
    attracted to a specific area in the ring.
    """

    def __init__(
            self, num_steps, left_right_stay_prob=[1, 1, 1], num_states=16,
            attractor_center=0, attractor_width=2, attractor_scale=3
            ):

        self.num_steps = num_steps
        self.left_right_stay_prob = np.array(left_right_stay_prob)
        self.left_right_stay_prob = self.left_right_stay_prob/np.sum(self.left_right_stay_prob)
        self.num_states = num_states
        self.attractor_center = attractor_center
        self.attractor_width = attractor_width
        self.attractor_scale = attractor_scale
        self.true_T = self.generate_T()
        self.dg_inputs, self.dg_modes, self.xs, self.ys, self.zs = self._walk()
        self.est_T = get_est_T(self.dg_inputs)

    def generate_T(self):
        true_T = np.zeros((self.num_states, self.num_states))
        for i in range(self.num_states):
            true_T[i, i-1] = self.left_right_stay_prob[0]
            true_T[i, i] = self.left_right_stay_prob[1]
            true_T[i, (i+1)%self.num_states] = self.left_right_stay_prob[2]
        attr_states = np.arange(
            self.attractor_center - self.attractor_width,
            self.attractor_center + self.attractor_width + 1
            )
        for idx, s in enumerate(attr_states):
            reflect_p = self.attractor_scale/(self.attractor_scale+1)
            if idx == 0: # Edge state
                true_T[s,:] = 0
                true_T[s, (s-1)%self.num_states] = 1 - reflect_p
                true_T[s, s] = reflect_p/2.
                true_T[s, (s+1)%self.num_states] = reflect_p/2.
            elif idx == attr_states.size-1: # Edge state
                true_T[s,:] = 0
                true_T[s, (s-1)%self.num_states] = reflect_p/2.
                true_T[s, s] = reflect_p/2.
                true_T[s, (s+1)%self.num_states] = 1 - reflect_p
            else:
                true_T[s,:] = 0
                true_T[s, (s-1)%self.num_states] = 1/3.
                true_T[s, s] = 1/3.
                true_T[s, (s+1)%self.num_states] = 1/3.
        return true_T

    def get_true_T(self):
        return self.true_T

    def _walk(self):
        curr_state = 0
        dg_inputs = np.zeros((self.num_states, self.num_steps))
        dg_modes = np.zeros((self.num_steps))
        xs = np.zeros(self.num_steps)
        ys = np.zeros(self.num_steps)
        zs = np.zeros(self.num_steps)
        for step in np.arange(self.num_steps):
            action_range = np.mod(np.arange(curr_state-1, curr_state+2), self.num_states)
            p = self.true_T[curr_state, action_range]
            action = np.random.choice([-1,0,1], p=p)
            curr_state = (curr_state + action)%self.num_states
            ys[step] = curr_state
            dg_inputs[curr_state, step] = 1
        return dg_inputs, dg_modes, xs, ys, zs

class Sim1DFeederAndCache(object):
    """
    Simulates a forward-biased walk in a 1D ring. Feeder opens for a brief
    interval within the session. When feeder is open, the bird visits the feeder
    at every lap. One cache occurs during the feeder open period.

    Args:
        steps_in_phases: array with number of timesteps for feeder off/on/off.
        num_spatial_states: num of spatial states; total states is twice this.
        left_right_stay_prob: sets the amount of forward bias in the walk
        feeder_state: spatial state where the feeder is.
        cache_state: spatial state where the cache is made.
        visit_length: how many timesteps a visit to feeder/cache state is.
    """

    def __init__(
            self, steps_in_phases=[500, 500, 500], num_spatial_states=16,
            left_right_stay_prob=[6, 1, 1],
            feeder_states=[8], cache_states=[12],
            visit_length=1, expansion=1, cache_prob=0.1
            ):

        self.steps_in_phases = steps_in_phases
        self.num_spatial_states = num_spatial_states
        self.left_right_stay_prob = np.array(left_right_stay_prob)
        self.left_right_stay_prob = self.left_right_stay_prob/np.sum(self.left_right_stay_prob)
        self.feeder_states = feeder_states
        self.cache_states = cache_states
        self.visit_length = visit_length
        self.expansion = expansion
        self.cache_prob = cache_prob
        self.num_steps = np.sum(steps_in_phases)
        self.num_states = num_spatial_states*2
        self.num_expanded_states = self.num_states * self.expansion
        self.state_inputs, self.expanded_state_ids = self._walk()
        self.expanded_state_inputs = self._expand_states(
            self.state_inputs, self.expanded_state_ids
            )
        self.dg_inputs = self._make_one_hot(self.expanded_state_inputs)
        self.xs, self.ys = self._convert_to_xy(self.state_inputs)

    def _walk(self):
        curr_loc = 0 # Current spatial location
        num_feeder_visits = 0 # Count number of feeder visits
        num_caches = [0 for loc in self.cache_states] # Number of caches made
        next_states = []
        state_inputs = []
        expanded_state_ids = []

        for phase, steps_in_phase in enumerate(self.steps_in_phases):
            for step in np.arange(steps_in_phase):
                # If states have been pre-determined, follow them
                if len(next_states) > 0:
                    next_state = next_states[0]
                    curr_loc = next_state[0] % self.num_spatial_states
                    state_inputs.append(next_state[0])
                    expanded_state_ids.append(next_state[1])
                    next_states = next_states[1:]
                    continue
                # Else, draw transitions as normal
                left_right_stay_prob = self.left_right_stay_prob.copy()
                action = np.random.choice([-1,1,0], p=left_right_stay_prob)
                curr_loc = (curr_loc + action) % self.num_spatial_states
                state_inputs.append(curr_loc)
                expanded_state_id = np.random.choice(self.expansion)
                expanded_state_ids.append(expanded_state_id)
                # Impose feeder transition in next timesteps
                if (phase == 1) and (curr_loc in self.feeder_states):
                    next_states = [(
                        self.num_spatial_states + curr_loc,
                        expanded_state_id
                        )]
                    next_states = next_states * self.visit_length
                    next_states.append((curr_loc, expanded_state_id))
                    num_feeder_visits += 1
                # Impose cache transition in next timesteps
                elif (phase == 1) and (curr_loc in self.cache_states):
                    cache_idx = self.cache_states.index(curr_loc)
                    if num_caches[cache_idx] > 0: continue
                    if np.random.rand() > self.cache_prob: continue
                    next_states = [(
                        self.num_spatial_states + curr_loc,
                        expanded_state_id
                        )]
                    next_states = next_states * self.visit_length
                    next_states.append((curr_loc, expanded_state_id))
                    num_caches[cache_idx] += 1
        return np.array(state_inputs), np.array(expanded_state_ids)

    def _expand_states(self, state_inputs, expanded_state_ids):
        return state_inputs*self.expansion + expanded_state_ids

    def _make_one_hot(self, expanded_state_inputs):
        one_hots = np.zeros((
            self.num_expanded_states, self.num_steps
            ))
        inds = np.column_stack((expanded_state_inputs, np.arange(self.num_steps)))
        np.put(one_hots, inds, 1)
        return one_hots

    def _convert_to_xy(self, state_inputs):
        x_encodings = np.zeros(self.num_states)
        y_encodings = np.zeros(self.num_states)
        radians = np.linspace(0, 2*np.pi, self.num_spatial_states, endpoint=False)
        x_encodings[:self.num_spatial_states] = np.cos(radians)
        y_encodings[:self.num_spatial_states] = np.sin(radians)
        x_encodings[self.num_spatial_states:] = 1.25*np.cos(radians)
        y_encodings[self.num_spatial_states:] = 1.25*np.sin(radians)
        return x_encodings[state_inputs], y_encodings[state_inputs]

## Experiment Simulations

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

    def get_true_T(self):
        warnings.warn("True T estimate not implemented in RBYXY dataset")
        true_T = np.zeros((self.num_states, self.num_states))
        return true_T

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
    def __init__(
            self, num_spatial_states, downsample_factor, skip_frame,
            vel_thresh=0, event_window=1, set_to_wedges=False
            ):

        if set_to_wedges and num_spatial_states != 17:
            raise ValueError("If walk is defined on wedges, spatial states must be 17")

        f = h5py.File(h5_path_dict['RBY45'][3].as_posix(), 'r')
        self.exp_data = ExpData(f)
        self.xmin = 0; self.xmax = 425
        self.ymin = 0; self.ymax = 425

        self.downsample_factor = downsample_factor
        self.skip_frame = skip_frame
        self.vel_thresh = vel_thresh
        self.event_window = event_window
        self.set_to_wedges = set_to_wedges
        if set_to_wedges:
            self.num_spatial_states = num_spatial_states
        else:
            self.num_spatial_states = int(ceil(sqrt(num_spatial_states))**2)
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

        if self.set_to_wedges:
            unraveled_spatial = np.zeros((17, 1))
            unraveled_spatial = state_vector[:17]
        else:
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
        if self.set_to_wedges:
            states = xs - 1
        else:
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

        if self.set_to_wedges:
            wedge_states = np.arange(16)
        else:
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

    def get_true_T(self):
        warnings.warn("True T estimate not implemented in RBYCacheWalk dataset")
        true_T = np.zeros((self.num_states, self.num_states))
        return true_T

    def _walk(self):
        if self.set_to_wedges:
            xs = self.exp_data.wedges
            ys = np.zeros(xs.size)
        else:
            xs = self.exp_data.x
            ys = self.exp_data.y
        zs = np.zeros(xs.size)
        dg_modes = np.zeros(xs.size)

        np.random.seed(0)
        if self.skip_frame is not None:
            valid_frames = np.random.choice(
                [0, 1], xs.size, p=[self.skip_frame, 1-self.skip_frame]
                ).astype(bool)
        else:
            valid_frames = np.ones(xs.size).astype(bool)

        valid_frames = np.logical_and(
            valid_frames, self.exp_data.speeds > self.vel_thresh
            )

        # Add cache/retrieval modes and states
        exp_data = self.exp_data
        c_hops, r_hops, ch_hops, noncrch_hops = exp_data.get_crch_hops()
        event_window = self.event_window # in frames
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

def get_est_T(dg_inputs):
    num_states, num_steps = dg_inputs.shape
    T = np.zeros((num_states, num_states))
    for step in np.arange(1, num_steps):
        T += np.outer(dg_inputs[:,step-1], dg_inputs[:,step])
    T = T/(np.sum(T, axis=1)[:, None]+1E-5)
    return T
    

