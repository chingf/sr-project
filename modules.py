import numpy as np
from utils import get_sr

import matplotlib.pyplot as plt

class DG(object):
    def __init__(self):
        pass

    def forward(self, input):
        out = input.copy()
        out[out != np.max(out)] = 0  # WTA dynamics
        return out

class CA3(object):
    def __init__(self, gamma, num_states):
        self.T = 0.001*np.random.rand(num_states, num_states)
        self.gamma = gamma
        self.prev_input = None
        self.curr_input = None
    
    def forward(self, input):
        M = get_sr(self.T, self.gamma)
        self.prev_input = self.curr_input


    def update(self):
        if self.prev_input is None:
            return
        self.T[self.prev_input>0, self.curr_input>0] += 1

    def get_T(self):
        return self.T

class STDP_Net(object):
    """
    STDP function k(x) is:
        k(x) = A_pos * exp{-x/tau_pos} for x > 0
        k(x) = -A_neg * exp{x/tau_neg} for x < 0
        where x = t_post - t_pre
    Notably, J = T^T where T is the SR transition matrix since J follows the
    convention of (to, from) for value (i, j).
    """

    def __init__(
            self, num_states,
            A_pos=1.4, tau_pos=0.3, A_neg=0., tau_neg=1,
            dt=0.1, decay_scale=0., tau_J=1, B_offset=0.5, update_min=1e-2
            self_potentiation=0.02,
            global_inhib=0., activity_ceil=1.8, gamma_0=0.5
            ):

        self.J = np.clip(0.01*np.random.rand(num_states, num_states), 0, 1)
        self.B_pos = np.zeros(num_states)
        self.B_neg = np.zeros(num_states)
        self.neuron_gain = 1/np.sum(self.J, axis=0)#np.ones(num_states)
        self.num_states = num_states

        self.A_pos = A_pos
        self.tau_pos = tau_pos
        self.A_neg = A_neg
        self.tau_neg = tau_neg
        self.dt = dt
        self.decay_scale = decay_scale
        self.tau_J = tau_J
        self.B_offset = B_offset
        self.update_min = update_min
        self.self_potentiation = self_potentiation
        self.global_inhib = global_inhib
        self.activity_ceil = activity_ceil
        self.gamma_0 = gamma_0

        self.curr_input = None
        self.prev_input = None

        # Below variables are useful for debugging
        self.real_T = 0.001*np.random.rand(num_states, num_states)
        self.X = np.zeros(num_states)*np.nan
        self.allX = np.zeros((num_states, 40*16))
        self.allX_t = 0
        self.last_update = np.zeros(self.J.shape)
        self.synapse_count = np.zeros(self.J.shape) # Refractory period?
    
    def forward(self, input):
        """ Returns activity of network given input. """

        M_hat = self.get_M_hat()
        activity = M_hat @ (input-self.global_inhib)
        activity = np.clip(activity, 0, self.activity_ceil)
        self.prev_input = np.copy(self.curr_input)
        self.curr_input = input
        self.X = activity

        # Update variables for debugging purposes
        self.allX[:,self.allX_t] = activity
        self.allX_t += 1

        return activity

    def update(self):
        """ Plasticity update """

        if self.prev_input.shape == (): return

        for k in np.arange(self.num_states):
            self._update_B_pos(k)
            self._update_B_neg(k)

        for i in np.arange(self.num_states):
            for j in np.arange(self.num_states):
                self._update_J_ij(i, j)

        for k in np.arange(self.num_states):
            self._update_neuron_gain(k)
                
        self.real_T[self.prev_input>0, self.curr_input>0] += 1

    def get_stdp_kernel(self):
        """ Returns plasticity kernel for plotting or debugging. """

        k_len = 15
        k = np.zeros(k_len)
        half_len = k_len//2
        k[:half_len] = -self.A_neg * np.exp(np.arange(-half_len, 0)/self.tau_neg)
        k[-half_len-1:] = self.A_pos * np.exp(-1*np.arange(half_len+1)/self.tau_pos)
        return k

    def get_M_hat(self, gamma=None):
        """ Inverts the learned T matrix to get putative M matrix """

        if gamma is None:
            gamma = self.gamma_0
        T = self.get_T()
        return np.linalg.pinv(np.eye(T.shape[0]) - gamma*T.T)

    def get_T(self):
        """ Returns the learned T matrix, where T = J^T. """

        return (self.J * self.neuron_gain).T

    def get_real_T(self):
        """ Returns the ideal T matrix """

        return self.real_T/np.sum(self.real_T, axis=1)

    def _update_J_ij(self, i, j):
        decay = 1 - self.decay_scale*self.dt
        if i == j:
            update = self.X[i]*self.self_potentiation
            if update < self.update_min:  update = 0
            self.J[i,j] = decay*self.J[i,j] + update

            # DEBUG
            self.last_update[i,j] += update
        else:
            potentiation = (self.dt/self.tau_J) * self.X[i] * self.B_pos[j]
            depression = (self.dt/self.tau_J) * self.X[j] * self.B_neg[i]
            update = potentiation + depression
            if update < self.update_min:  update = 0
            self.J[i,j] = decay*self.J[i,j] + update

            # DEBUG
            self.last_update[i,j] += update 
        self.J[i,j] = np.clip(self.J[i,j], 0, 1)

    def _update_B_pos(self, k):
        decay = 1 - self.dt/self.tau_pos
        activity = max(0, self.X[k] - self.B_offset)
        self.B_pos[k] = decay*self.B_pos[k] + self.dt*self.A_pos*activity

    def _update_B_neg(self, k):
        decay = 1 - self.dt/self.tau_neg
        activity = max(0, self.X[k] - self.B_offset)
        self.B_neg[k] = decay*self.B_neg[k] + self.dt*self.A_neg*activity

    def _update_neuron_gain(self, k):
        self.neuron_gain = 1/np.sum(self.J, axis=0)

