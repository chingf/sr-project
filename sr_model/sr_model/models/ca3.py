import numpy as np
import torch
import torch.nn as nn
import module
from utils import get_sr

class CA3(module.Module):
    def __init__(self, num_states, gamma_M0, gamma_T=1.):
        self.T_tilde = np.clip(np.random.rand(num_states, num_states), 0, 1)
        self.T_tilde = 0.001*self.T_tilde*(1/np.sum(self.T_tilde, axis=0))
        self.state_counts = 0.001*np.ones(num_states)
        self.num_states = num_states
        self.gamma_M0 = gamma_M0
        self.gamma_T = gamma_T
        self.prev_input = None
        self.curr_input = None
    
    def forward(self, input, update_transition=True):
        M = get_sr(self.get_T(), self.gamma_M0)
        if update_transition:
            self.prev_input = self.curr_input
            self.curr_input = input

    def update(self):
        if self.prev_input is None:
            return
        self.T_tilde[self.prev_input>0, self.curr_input>0] += 1
        self.state_counts = self.prev_input + self.gamma_T*self.state_counts

    def get_T(self):
        return self.T_tilde/self.state_counts[:,None]

class STDP_CA3(nn.Module):
    """
    STDP function k(x) is:
        k(x) = A_pos * exp{-x/tau_pos} for x > 0
        k(x) = -A_neg * exp{x/tau_neg} for x < 0
        where x = t_post - t_pre
    Notably, J = T^T where T is the SR transition matrix since J follows the
    convention of (to, from) for value (i, j). Learning rate is modulated by
    neuron activity and acts as a probability normalization term.
    """

    def __init__(
            self, num_states, gamma_M0,
            A_pos=7., tau_pos=0.12, A_neg=0., tau_neg=1,
            dt=0.1, tau_J=1, B_offset=0.99, update_min=0, gamma_T=0.99,
            self_potentiation=0.05,
            global_inhib=0., activity_ceil=1.,
            debug=False, debug_print=False
            ):

        self.J = np.clip(np.random.rand(num_states, num_states), 0, 1)
        np.fill_diagonal(self.J, 0)
        self.J = self.J*(1/np.sum(self.J, axis=0))
        self.eta_invs = np.ones(self.J.shape[0])*0.001
        self.B_pos = np.zeros(num_states)
        self.B_neg = np.zeros(num_states)
        self.num_states = num_states

        self.A_pos = A_pos
        self.tau_pos = tau_pos
        self.A_neg = A_neg
        self.tau_neg = tau_neg
        self.dt = dt
        self.tau_J = tau_J
        self.B_offset = B_offset
        self.update_min = update_min
        self.gamma_T = gamma_T
        self.self_potentiation = self_potentiation
        self.global_inhib = global_inhib
        self.activity_ceil = activity_ceil
        self.gamma_M0 = gamma_M0

        self.curr_input = None
        self.prev_input = None

        self.real_T_tilde = 0.001*self.J.T.copy()
        self.real_T_count = 0.001*np.ones(num_states)

        # Below variables are useful for debugging
        self.debug = debug
        self.debug_print = debug_print
        if debug:
            self.X = np.zeros(num_states)*np.nan
            self.allX = np.zeros((num_states, 2000))
            self.allinputs = np.zeros(2000)
            self.allBpos = np.zeros(self.allX.shape)
            self.allX_t = -1
            self.last_update = np.zeros(self.J.shape)
            self.allMs = []
            self.allMs_title = []
            self.allTs = []
    
    def forward(self, input, update_transition=True):
        """ Returns activity of network given input. """

        M_hat = self.get_M_hat()
        activity = M_hat.T @ (input-self.global_inhib)
        activity = np.clip(activity, 0, self.activity_ceil)
        if update_transition:
            self.prev_input = np.copy(self.curr_input)
            self.curr_input = input
            self.X = activity
    
            # DEBUG
            if self.debug:
                try:
                    self.prev_state = np.argwhere(self.prev_input)[0,0]
                except:
                    self.prev_state = -1
                self.curr_state = np.argwhere(self.curr_input)[0,0]
                self.allX_t += 1
                self.allX[:,self.allX_t] = activity
                self.allinputs[self.allX_t] = self.curr_state
                self.allMs.append(M_hat)
                self.allMs_title.append(f'{self.prev_state} to {self.curr_state}')
                self.allTs.append(self.get_T().copy())

        return activity

    def update(self):
        """ Plasticity update """

        for k in np.arange(self.num_states):
            self._update_B_pos(k)
            self._update_B_neg(k)

        if self.prev_input.shape == (): return

        for k in np.arange(self.num_states):
            self._update_eta_invs(k)

        for i in np.arange(self.num_states):
            for j in np.arange(self.num_states):
                if i != j:
                    self._update_J_ij(i, j)
                else:
                    self._update_J_ij(i, j)

        self.real_T_tilde = self.gamma_T*self.real_T_tilde
        self.real_T_tilde[self.prev_input>0, self.curr_input>0] += 1
        self.real_T_count = self.gamma_T*self.real_T_count
        self.real_T_count[self.prev_input>0] += 1

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
            gamma = self.gamma_M0
        T = self.get_T()
        return np.linalg.pinv(np.eye(T.shape[0]) - gamma*T)

    def get_T(self):
        """ Returns the learned T matrix, where T = J^T. """

        return self.J.T

    def get_real_T(self):
        """ Returns the ideal T matrix """

        return self.real_T_tilde/self.real_T_count[:,None]

    def _update_J_ij(self, i, j):
        if self.prev_input[j] != 1: return # TODO: idealized
        eta = 1/self.eta_invs[j]
        decay = 1 - eta
        if i == j:
            activity = 0 if self.X[i] < self.B_offset else self.X[i]
            B_pos = 0 if self.B_pos[i] < self.update_min else self.B_pos[i]
            potentiation = (self.dt/self.tau_J)*activity*B_pos
            update = potentiation*1.65
            self.J[i,j] = decay*self.J[i,j] + eta*update

            # DEBUG
            if self.debug:
                self.last_update[i,j] += update
            if self.debug_print and self.curr_input[i] == 1:
                str1 = f'Correct self {i}: {update:.2f} '
                str2 = f'with i activity {self.X[i]:.2f} '
                str3 = f'and j activity {self.X[j]:.2f} '
                str4 = f'and j plasticity {self.B_pos[j]:.2f}'
                print(str1 + str2 + str3 + str4)
        else:
            activity_i = 0 if self.X[i] < self.B_offset else self.X[i]
            activity_j = 0 if self.X[j] < self.B_offset else self.X[j]
            B_pos = 0 if self.B_pos[j] < self.update_min else self.B_pos[j]
            B_neg = 0 if self.B_neg[i] < self.update_min else self.B_neg[i]
            potentiation = (self.dt/self.tau_J) * activity_i * B_pos
            depression = (self.dt/self.tau_J) * activity_j * B_neg
            update = (potentiation + depression)*10.08
            self.J[i,j] = decay*self.J[i,j] + eta*update

            # DEBUG
            if self.debug:
                self.last_update[i,j] += eta*update
            if self.debug_print and self.curr_input[i] == 1:
                str1 = f'Correct {j} to {i}: {update:.2f} '
                str2 = f'with i activity {self.X[i]:.2f} '
                str3 = f'and j activity {self.X[j]:.2f} '
                str4 = f'and j plasticity {self.B_pos[j]:.2f}'
                print(str1 + str2 + str3 + str4)
            if self.debug_print and self.curr_input[i] != 1 and update > 0:
                str1 = f'Wrong totally, {j} to {i}: {update:.2f} '
                str2 = f'with i activity {self.X[i]:.2f} '
                str3 = f'and j activity {self.X[j]:.2f} '
                str4 = f'and j plasticity {self.B_pos[j]:.2f}'
                print(str1 + str2 + str3 + str4)
        self.J[i,j] = np.clip(self.J[i,j], 0, 1)

    def _update_B_pos(self, k):
        learning_rate = self.dt/self.tau_pos
        decay = 1 - learning_rate
        activity = 0 if self.X[k] < self.B_offset else 1
        self.B_pos[k] = decay*self.B_pos[k] + learning_rate*self.A_pos*activity
        self.B_pos[k] = min(self.B_pos[k], 6)

        # DEBUG
        if self.debug:
            self.allBpos[k, self.allX_t] = self.B_pos[k]

    def _update_B_neg(self, k):
        learning_rate = self.dt/self.tau_neg
        decay = 1 - learning_rate
        activity = 0 if self.X[k] < self.B_offset else 1
        self.B_neg[k] = decay*self.B_neg[k] + learning_rate*self.A_neg*activity

    def _update_eta_invs(self, k):
        self.eta_invs[k] = self.prev_input[k] + self.gamma_T*self.eta_invs[k]

    def _decay_all_eta_invs(self):
        self.eta_invs = self.gamma_T*self.eta_invs


