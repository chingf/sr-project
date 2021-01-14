import numpy as np
from utils import get_sr

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
    
    def forward(self, input):
        M = get_sr(self.T, self.gamma)
        return M @ input

    def update(self, input, prev_input):
        if np.sum(input > 0) > 1:
            print("input not one hot")
            raise ValueError("d")
        elif np.sum(prev_input > 0) > 1:
            print("prev input not one hot")
            raise ValueError("d")
        self.T[prev_input>0, input>0] += 1

class STDP_Net(object):
    """
    STDP function W(x) is:
        W(x) = A_pos * exp{-x/tau_pos} for x > 0
        W(x) = -A_neg * exp{x/tau_neg} for x < 0
        where x = t_post - t_pre
    Notably, J = T^T where T is the SR transition matrix since J follows the
    convention of (to, from) for value (i, j)
    """

    def __init__(
            self, gamma, num_states,
            A_pos, tau_pos, A_neg, tau_neg
            ):

        self.J = 0.001*np.random.rand(num_states, num_states)
        self.gamma = gamma
    
    def forward(self, input):
        M = get_sr(self.T, self.gamma)
        return M @ input

    def update(self, input, prev_inputs):
        newT = self.T.copy()
        # Update input as pre-synaptic activation
        newT += self.T
        # Update input as post-synaptic activation
        # Normalize

        #self.T[prev_input>0, input>0] += 

class rSTDP_Net(object):
    pass

