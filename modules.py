import numpy as np
from utils import get_sr

class DG(object):
    def __init__(self):
        pass

    def forward(self, input, step): # TODO: expand beyond two-step steady state
        if step == 0:
            return input
        else:
            input[input != np.max(input)] = 0
            return input

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

