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
        self.curr_input = input
        return M @ input

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
    convention of (to, from) for value (i, j). The learning rate of the update
    equation is removed as it can equivalently be pushed to A_pos and A_neg
    """

    def __init__(
            self, num_states,
            A_pos=0.1, tau_pos=1, A_neg=0.1, tau_neg=1, k_len=5
            ):

        self.J = np.clip(0.01*np.random.rand(num_states, num_states), 0, 1)
        self.num_states = num_states
        self.A_pos = A_pos
        self.tau_pos = tau_pos
        self.A_neg = A_neg
        self.tau_neg = tau_neg
        self.k_len = (k_len//2)*2 + 1 #Ensures k_len is an odd number
        self.X_len = self.k_len
        self.k = self.get_stdp_kernel()
        self.X = np.zeros((num_states, self.X_len))*np.nan
        self.allX = np.zeros((num_states, 40*8))
        self.allX_t = 0
        self.update_t = -1
        self.real_T = 0.001*np.random.rand(num_states, num_states)
        self.curr_input = None
        self.prev_input = None
        self.all_updates = []
        self.synapse_count = np.zeros(self.J.shape)
    
    def forward(self, input):
        M_hat = self.get_M_hat()
        activity = M_hat @ (input-0.2) # TODO: param inhib
        activity = np.clip(activity, 0, 1.5) # TODO Param: Clip
        self.X[:,:-1] = self.X[:, 1:]
        self.X[:,-1] = activity
        self.allX[:,self.allX_t] = activity
        self.allX_t += 1

        self.prev_input = np.copy(self.curr_input)
        self.curr_input = input
        return activity

    def update(self):
        self.update_t += 1
        # Update input as pre-synaptic activation
        if np.isnan(self.X[0,0]): return # Ensure queue is full
        printed = False
        updates = np.zeros(self.J.shape)
        for i in np.arange(self.num_states):
            for j in np.arange(i, min(self.num_states, i+2)):
                if not (self.X[i,-1] > 1e-3 or self.X[j,-1] > 1e-3):
                    continue
                xcorr = np.correlate(self.X[i], self.X[j], 'same')
                upd = np.dot(xcorr, self.k)
                if abs(upd) < 8e-2: continue
                if i == j:
                    if self.synapse_count[i,i] <= 0:
                        self_pot = 0.7
                        self.J[i,i] += self_pot
                        updates[i,i] = self_pot
                        self.synapse_count[i,i] = 0
                else:
                    if self.synapse_count[i,j] <= 0:
                        self.J[i,j] += min(np.dot(xcorr, self.k), 0.2)
                        updates[i,j] = min(np.dot(xcorr, self.k), 0.2)
                        self.synapse_count[i,j] = 0
                    if self.synapse_count[j,i] <= 0:
                        self.J[j,i] += min(np.dot(xcorr, self.k[::-1]), 0.2)
                        updates[j,i] = min(np.dot(xcorr, self.k[::-1]), 0.2)
                        self.synapse_count[j,i] = 0
                self.all_updates.append(upd)
                self.all_updates.append(np.dot(xcorr, self.k[::-1]))
                
                if False: 
                    print(f'ij update: {np.dot(xcorr, self.k)}')
                    print(f'ji update: {np.dot(xcorr, self.k[::-1])}')
                    print()
                    fig, axs = plt.subplots(5, 1, figsize=(4, 9))
                    axs[0].plot(self.X[i])
                    axs[0].set_ylim(0, 1)
                    axs[0].set_title(f'Neuron i={i}')
                    axs[1].plot(self.X[j])
                    axs[1].set_ylim(0, 1)
                    axs[1].set_title(f'Neuron j={j}')
                    axs[2].plot(xcorr)
                    axs[2].set_title('Cross correlation')
                    axs[3].plot(self.k)
                    axs[3].set_title('Kernel')
                    axs[4].imshow(self.X)
                    plt.tight_layout(h_pad=5)
                    plt.show(block=False)
                    printed = True
        #plt.figure(); plt.imshow(self.allX[:,:self.update_t]);plt.show()
        #plt.figure();plt.imshow(updates);plt.show()
        #import pdb; pdb.set_trace()
        self.J = np.clip(self.J, 0, 1)
        self.J = self.J/np.sum(self.J, axis=0)
        self.real_T[self.curr_input>0, self.prev_input>0] += 1
        self.synapse_count -= 1

    def get_stdp_kernel(self):
        k = np.zeros(self.k_len)
        half_len = self.k_len//2
        k[:half_len] = -self.A_neg * np.exp(np.arange(-half_len, 0)/self.tau_neg)
        k[-half_len-1:] = self.A_pos * np.exp(-1*np.arange(half_len+1)/self.tau_pos)
        plt.figure(); plt.plot(k); plt.show()
        return k

    def get_M_hat(self):
        return np.linalg.pinv(np.eye(self.J.shape[0]) - self.J)

    def get_T(self):
        return self.J

    def get_real_T(self):
        return self.real_T/np.sum(self.real_T, axis=0)

