import numpy as np
import torch
import torch.nn as nn
from sr_model.models import module
from sr_model.utils import get_sr

leaky_slope=1e-5

class CA3(module.Module):
    def __init__(self, num_states, gamma_M0, gamma_T=1.):
        super(CA3, self).__init__()
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
            self, num_states, gamma_M0, gamma_T=0.99,
            debug=False, debug_print=False
            ):

        super(STDP_CA3, self).__init__()
        self.num_states = num_states
        self.gamma_T = gamma_T
        self.gamma_M0 = gamma_M0

        self.reset()
        self._init_trainable()

        self.curr_input = None
        self.prev_input = None

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
        """
        Returns activity of network given input. 

        Args:
            input: (batch, states)
        """

        input = torch.squeeze(input) # TODO: batching
        M_hat = self.get_M_hat()
        activity = torch.matmul(M_hat.t(), input)
        activity = self._forward_activity_clamp(activity)
        if update_transition:
            self.prev_input = self.curr_input # Was cloned?
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

        self._update_B_pos()
        self._update_B_neg()

        if self.prev_input is None: return

        self._update_eta_invs()

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
        k[:half_len] = -self.A_neg.item() * np.exp(
            np.arange(-half_len, 0)/self.tau_neg.item()
            )
        k[-half_len-1:] = self.A_pos.item() * np.exp(
            -1*np.arange(half_len+1)/self.tau_pos.item()
            )
        return k

    def get_M_hat(self, gamma=None):
        """ Inverts the learned T matrix to get putative M matrix """

        if gamma is None:
            gamma = self.gamma_M0
        T = self.get_T()
        return torch.pinverse(torch.eye(T.shape[0]) - gamma*T)

    def get_T(self):
        """ Returns the learned T matrix, where T = J^T. """

        return self.J.t()

    def get_real_T(self):
        """ Returns the ideal T matrix """

        return self.real_T_tilde/self.real_T_count[:,None]

    def set_J_to_real_T(self):
        """ Sets J to the correct transition matrix. """

        self.J = torch.tensor(self.get_real_T().T).float()

    def reset(self):
        self.J = np.clip(np.random.rand(self.num_states, self.num_states), 0, 1)
        self.J = self.J*(1/np.sum(self.J, axis=0))
        self.eta_invs = np.ones(self.J.shape[0])*0.001
        self.B_pos = np.zeros(self.num_states)
        self.B_neg = np.zeros(self.num_states)

        # Detached Hebbian tensors
        self.J = torch.tensor(self.J).float()
        self.B_pos = torch.tensor(self.B_pos).float()
        self.B_neg = torch.tensor(self.B_neg).float()
        self.J.detach_()
        self.B_pos.detach_()
        self.B_neg.detach_()

        self.prev_input = None
        self.curr_input = None

        self.real_T_tilde = 0.001*self.J.t().clone().numpy()
        self.real_T_count = 0.001*np.ones(self.num_states)

    def _update_J_ij(self, i, j):
        if self.prev_input[j] != 1: return # TODO: idealized
        eta = 1/self.eta_invs[j]
        decay = 1 - eta
        B_pos_j = torch.unsqueeze(self.B_pos[j], dim=0)
        B_neg_i = torch.unsqueeze(self.B_neg[i], dim=0)
        if i == j:
            activity = self._update_activity_clamp(self.X[i])
            B_pos = self.update_B_clamp(B_pos_j)
            potentiation = (self.dt/self.tau_J)*activity*B_pos
            update = potentiation*self.alpha_self
            self.J[i,j] = self._forward_activity_clamp(
                decay*self.J[i,j] + eta*update
                )

            # DEBUG
            if self.debug:
                self.last_update[i,j] += update
            if self.debug_print and self.curr_input[i] == 1:
                str1 = f'Correct self {i}: {update.item():.2f} '
                str2 = f'with i activity {self.X[i].item():.2f} '
                str3 = f'and j activity {self.X[j].item():.2f} '
                str4 = f'and j plasticity {B_pos_j.item():.2f}'
                print(str1 + str2 + str3 + str4)
        else:
            activity_i = self._update_activity_clamp(self.X[i])
            activity_j = self._update_activity_clamp(self.X[j])
            B_pos = self.update_B_clamp(B_pos_j)
            B_neg = self.update_B_clamp(B_neg_i)
            potentiation = (self.dt/self.tau_J) * activity_i * B_pos
            depression = (self.dt/self.tau_J) * activity_j * B_neg
            update = (potentiation + depression)*self.alpha_other
            self.J[i,j] = self._forward_activity_clamp(
                decay*self.J[i,j] + eta*update
                )

            # DEBUG
            if self.debug:
                self.last_update[i,j] += eta*update
            if self.debug_print and self.curr_input[i] == 1:
                str1 = f'Correct {j} to {i}: {update.item():.2f} '
                str2 = f'with i activity {self.X[i].item():.2f} '
                str3 = f'and j activity {self.X[j].item():.2f} '
                str4 = f'and j plasticity {B_pos_j.item():.2f}'
                print(str1 + str2 + str3 + str4)
            if self.debug_print and self.curr_input[i] != 1 and update > 0:
                str1 = f'Wrong totally, {j} to {i}: {update.item():.2f} '
                str2 = f'with i activity {self.X[i].item():.2f} '
                str3 = f'and j activity {self.X[j].item():.2f} '
                str4 = f'and j plasticity {B_pos_j.item():.2f}'
                print(str1 + str2 + str3 + str4)

    def _update_B_pos(self):
        learning_rate = self.tau_pos #self.dt/self.tau_pos .1/.12
        decay = 1 - learning_rate
        activity = self._update_activity_clamp(self.X)
        self.B_pos = decay*self.B_pos + learning_rate*self.A_pos*activity
        self.B_pos = self._update_B_ceiling(self.B_pos)

        # DEBUG
        if self.debug:
            self.allBpos[:, self.allX_t] = self.B_pos

    def _update_B_neg(self):
        learning_rate = self.dt/self.tau_neg
        decay = 1 - learning_rate
        activity = self._update_activity_clamp(self.X)
        self.B_neg = decay*self.B_neg + learning_rate*self.A_neg*activity
        self.B_neg = self._update_B_ceiling(self.B_neg)

        # DEBUG
        if self.debug:
            self.allBneg[:, self.allX_t] = self.B_neg

    def _update_eta_invs(self):
        self.eta_invs = self.prev_input + self.gamma_T*self.eta_invs

    def _decay_all_eta_invs(self):
        self.eta_invs = self.gamma_T*self.eta_invs

    def _forward_activity_clamp(self, x):
        _ceil = self._ceil #1.2
        _floor = 0
        ceil_x = -1*(
            nn.functional.leaky_relu(-x + _ceil, negative_slope=leaky_slope) - _ceil
            )
        #floor_x = nn.functional.leaky_relu(ceil_x, negative_slope=leaky_slope)
        return ceil_x #floor_x
#        clamp_x = nn.functional.sigmoid(
#            x*self.fac_scale_1 + self.fac_bias_1
#            )
#        return self.fac_scale_2*clamp_x

    def _update_activity_clamp(self, x):
        return nn.functional.leaky_relu(x-0.99, negative_slope=leaky_slope)*100
#        clamp_x = nn.functional.sigmoid(
#            x*self.uac_scale_1 + self.uac_bias_1
#            )
#        return self.uac_scale_2*clamp_x

    def _update_B_ceiling(self, x):
        _ceil = 6
        ceil_x = -1*(
            nn.functional.leaky_relu(-x + _ceil, negative_slope=leaky_slope) - _ceil
            )
        return ceil_x
#        clamp_x = nn.functional.sigmoid(
#            x*self.ubc_scale_1 + self.ubc_bias_1
#            )
#        return self.ubc_scale_2*clamp_x

    def update_B_clamp(self, x):
        return x
        #return torch.clamp(x, min=0)

    def _init_trainable(self):
        self.A_pos = 7 #nn.Parameter(torch.ones(1)*14) # 7
        self.tau_pos = nn.Parameter(torch.ones(1)*(0.1/0.6)) # .12
        self.A_neg = 0 #nn.Parameter(torch.randn(1)) # 0
        self.tau_neg = 1 #nn.Parameter(torch.randn(1)) # 1
        self.dt = 0.1 #nn.Parameter(torch.randn(1)*0.4) # 0.1
        self.tau_J = 1 #nn.Parameter(torch.randn(1)) # 1
        self.alpha_self = 1.65 #nn.Parameter(torch.ones(1)) # 1.65
        self.alpha_other = 10.08 #nn.Parameter(torch.ones(1)*8) # 10.08
        self._ceil = nn.Parameter(torch.ones(1))
#        self.tau_pos.register_hook(print)

        # Forward activity clamp
#        self.fac_scale_1 = nn.Parameter(torch.randn(1))
#        self.fac_bias_1 = nn.Parameter(torch.randn(1))
#        self.fac_scale_2 = nn.Parameter(torch.randn(1))

        # Update activity clamp
#        self.uac_scale_1 = nn.Parameter(torch.randn(1))
#        self.uac_bias_1 = nn.Parameter(torch.randn(1))
#        self.uac_scale_2 = nn.Parameter(torch.randn(1))

        # Update B ceiling
#        self.ubc_scale_1 = nn.Parameter(torch.randn(1))
#        self.ubc_bias_1 = nn.Parameter(torch.randn(1))
#        self.ubc_scale_2 = nn.Parameter(torch.randn(1))

        # Update B clamp
#        self.update_B_clamp = nn.Sequential(
#            nn.Linear(1, 1),
#            nn.Sigmoid(),
#            nn.Linear(1, 1, bias=False)
#0.1 #            )

