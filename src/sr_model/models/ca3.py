import numpy as np
import torch
import torch.nn as nn
from sr_model.models import module
from sr_model.utils import get_sr
from sr_model.utils_modules import LeakyClamp, LeakyThreshold

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
            self, num_states, gamma_M0, gamma_T=0.99, leaky_slope=1e-4,
            debug=False, debug_print=False
            ):

        super(STDP_CA3, self).__init__()
        self.num_states = num_states
        self.gamma_T = gamma_T
        self.gamma_M0 = gamma_M0
        self.leaky_slope = leaky_slope

        self.reset()
        self._init_constants()
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

    def set_num_states(self, num_states):
        self.num_states = num_states

    def forward(self, input, update_transition=True):
        """
        Returns activity of network given input. 

        Args:
            input: (batch, states)
        """

        input = torch.squeeze(input) # TODO: batching
        M_hat = self.get_M_hat()
        activity = torch.matmul(M_hat.t(), input)
        activity = self.forward_activity_clamp(activity, self.leaky_slope)
        if update_transition:
            self.prev_input = self.curr_input
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
                self.allTs.append(self.get_T().clone())

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

    def get_stdp_kernel(self, kernel_len=15):
        """ Returns plasticity kernel for plotting or debugging. """

        k = np.zeros(kernel_len)
        half_len = kernel_len//2
        scaling = self.A_scaling
        k[:half_len] = scaling*self.A_neg.data.item() * np.exp(
            np.arange(-half_len, 0)/self.tau_neg.data.item()
            )
        k[-half_len-1:] = scaling*self.A_pos.data.item() * np.exp(
            -1*np.arange(half_len+1)/self.tau_pos.data.item()
            )
        return k

    def get_M_hat(self, gamma=None):
        """ Inverts the learned T matrix to get putative M matrix """

        if gamma is None:
            gamma = self.gamma_M0
        T = self.get_T()
        try:
            M_hat = torch.linalg.pinv(torch.eye(T.shape[0]) - gamma*T)
        except:
            print("Pseudo-Inverse did not converge.")
            M_hat = torch.linalg.pinv(
                torch.eye(T.shape[0]) - gamma*T + torch.eye(T.shape[0])*1e-1
                )
        return M_hat

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
        #if self.prev_input[j] != 1: return # TODO: idealized
        eta = 1/self.eta_invs[j]
        decay = (self.eta_invs[j] - self.prev_input[j])/self.eta_invs[j]
        B_pos_j = torch.unsqueeze(self.B_pos[j], dim=0)
        B_neg_i = torch.unsqueeze(self.B_neg[i], dim=0)
        
        if i == j:
            activity = self.update_activity_clamp(self.X[i], self.leaky_slope)
            potentiation = (self.dt/self.tau_J)*activity*B_pos_j*self.prev_input[j]
            update = self.update_clamp(
                potentiation*self.alpha_self*self.alpha_self_scaling,
                self.leaky_slope
                )
            self.J[i,j] = self.J_weight_clamp(
                decay*self.J[i,j] + eta*update, self.leaky_slope
                )

            # DEBUG
            if self.debug:
                self.last_update[i,j] += update
            if self.debug_print and self.curr_input[i]==1 and self.prev_input[j]==1:
                str1 = f'Correct self {i}: {update.item():.2f} '
                str2 = f'with i activity {self.X[i].item():.2f} '
                str3 = f'and j activity {self.X[j].item():.2f} '
                str4 = f'and j plasticity {B_pos_j.item():.2f}'
                print(str1 + str2 + str3 + str4)
            if self.debug_print and (self.curr_input[i]!=1 or self.prev_input[j]!=1) and update > 1e-4:
                str1 = f'Incorrect self {i}: {update.item():.2f} '
                str2 = f'with i activity {self.X[i].item():.2f} '
                str3 = f'and j activity {self.X[j].item():.2f} '
                str4 = f'and j plasticity {B_pos_j.item():.2f}'
                print(str1 + str2 + str3 + str4)
        else:
            activity_i = self.update_activity_clamp(self.X[i], self.leaky_slope)
            activity_j = self.update_activity_clamp(self.X[j], self.leaky_slope)
            potentiation = (self.dt/self.tau_J) * activity_i * B_pos_j
            depression = (self.dt/self.tau_J) * activity_j * B_neg_i
            update = (potentiation + depression)*self.alpha_other*self.alpha_other_scaling
            if update < 0:
                update = -1 *self.update_clamp(-update, self.leaky_slope)
            else:
                update = self.update_clamp(update, self.leaky_slope)
            self.J[i,j] = self.J_weight_clamp(
                decay*self.J[i,j] + eta*update, self.leaky_slope
                )

            # DEBUG
            if self.debug:
                self.last_update[i,j] += eta*update
            if self.debug_print and self.curr_input[i]==1 and self.prev_input[j]==1:
                str1 = f'Correct {j} to {i}: {update.item():.2f} '
                str2 = f'with i activity {self.X[i].item():.2f} '
                str3 = f'and j activity {self.X[j].item():.2f} '
                str4 = f'and j plasticity {B_pos_j.item():.2f}'
                print(str1 + str2 + str3 + str4)
            if self.debug_print and (self.curr_input[i] != 1 or self.prev_input[j]!=1) and update > 1e-4:
                str1 = f'Wrong totally, {j} to {i}: {update.item():.2f} '
                str2 = f'with i activity {self.X[i].item():.2f} '
                str3 = f'and j activity {self.X[j].item():.2f} '
                str4 = f'and j plasticity {B_pos_j.item():.2f}'
                print(str1 + str2 + str3 + str4)

    def _update_B_pos(self):
        tau_pos = nn.functional.leaky_relu(
            self.tau_pos, negative_slope=self.leaky_slope
            )
        decay = 1 - 1/tau_pos
        activity = self.update_activity_clamp(self.X, self.leaky_slope)
        A = self.A_pos * self.A_scaling
        self.B_pos = decay*self.B_pos + A*activity
        self.B_pos = self.B_integration_clamp(self.B_pos, self.leaky_slope)

        if self.debug:
            self.allBpos[:, self.allX_t] = self.B_pos

    def _update_B_neg(self):
        tau_neg = nn.functional.leaky_relu(
            self.tau_neg, negative_slope=self.leaky_slope
            )
        decay = 1 - 1/tau_neg
        activity = self.update_activity_clamp(self.X, self.leaky_slope)
        A = self.A_neg * self.A_scaling
        self.B_neg = decay*self.B_neg + A*activity
        self.B_neg = self.B_integration_clamp(self.B_neg, self.leaky_slope)

        if self.debug:
            self.allBneg[:, self.allX_t] = self.B_neg

    def _update_eta_invs(self):
        self.eta_invs = self.prev_input + self.gamma_T*self.eta_invs

    def _decay_all_eta_invs(self):
        self.eta_invs = self.gamma_T*self.eta_invs

    def _init_constants(self):
        self.dt = 1
        self.tau_J = 1
        self.A_scaling = 10 # Scale factors are useful for gradient descent
        self.alpha_other_scaling = 10
        self.alpha_self_scaling = 1

        self.learning_rate_clamp = LeakyClamp(floor=0, ceil=1)
        self.forward_activity_clamp = LeakyClamp(floor=None, ceil=1)

        self.J_weight_clamp = LeakyClamp(floor=None, ceil=None) # Not needed
        self.B_integration_clamp = LeakyClamp(floor=None, ceil=None) # Not needed

    def _init_trainable(self):
        self.A_pos = nn.Parameter(torch.rand(1))
        self.tau_pos = nn.Parameter(torch.rand(1)*2)
        self.A_neg = nn.Parameter(-torch.abs(torch.rand(1)))
        nn.init.constant_(self.A_neg, -0.)
        self.tau_neg = nn.Parameter(torch.abs(torch.rand(1)))
        self.alpha_self = nn.Parameter(torch.abs(torch.randn(1)))
        self.alpha_other = nn.Parameter(torch.abs(torch.randn(1)))

        self.update_clamp = LeakyThreshold(
            x0=nn.Parameter(torch.abs(torch.rand(1)/2)),
            x1=1, floor=0, ceil=1.
            )
        self.update_activity_clamp = LeakyThreshold(
            x0=nn.Parameter(torch.abs(torch.rand(1))),
            x1=1, floor=0, ceil=None
            )

        self.reset_trainable_ideal()

    def reset_trainable_ideal(self, requires_grad=True):
        nn.init.constant_(self.A_pos, 0.5) #0.69)
        nn.init.constant_(self.tau_pos, 1.15) #0.8742)
        nn.init.constant_(self.A_neg, -0)
        nn.init.constant_(self.tau_neg, 1.)
        nn.init.constant_(self.alpha_other, .18) #0.82)
        nn.init.constant_(self.alpha_self, 0.2) #1)
        self.update_clamp = LeakyThreshold(
            x0=nn.Parameter(torch.ones(1)*.2),
            x1=1, floor=0, ceil=1.
            )
        self.update_activity_clamp = LeakyThreshold(
            x0=nn.Parameter(torch.ones(1)*0.7),
            x1=1, floor=0, ceil=None
            )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def set_differentiability(self, differentiable):
        self.differentiable = differentiable
        if differentiable == True:
            self.leaky_slope = 1e-5
        else:
            self.leaky_slope = 0

