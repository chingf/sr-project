import numpy as np
import torch
import torch.nn as nn
from sr_model.models import module
from sr_model.utils import get_sr
from sr_model.utils_modules import LeakyClamp, LeakyThreshold, TwoSidedLeakyThreshold

class CA3(module.Module):
    def __init__(self, num_states, gamma_M0, gamma_T=1.):
        super(CA3, self).__init__()
        self.T_tilde = torch.clamp(torch.rand(num_states, num_states), 0, 1)
        self.T_tilde = 0.001*self.T_tilde*(1/torch.sum(self.T_tilde, dim=0))
        self.state_counts = 0.001*torch.ones(num_states)
        self.num_states = num_states
        self.gamma_M0 = gamma_M0
        self.gamma_T = gamma_T
        self.prev_input = None
        self.curr_input = None
    
    def forward(self, input, update_transition=True):
        M = get_sr(self.get_T(), self.gamma_M0)
        if update_transition:
            if self.curr_input is not None:
                self.prev_input = torch.squeeze(self.curr_input)
            self.curr_input = torch.squeeze(input)
        return torch.matmul(M, input.T)

    def update(self):
        if self.prev_input is None:
            return
        self.T_tilde[self.prev_input>0, self.curr_input>0] += 1
        self.state_counts = self.prev_input + self.gamma_T*self.state_counts

    def get_T(self):
        return self.T_tilde/self.state_counts[:,None]

    def reset(self):
        self.prev_input = None
        self.curr_input = None
        self.T_tilde = torch.clamp(torch.rand(self.num_states, self.num_states), 0, 1)
        self.T_tilde = 0.001*self.T_tilde*(1/torch.sum(self.T_tilde, dim=0))
        self.state_counts = 0.001*torch.ones(self.num_states)

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
            A_pos_sign=None, A_neg_sign=None, approx_B=False
            ):

        super(STDP_CA3, self).__init__()
        self.num_states = num_states
        self.gamma_T = gamma_T
        self.gamma_M0 = gamma_M0
        self.leaky_slope = leaky_slope
        self.A_pos_sign = A_pos_sign
        self.A_neg_sign = A_neg_sign
        self.approx_B = approx_B
        self.x_queue = torch.zeros((self.num_states, 1)) # Not used if approx B

        self.reset()
        self._init_constants()
        self._init_trainable()

        self.curr_input = None
        self.prev_input = None

    def set_num_states(self, num_states):
        self.num_states = num_states
        self.x_queue = torch.zeros((self.num_states, self.x_queue_length))

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
        self.x_queue[:, :-1] = self.x_queue[:, 1:]
        self.x_queue[:, -1] = activity
        return activity

    def update(self):
        """ Plasticity update """

        self._update_B_pos()
        self._update_B_neg()

        if self.prev_input is None: return

        self._update_eta_invs()

        self._update_J()

        self.real_T_tilde = self.gamma_T*self.real_T_tilde
        self.real_T_tilde[self.prev_input>0, self.curr_input>0] += 1
        self.real_T_count = self.gamma_T*self.real_T_count
        self.real_T_count[self.prev_input>0] += 1

    def get_stdp_kernel(self, kernel_len=15):
        """ Returns plasticity kernel for plotting or debugging. """

        k = np.zeros(kernel_len)
        half_len = kernel_len//2
        scaling = self.A_scaling
        if self.A_pos_sign is not None:
            A_pos = self.A_pos_sign * torch.abs(self.A_pos)
        else:
            A_pos = self.A_pos
        if self.A_neg_sign is not None:
            A_neg = self.A_neg_sign * torch.abs(self.A_neg)
        else:
            A_neg = self.A_neg
        k[:half_len] = scaling*A_neg.data.item() * np.exp(
            np.arange(-half_len, 0)/self.tau_neg.data.item()
            )
        k[-half_len-1:] = scaling*A_pos.data.item() * np.exp(
            -1*np.arange(half_len+1)/self.tau_pos.data.item()
            )
        return k

    def get_M_hat(self, gamma=None):
        """ Inverts the learned T matrix to get putative M matrix """

        if gamma is None:
            gamma = self.gamma_M0
        T = self.get_T()
        M_hat = torch.linalg.pinv(torch.eye(T.shape[0]) - gamma*T)
        return M_hat

    def get_T(self):
        """ Returns the learned T matrix, where T = J^T. """

        return self.J.t()

    def get_ideal_T_estimate(self):
        """ Returns the ideal T matrix """

        return self.real_T_tilde/self.real_T_count[:,None]

    def set_J_to_T_estimate(self):
        """ Sets J to the correct transition matrix. """

        self.J = torch.tensor(self.get_ideal_T_estimate().T).float()

    def reset(self):
        self.J = np.clip(np.random.rand(self.num_states, self.num_states), 0, 1)
        self.J = self.J*(1/np.sum(self.J, axis=0))
        self.eta_invs = np.ones(self.J.shape[0])*0.001
        self.B_pos = np.zeros(self.num_states)
        self.B_neg = np.zeros(self.num_states)

        # Detached Hebbian tensors
        self.J = torch.tensor(self.J).float()
        self.eta_invs = torch.tensor(self.eta_invs).float()
        self.B_pos = torch.tensor(self.B_pos).float()
        self.B_neg = torch.tensor(self.B_neg).float()
        self.J.detach_()
        self.B_pos.detach_()
        self.B_neg.detach_()

        self.prev_input = None
        self.curr_input = None

        self.real_T_tilde = 0.001*self.J.t().clone().numpy()
        self.real_T_count = 0.001*np.ones(self.num_states)

        nn.init.constant_(self.x_queue, 0)

    def _update_J(self):

        num_states = self.num_states
        # Format learning rates for each neuron's outgoing synapses (shared by js)
        etas = 1/self.eta_invs # (N,)
        etas = torch.tile(etas, (num_states, 1)) # (N, N)
        decays = (self.eta_invs - self.prev_input)/self.eta_invs # (N,)
        decays = torch.tile(decays, (num_states, 1)) # (N, N)

        # Get activity and plasticity integration
        B_pos = torch.unsqueeze(self.B_pos, dim=0) # (1, N)
        B_neg = torch.unsqueeze(self.B_neg, dim=0) # (1, N)
        activity = self.update_activity_clamp(self.X, self.leaky_slope) # (N,)
        activity = torch.unsqueeze(activity, dim=0) # (1, N)

        # Calculate potentiation and depression over all i,j for i != j
        potentiation = (self.dt/self.tau_J) * activity.t() * B_pos # (N, N); activity i, B_pos j
        depression = (self.dt/self.tau_J) * B_neg.t() * activity # (N, N); B_neg i, activity j
        alpha_other = torch.abs(self.alpha_other)*self.alpha_other_scaling
        update = (potentiation + depression)*alpha_other*alpha_other

        # Calculate self-potentiation (i == j)
        self_potentiation = (self.dt/self.tau_J)*activity*B_pos*self.prev_input
        alpha_self = torch.abs(self.alpha_self)*self.alpha_self_scaling
        self_update = self_potentiation*alpha_self
        diag_mask = torch.ones(num_states, num_states) - torch.eye(num_states)
        update = update*diag_mask + torch.diag(torch.squeeze(self_update))

        # Make the update over all N^2 synapses
        update = self.update_clamp(update)
        self.J = self.J_weight_clamp(
            decays*self.J + etas*update, self.leaky_slope
            )

    def _update_B_pos(self):
        # Calculate scaling factor
        if self.A_pos_sign is not None:
            A_pos = self.A_pos_sign * torch.abs(self.A_pos)
        else:
            A_pos = self.A_pos
        A = A_pos * self.A_scaling * self.dt
        tau_pos = nn.functional.leaky_relu(
            self.tau_pos, negative_slope=self.leaky_slope
            )

        # Either use Euler integration or calculate convolution
        if self.approx_B:
            decay = 1 - self.dt/tau_pos
            activity = self.update_activity_clamp(self.X, self.leaky_slope)
            self.B_pos = decay*self.B_pos + A*activity
            self.B_pos = self.B_integration_clamp(self.B_pos, self.leaky_slope)
        else:
            exp_function = torch.exp(torch.arange(-self.x_queue_length, 0)/tau_pos)
            exp_function = torch.nan_to_num(exp_function, posinf=1E10, neginf=1E10)
            convolution = self.update_activity_clamp(self.x_queue, self.leaky_slope) @ exp_function
            self.B_pos = A*convolution

    def _update_B_neg(self):
        # Calculate scaling factor 
        if self.A_neg_sign is not None:
            A_neg = self.A_neg_sign * torch.abs(self.A_neg)
        else:
            A_neg = self.A_neg
        A = A_neg * self.A_scaling * self.dt
        tau_neg = nn.functional.leaky_relu(
            self.tau_neg, negative_slope=self.leaky_slope
            )

        # Either use Euler integration or calculate convolution
        if self.approx_B:
            decay = 1 - self.dt/tau_neg
            activity = self.update_activity_clamp(self.X, self.leaky_slope)
            self.B_neg = decay*self.B_neg + A*activity
            self.B_neg = self.B_integration_clamp(self.B_neg, self.leaky_slope)
        else:
            exp_function = torch.exp(torch.arange(-self.x_queue_length, 0)/tau_neg)
            exp_function = torch.nan_to_num(exp_function, posinf=1E10, neginf=1E10)
            convolution = self.update_activity_clamp(self.x_queue,self.leaky_slope) @ exp_function
            self.B_neg = A*convolution

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

        self.x_queue_length = 20 # max needed for tau of 4
        self.x_queue = torch.zeros((self.num_states, self.x_queue_length))

    def _init_trainable(self):
        self.A_pos = nn.Parameter(torch.rand(1))
        self.tau_pos = nn.Parameter(1 + torch.randn(1)*0.1)
        self.A_neg = nn.Parameter(-torch.abs(torch.rand(1)))
        self.tau_neg = nn.Parameter(1 + torch.randn(1)*0.1)
        self.alpha_self = nn.Parameter(torch.abs(torch.randn(1)))
        self.alpha_other = nn.Parameter(torch.abs(torch.randn(1)))

        self.update_clamp = TwoSidedLeakyThreshold(
            x0=nn.Parameter(torch.abs(torch.rand(1)/2)),
            x1=1, floor=0, ceil=1.
            )
        self.update_activity_clamp = LeakyThreshold(
            x0=nn.Parameter(torch.abs(torch.rand(1))),
            x1=1, floor=0, ceil=None
            )

    def reset_trainable_ideal(self, requires_grad=True):
        nn.init.constant_(self.A_pos, 1) #0.5
        nn.init.constant_(self.tau_pos, 1.4) #1.15
        nn.init.constant_(self.A_neg, 0.15) #0
        nn.init.constant_(self.tau_neg, 1.) #1
        nn.init.constant_(self.alpha_other, .18)
        nn.init.constant_(self.alpha_self, 0.2)
        self.update_clamp = LeakyThreshold(
            x0=nn.Parameter(torch.ones(1)*.1), #0.2
            x1=1, floor=0, ceil=1.
            )
        self.update_activity_clamp = LeakyThreshold(
            x0=nn.Parameter(torch.ones(1)*0.5), #0.7
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

