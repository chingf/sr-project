import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu, sigmoid, tanh
from sr_model.models import module
from sr_model.utils import get_sr
from sr_model.utils_modules import Clamp, LeakyThreshold, TwoSidedLeakyThreshold

posinf = 1E30

class CA3(module.Module):
    def __init__(
        self, num_states, gamma_M0, gamma_T=1., use_dynamic_lr=False, lr=1E-3,
        parameterize=False, rollout=None, forget=None, T_grad_on=False,
        output_params={}
        ):

        super(CA3, self).__init__()
        self.num_states = num_states
        self.gamma_M0 = gamma_M0
        self.gamma_T = gamma_T
        self.use_dynamic_lr = use_dynamic_lr
        self.lr = lr
        self.parameterize = parameterize
        self.rollout = rollout
        self.forget = forget # for modifying forget term in learning rule
        self.T_grad_on = T_grad_on
        self.add_perturb = T_grad_on
        self.output_params = {
            'num_iterations': np.inf, 'nonlinearity': None,
            'nonlinearity_args': None # So far, just [floor, ceil] for clamp
            }
        self.output_params.update(output_params)
        self._init_trainable()
        self.reset()
    
    def forward(self, _input, update_transition=True, gamma=None):
        if gamma is None:
            gamma = self.gamma_M0
        num_iterations = self.output_params['num_iterations']
        nonlinearity = self.output_params['nonlinearity']
        if update_transition:
            if self.curr_input is not None:
                self.prev_input = torch.squeeze(self.curr_input)
            self.curr_input = torch.squeeze(_input)

        if np.isinf(num_iterations):
            M = self.get_M_hat(gamma)
            output = torch.matmul(M.t(), torch.squeeze(_input))
        else:
            output = torch.squeeze(torch.zeros_like(_input))
            dt = 1.
            for iteration in range(num_iterations):
                #current = torch.matmul(self.T.t(), output.t())

                current = output.t()

                # Option: apply nonlinearity onto current
                if nonlinearity == 'sigmoid':
                    nonlin_a = torch.absolute(self.nonlin_a)
                    current = sigmoid(nonlin_a*current + self.nonlin_b)
                elif nonlinearity == 'tanh':
                    current = tanh(current)
                elif nonlinearity == 'clamp':
                    clamp_offset = torch.absolute(self.clamp_offset)
                    clamp_min = self.clamp_min
                    clamp_max = self.clamp_min + clamp_offset
                    current = torch.clamp(
                        current, min=clamp_min.item(),
                        max=clamp_max.item()
                        )

                current = torch.matmul(self.T.t(), current)

                # Apply neuromodulatory gamma
                current = gamma*current

                # Provide input current
                current = current + torch.squeeze(_input)

                # Iterate output
                dxdt = -output + current
                output = output + dt*dxdt
                output = relu(output)
                output = torch.nan_to_num(output, posinf=posinf)
        return output

    def update(self, update_type=None, view_only=False):
        if update_type is None:
            update_type = self.forget
        return self._update(update_type, view_only)

    def _update(self, update_type, view_only):
        if self.prev_input is None:
            return

        if update_type is None:
            forget_term = torch.outer(self.prev_input, self.prev_input)@self.T
        elif update_type == "oja":
            forget_term = torch.outer(
                torch.square(self.prev_input), torch.ones_like(self.prev_input)
                ) * self.T
        else:
            raise ValueError("Invalid forget parameter.")
        update_term = torch.outer(self.prev_input, self.curr_input)
        if self.use_dynamic_lr:
            new_state_counts = self.prev_input + self.gamma_T*self.state_counts
            lr_update = 1./new_state_counts
            lr_update = torch.clamp(
                torch.nan_to_num(lr_update), 0, torch.abs(self.lr_ceil)
                )
            lr_decay = lr_update
            full_update = forget_term * lr_decay[:,None] + update_term * lr_update[:,None]
            self.state_counts = new_state_counts
        else:
            if self.parameterize:
                lr_update = torch.abs(self.lr_update)*0.01
                lr_decay = torch.abs(self.lr_decay)*0.01
                full_update = lr_update*update_term - lr_decay*forget_term
            else:
                full_update = torch.abs(self.lr)*(update_term - forget_term)

        if view_only:
            return update_term, forget_term

        self.T = self.T + full_update
        self.T[:] = torch.clamp(self.T, min=0)
        return update_term, forget_term

    def get_T(self):
        return self.T

    def reset(self):
        self.prev_input = None
        self.curr_input = None
        self.state_count_init = 1E-3
        self.lr_ceil = self.lr
        self.state_counts = 10.*self.state_count_init*torch.ones(self.num_states)
        if self.T_grad_on:
            self.T = torch.nn.Parameter(
                torch.zeros(self.num_states, self.num_states)
                )
        else:
            self.T = torch.zeros(self.num_states, self.num_states)

    def get_M_hat(self, gamma=None):
        """ Inverts the learned T matrix to get putative M matrix """

        if gamma is None:
            gamma = self.gamma_M0
        T = self.get_T()
        if self.rollout == None:
            try:
                if self.add_perturb:
                    perturb = torch.diag(torch.diagonal((1E-3)*torch.rand(T.shape)))
                else:
                    perturb = 0
                M_hat = torch.linalg.pinv(
                    perturb + torch.eye(T.shape[0]) - gamma*T)
            except:
                print('SVD did not converge. Small values added on diagonal.')
                M_hat = torch.linalg.pinv(
                    1.001*torch.eye(T.shape[0]) - gamma*T)
        else:
            M_hat = torch.zeros_like(T)
            scale_term = gamma
            for t in range(self.rollout):
                M_hat = M_hat + scale_term**t * torch.matrix_power(T.clone(), t)
        return M_hat

    def set_num_states(self, num_states):
        self.num_states = num_states

    def _init_trainable(self):
        if self.parameterize:
            if self.output_params['nonlinearity'] is None:
                raise ValueError('Please specify nonlinearity')
            self.lr_update = nn.Parameter(torch.tensor([1E-3]))
            self.lr_decay = nn.Parameter(torch.tensor([1E-3]))

            if self.output_params['nonlinearity'] == 'clamp':
                self.clamp_min = nn.Parameter(torch.tensor([0.]))
                self.clamp_offset = nn.Parameter(torch.empty(1))
                nn.init.uniform_(self.clamp_offset, 1.0, 4.0)
            elif self.output_params['nonlinearity'] == 'sigmoid' or \
                    self.output_params['nonlinearity'] == 'tanh':
                self.nonlin_a = nn.Parameter(torch.tensor([1.]))
                self.nonlin_b = nn.Parameter(torch.tensor([0.]))
        else:
            self.lr = torch.tensor([self.lr])
            # Nonlinearity may be a clamp
            if self.output_params['nonlinearity'] == 'clamp':
                if self.output_params['nonlinearity_args'] is not None:
                    _min, _offset = self.output_params['nonlinearity_args']
                    self.clamp_min = torch.tensor([_min])
                    self.clamp_offset = torch.tensor([_offset])
            elif self.output_params['nonlinearity'] == 'sigmoid' or \
                    self.output_params['nonlinearity'] == 'tanh':
                self.nonlin_a = torch.tensor([1.])
                self.nonlin_b = torch.tensor([0.])

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
            self, num_states, gamma_M0, gamma_T=0.99999,
            A_pos_sign=None, A_neg_sign=None,
            approx_B=False, use_kernels_in_update=False,
            output_params={}, static_eta=None, use_B_norm=False
            ):

        super(STDP_CA3, self).__init__()
        self.num_states = num_states
        self.gamma_T = gamma_T
        self.gamma_M0 = gamma_M0
        self.A_pos_sign = A_pos_sign
        self.A_neg_sign = A_neg_sign
        self.approx_B = approx_B
        self.use_kernels_in_update = use_kernels_in_update
        self.output_params = {
            'num_iterations': np.inf, 'input_clamp': np.inf,
            'nonlinearity': None, 'nonlinearity_args': None,
            'no_transform': False
            }
        self.output_params.update(output_params)
        self.static_eta = static_eta
        self.use_B_norm = use_B_norm
        self.x_queue = torch.zeros((self.num_states, 1)) # Not used if approx B

        self.reset()
        self._init_constants()
        self._init_trainable()

    def set_num_states(self, num_states):
        self.num_states = num_states
        self.x_queue = torch.zeros((self.num_states, self.x_queue_length))

    def forward(self, _input, update_transition=True, gamma=None):
        """
        Returns activity of network given input. 

        Args:
            _input: (batch, states)
        """

        _input = torch.squeeze(_input) # TODO: batching
        activity = self.get_recurrent_output(_input, gamma)

        _activity = self.plasticity_activity_clamp(activity)

        if update_transition:
            self.prev_input = self.curr_input
            self.curr_input = _input
            self.X = _activity
            self.x_queue[:, :-1] = self.x_queue[:, 1:]
            self.x_queue[:, -1] = _activity
        return activity

    def get_recurrent_output(self, _input, gamma):
        if gamma is None: gamma = self.gamma_M0
        num_iterations = self.output_params['num_iterations']
        input_clamp = self.output_params['input_clamp']
        nonlinearity = self.output_params['nonlinearity']
        nonlinearity_args = self.output_params['nonlinearity_args']

        if np.isinf(num_iterations):
            M_hat = self.get_M_hat(gamma=gamma)
            activity = torch.matmul(M_hat.t(), _input)
        else:
            activity = torch.zeros_like(self.x_queue[:, -1]) #TODO: without zero?
            dt = 1.
            for iteration in range(num_iterations):
                #current = torch.matmul(gamma*self.J, activity)
                current = activity

                # Option: apply nonlinearity onto current
                if nonlinearity == 'tanh':
                    if type(nonlinearity_args) is tuple:
                        x_scale, y_scale = nonlinearity_args
                    elif nonlinearity_args is None:
                        x_scale = y_scale = 1
                    else:
                        x_scale = y_scale = nonlinearity_args
                    current = tanh(current/x_scale)*y_scale

                current = torch.matmul(gamma*self.J, current)

                # Option: provide input only briefly
                if iteration <= input_clamp:
                    current = current + _input

                # Iterate activity
                dxdt = -activity + current
                activity = activity + dt*dxdt
                activity = torch.nan_to_num(activity) # for training

        return activity

    def update(self):
        """ Plasticity update """

        if self.use_kernels_in_update:
            self._update_B_pos()
            self._update_B_neg()
            if self.use_B_norm:
                self._update_B_norm()

        if self.prev_input is None: return
        
        self._update_eta_invs()

        self._update_J()

        self.real_T_tilde = self.gamma_T*self.real_T_tilde
        self.real_T_tilde += np.outer(self.prev_input, self.curr_input)
        self.real_T_count = self.gamma_T*self.real_T_count
        self.real_T_count[self.prev_input>0] += 1

    def get_stdp_kernel(self, kernel_len=20, scale=True):
        """ Returns plasticity kernel for plotting or debugging. """

        if self.A_pos_sign is not None:
            A_pos = self.A_pos_sign * torch.abs(self.A_pos)
        else:
            A_pos = self.A_pos
        if self.A_neg_sign is not None:
            A_neg = self.A_neg_sign * torch.abs(self.A_neg)
        else:
            A_neg = self.A_neg
       
        pos_xs = -1*np.arange(0, kernel_len+0.01, 0.5)
        pos_ys = np.exp(
            pos_xs/self.tau_pos.data.item()
            )
        if scale:
            pos_ys *= A_pos.data.item()
        else:
            pos_ys *= np.sign(A_pos.data.item())
        neg_xs = np.arange(-kernel_len, 0.01, 0.5)
        neg_ys = np.exp(
            neg_xs/self.tau_neg.data.item()
            )
        if scale:
            neg_ys *= A_neg.data.item()
        else:
            neg_ys *= np.sign(A_neg.data.item())
        xs = np.concatenate((neg_xs, -1*pos_xs))
        ys = np.concatenate((neg_ys, pos_ys))
        return xs, ys

    def get_M_hat(self, gamma=None):
        """ Inverts the learned T matrix to get putative M matrix """

        if gamma is None:
            gamma = self.gamma_M0
        T = self.get_T()
        num_iterations = self.output_params['num_iterations']

        if np.isinf(num_iterations):
            M_hat = torch.linalg.pinv(torch.eye(T.shape[0]) - gamma*T)
        else:
            M_hat = []
            for i in range(self.num_states):
                dg_input = torch.zeros(1, self.num_states)
                dg_input[0,i] = 1.
                with torch.no_grad():
                    out = self.forward(
                        dg_input, update_transition=False, gamma=gamma
                        )
                M_hat.append(out.detach().numpy().squeeze())
            M_hat = np.array(M_hat)
        return M_hat

    def get_T(self):
        """ Returns the learned T matrix, where T = J^T. """

        return self.J.t()

    def reset(self):
        init_scale = 0.0001
        self.J = np.zeros((self.num_states, self.num_states))
        self.eta_invs = np.ones(self.J.shape[0])*init_scale
        self.B_pos = np.zeros(self.num_states)
        self.B_neg = np.zeros(self.num_states)
        self.B_norm = np.zeros(self.num_states)

        # Detached Hebbian tensors
        self.J = torch.tensor(self.J).float()
        self.eta_invs = torch.tensor(self.eta_invs).float()
        self.B_pos = torch.tensor(self.B_pos).float()
        self.B_neg = torch.tensor(self.B_neg).float()
        self.B_norm = torch.tensor(self.B_norm).float()
        self.J.detach_()
        self.B_pos.detach_()
        self.B_neg.detach_()
        self.B_norm.detach_()

        self.prev_input = None
        self.curr_input = None

        self.real_T_tilde = init_scale*self.J.t().clone().numpy()
        self.real_T_count = init_scale*np.ones(self.num_states)

        nn.init.constant_(self.x_queue, 0)

    def _update_J(self):
        # Format learning rates for each neuron's outgoing synapses (shared by js)
        num_states = self.num_states
        if self.static_eta is None: # adaptive learning rate
            etas = torch.nan_to_num(1/self.eta_invs, posinf=posinf)
            etas = self.learning_rate_clamp(etas)
            etas = torch.tile(etas, (num_states, 1)) # (N, N)
            etas = torch.clamp(etas, min=1E-3)
        else:
            etas = self.static_eta

        if self.use_kernels_in_update:
            # Get activity and plasticity integration
            B_pos = torch.unsqueeze(self.B_pos, dim=0) # (1, N)
            B_neg = torch.unsqueeze(self.B_neg, dim=0) # (1, N)
            activity = self.X # (N,)
            activity = torch.unsqueeze(activity, dim=0) # (1, N)
    
            # Calculate potentiation and depression over all i,j for i != j
            potentiation = activity.t() * B_pos # (N, N); activity i, B_pos j
            depression = B_neg.t() * activity # (N, N); B_neg i, activity j
            alpha_other = torch.abs(self.alpha_other)
            update = (potentiation + depression)*alpha_other*alpha_other
    
            # Calculate self-potentiation (i == j)
            self_potentiation = activity*B_pos*self.prev_input
            alpha_self = torch.abs(self.alpha_self)
            self_update = self_potentiation*alpha_self
            diag_mask = torch.ones(num_states, num_states) - torch.eye(num_states)
            update = update*diag_mask + torch.diag(torch.squeeze(self_update))
            update = update - self.B_norm
            update = torch.nan_to_num(update)
        else:
            update = torch.outer(self.x_queue[:,-1], self.x_queue[:,-2])
            x = self.x_queue[:, -2]
            update = update - torch.outer(torch.matmul(self.J, x), x)

        self.J = self.J + etas*(update)
        self.J = torch.nan_to_num(self.J)

    def _update_B_pos(self):
        # Calculate scaling factor
        if self.A_pos_sign is not None:
            A_pos = self.A_pos_sign * torch.abs(self.A_pos)
        else:
            A_pos = self.A_pos
        A = A_pos
        tau_pos = relu(self.tau_pos)

        # Either use Euler integration or calculate convolution
        if self.approx_B:
            decay = 1 - 1/tau_pos
            activity = self.X
            self.B_pos = decay*self.B_pos + A*activity
        else:
            exp_function = torch.exp(torch.arange(-self.x_queue_length, 0)/tau_pos)
            exp_function = torch.nan_to_num(exp_function, posinf=posinf) # for training
            activity = self.x_queue
            convolution = activity @ exp_function
            self.B_pos = A*convolution

    def _update_B_neg(self):
        # Calculate scaling factor 
        if self.A_neg_sign is not None:
            A_neg = self.A_neg_sign * torch.abs(self.A_neg)
        else:
            A_neg = self.A_neg
        A = A_neg
        tau_neg = relu(self.tau_neg)

        # Either use Euler integration or calculate convolution
        if self.approx_B:
            decay = 1 - 1/tau_neg
            activity = self.X
            self.B_neg = decay*self.B_neg + A*activity
        else:
            exp_function = torch.exp(torch.arange(-self.x_queue_length, 0)/tau_neg)
            exp_function = torch.nan_to_num(exp_function, posinf=posinf) # for training
            activity = self.x_queue
            convolution = activity @ exp_function
            self.B_neg = A*convolution

    def _update_B_norm(self):
        # Calculate scaling factor 
        A = torch.abs(self.A_norm)
        tau_norm = relu(self.tau_norm)

        # Use Euler integration
        decay = 1 - 1/tau_norm
        x = self.X
        self.B_norm = decay*self.B_norm + A*torch.outer(torch.matmul(self.J, x), x)

    def _update_eta_invs(self):
        self.eta_invs = self.X*self.eta_inv_scale + self.gamma_T*self.eta_invs

    def _init_constants(self):
        self.learning_rate_clamp = Clamp(floor=0, ceil=1)
        self.plasticity_activity_clamp = Clamp( # For stability during training
            floor=-10, ceil=10
            )
        self.x_queue_length = 20 # max needed for tau of 4
        self.x_queue = torch.zeros((self.num_states, self.x_queue_length))

    def _init_trainable(self):
        if self.use_kernels_in_update:
            self.A_pos = nn.Parameter(torch.rand(1))
            self.tau_pos = nn.Parameter(1 + torch.randn(1)*0.1)
            self.A_neg = nn.Parameter(-torch.abs(torch.rand(1)))
            self.tau_neg = nn.Parameter(1 + torch.randn(1)*0.1)
            self.alpha_self = nn.Parameter(torch.abs(torch.randn(1)))
            self.alpha_other = nn.Parameter(torch.abs(torch.randn(1)))
            self.eta_inv_scale = nn.Parameter(torch.tensor([1.]))
            if self.use_B_norm:
                self.A_norm = nn.Parameter(-torch.abs(torch.rand(1)))
                self.tau_norm = nn.Parameter(1 + torch.randn(1)*0.1)
        else:
            self.eta_inv_scale = 1

