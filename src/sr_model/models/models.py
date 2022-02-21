import numpy as np
import torch
import torch.nn as nn

from sr_model.models import module, dg, ca3

class AnalyticSR(module.Module):
    """
    Output is M.T @ i
    Args:
        stay_case: 
            0: run the model with no modifications
            1: ignore stay inputs by stitching together transition inputs,
    """

    def __init__(self, num_states, gamma, stay_case=0, ca3_kwargs={}):
        super(AnalyticSR, self).__init__()
        self.dg = dg.DG()
        self.ca3 = ca3.CA3(num_states, gamma, **ca3_kwargs)
        self.num_states = num_states
        self.gamma = gamma
        self.estimates_T = False
        self.stay_case = stay_case

    def forward(
            self, inputs, dg_modes=None, reset=True, update=True,
            gamma=None, update_transition=None
            ):
        """
        Args:
            inputs: (steps, batch, states) one-hot inputs
            dg_modes: (steps, batch, 1) flag of global mode
        """

        num_steps, batch_size, num_states = inputs.shape
        out = []
        if update_transition is None:
            update_transition == update
        if reset:
            self.reset()
        for step in np.arange(num_steps):
            _input = inputs[step, :, :]
            ca3_out = self.ca3(
                _input, update_transition=update_transition, gamma=gamma
                )
            if update:
                self.update()
            out.append(ca3_out)
        out = torch.stack(out)
        return None, out

    def update(self):
        self.ca3.update()

    def get_T(self):
        return self.ca3.get_T()

    def get_M(self, gamma=None):
        return self.ca3.get_M_hat(gamma=gamma)

    def reset(self):
        self.ca3.reset()

    def set_num_states(self, num_states):
        self.ca3.set_num_states(num_states)

class STDP_SR(AnalyticSR):
    """ Output is M.T @ i """

    def __init__(
        self, num_states, gamma, stay_case=0, ca3_kwargs={}
        ):
        """
        Args:
            stay_case: 
                0: run the model with no modifications
                1: ignore stay inputs by stitching together transition inputs,
                2: ignore stay inputs but continue learning rate update during stay
        """

        super(STDP_SR, self).__init__(num_states, gamma, stay_case)
        self.dg = dg.DG()
        self.ca3 = ca3.STDP_CA3(num_states, gamma, **ca3_kwargs) 
        self.estimates_T = True

class Linear(module.Module):
    """ Output is i @ M"""

    def __init__(self, input_size, gamma=0.6, lr=1E-3):
        super(Linear, self).__init__()
        self.M = torch.zeros(1, input_size, input_size)
        self.gamma = gamma
        self.lr = lr
        self.prev_input = None

    def forward(self, inputs, reset=False, update=True):
        """
        inputs is (steps, batch, states)
        """

        if reset:
            self.reset()
        outputs = []
        for input in inputs:
            input = input.unsqueeze(0)
            output = torch.bmm(input, self.M)
            output = output.squeeze(0)
            outputs.append(output)

            if update and (self.prev_input is not None):
                phi = self.prev_input
                psi_s = torch.bmm(phi, self.M)
                psi_s_prime = output
                value_function = psi_s
                expected_value_function = phi + self.gamma*psi_s_prime
                error = expected_value_function - value_function
                self.M[0,:,:] = self.M[0,:,:] + self.lr*error[0]*phi[0].t()
            self.prev_input = input
        outputs = torch.stack(outputs)
        return outputs

    def get_T(self):
        return self.M.clone()

    def get_M(self):
        return self.M.clone()

    def reset(self):
        torch.nn.init.zeros_(self.M)
        self.prev_input = None

    def set_num_states(self, num_states):
        self.M = torch.zeros(1, num_states, num_states)
        self.reset()

class MLP(module.Module):
    """ Output is M @ i """

    def __init__(self, input_size, hidden_size):
        """
        Assumes input size is == output size
        """

        super(MLP, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, input_size)
            )

    def forward(self, inputs, reset=False):
        if reset:
            self.reset()
        return self.f(inputs)

    def reset(self):
        for layer in self.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()

class Hopfield(module.Module):
    def __init__(self, input_size, lr=1E-3, clamp=np.inf):
        super(Hopfield, self).__init__()
        self.M = torch.zeros(1, input_size, input_size)
        self.lr = lr
        self.clamp = clamp

    def forward(self, inputs, reset=False):
        """
        inputs is (steps, batch, states)
        """

        if reset:
            self.reset()
        outputs = []
        for input in inputs:
            input = input.unsqueeze(1)
            output = torch.bmm(input, self.M)
            output = output.squeeze(0)
            outputs.append(output)
            # Sloppy outer product calculation
            learn_term = torch.outer(input.squeeze(), input.squeeze())
            learn_term = learn_term.unsqueeze(0)
            self.M = self.M + self.lr*learn_term
            self.M = torch.clamp(self.M, min=-self.clamp, max=self.clamp)
        outputs = torch.stack(outputs)
        return outputs

    def reset(self):
        torch.nn.init.zeros_(self.M)

class OjaRNN(AnalyticSR):
    """ Output is M.T @ i """

    def __init__(
        self, num_states, gamma, stay_case=0, ca3_kwargs={}
        ):
        """
        Args:
            stay_case: 
                0: run the model with no modifications
                1: ignore stay inputs by stitching together transition inputs,
                2: ignore stay inputs but continue learning rate update during stay
        """

        super(OjaRNN, self).__init__(num_states, gamma, stay_case)
        self.dg = dg.DG()
        self.ca3 = ca3.OjaCA3(num_states, gamma, **ca3_kwargs) 
        self.estimates_T = True
