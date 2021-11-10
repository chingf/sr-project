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

    def forward(self, dg_inputs, dg_modes=None, reset=True, update=True):
        """
        Args:
            dg_inputs: (steps, batch, states) one-hot inputs
            dg_modes: (steps, batch, 1) flag of global mode
        """

        num_steps, batch_size, num_states = dg_inputs.shape
        out = []
        if reset:
            self.reset()
        for step in np.arange(num_steps):
            dg_input = dg_inputs[step, :, :]
            if dg_modes is not None:
                dg_mode = dg_modes[step, :]
                prev_dg_mode = dg_modes[step-1, :] if step != 0 else np.nan

            if dg_modes is None or dg_mode == 0: # Predictive mode
                is_stay = step > 1 and torch.equal(self.ca3.prev_input, self.ca3.curr_input)
                if (not is_stay) or (self.stay_case == 0):
                    dg_out = self.dg(dg_input, update_transition=update)
                    ca3_out = self.ca3(dg_input, update_transition=update)
                    self.update()
                elif self.stay_case == 1:
                    dg_out = self.dg(dg_input, update_transition=False)
                    ca3_out = self.ca3(dg_input, update_transition=False)
                else:
                    dg_out = self.dg(dg_input, update_transition=False)
                    ca3_out = self.ca3(dg_input, update_transition=False)
                    self.ca3._decay_all_eta_invs()
                out.append(ca3_out)
            elif (dg_mode == 1) and (prev_dg_mode == 0): # Query Mode
                dg_out, ca3_out = self.query(dg_input)
        out = torch.stack(out)
        return None, out

    def query(self, query_input): # TODO
        return None, None

    def update(self):
        self.ca3.update()

    def get_M(self, gamma=None):
        return self.ca3.get_M_hat(gamma=gamma)

    def reset(self):
        self.ca3.reset()

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

    def __init__(self, input_size):
        super(Linear, self).__init__()
        self.M = torch.zeros(1, input_size, input_size)

    def forward(self, inputs, reset=False):
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
        outputs = torch.stack(outputs)
        return outputs

    def reset(self):
        torch.nn.init.zeros_(self.M)

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

