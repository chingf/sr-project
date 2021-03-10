import numpy as np
import modules

class AnalyticSR(object):
    def __init__(self, gamma, num_states):
        self.dg = modules.DG()
        self.ca3 = modules.CA3(gamma, num_states)
        self.num_states = num_states
        self.gamma = gamma
        self.estimates_T = False

    def forward(self, input, update=True):
        dg_out = self.dg.forward(input)
        ca3_out = self.ca3.forward(input)
        if update:
            self.update()
        return dg_out, ca3_out

    def query(self, query_input): # TODO
        return None, None

    def update(self):
        self.ca3.update()

class STDP_SR(STDP_SR):
    def __init__(self, gamma, num_states, stay_case=0):
        """
        Args:
            stay_case: 
                0: run the model with no modifications
                1: ignore stay inputs by stitching together transition inputs,
                2: ignore stay inputs but continue learning rate update during stay
        """

        super().__init__(gamma, num_states)
        self.ca3 = modules.STDP_LR_Net(num_states)
        self.stay_case = stay_case

    def forward(self, input, update=True):
        is_stay = np.mean(self.ca3.prev_input == self.ca3.curr_input) == 1.0
        if (not is_stay) or (self.stay_case == 0):
            dg_out = self.dg.forward(input)
            ca3_out = self.ca3.forward(input)
            if update:
                self.update()
        elif self.stay_case == 1:
            dg_out = None
            ca3_out = None
        else:
            dg_out = None
            ca3_out = None
            self.ca3._decay_all_eta_invs()
        return dg_out, ca3_out

