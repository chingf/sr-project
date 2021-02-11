import numpy as np
import modules

class AnalyticSR(object):
    def __init__(self, gamma, num_states):
        self.dg = modules.DG()
        self.ca3 = modules.CA3(gamma, num_states)
        self.num_states = num_states
        self.gamma = gamma

    def forward(self, input, update=True):
        dg_out = self.dg.forward(input)
        ca3_out = self.ca3.forward(input)
        if update:
            self.update()
        return dg_out, ca3_out

    def query(self, query_input, prev_input): # TODO
        return None, None

    def update(self):
        self.ca3.update()

class STDPSR(object):
    def __init__(self, gamma, num_states):
        self.dg = modules.DG()
        self.ca3 = modules.STDP_Net(num_states)
        self.num_states = num_states
        self.gamma = gamma

    def forward(self, input, update=True):
        dg_out = self.dg.forward(input)
        ca3_out = self.ca3.forward(input)
        if update:
            self.update()
        return dg_out, ca3_out

    def query(self, query_input, prev_input): #TODO
        return None, None

    def update(self):
        self.ca3.update()
        self.prev_input = input

