import numpy as np
import modules

class AnalyticSR(object):
    def __init__(self, gamma, num_states):
        self.dg = modules.DG()
        self.ca3 = modules.CA3(gamma, num_states)
        self.num_states = num_states

    def forward(self, input, step=0):
        dg_out = self.dg.forward(input, step)
        ca3_out = self.ca3.forward(input)
        return dg_out, ca3_out

    def query(self, query_input, prev_input):
        step = 0
        _, prev_out = self.forward(prev_input)
        input = query_input + prev_out
        while True: 
            dg_out, ca3_out = self.forward(input, step)
            if step == 1: #np.sum(dg_out > 0) <= 1:# TODO: run until settling
                break
            else:
                step += 1
                input = ca3_out
        return dg_out, ca3_out

    def update(self, input, prev_input):
        self.ca3.update(input, prev_input)
