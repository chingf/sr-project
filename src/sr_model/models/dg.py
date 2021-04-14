import numpy as np
import torch
import torch.nn as nn
from sr_model.utils import get_sr

class DG(nn.Module):
    def __init__(self):
        super(DG, self).__init__()
        pass

    def forward(self, input, update_transition=True):
        return input

