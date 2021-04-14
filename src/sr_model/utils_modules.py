import torch
import torch.nn as nn

class LeakyClamp(nn.Module):
    """
    A leaky clamp
    """

    def __init__(self, floor, ceil):
        super(LeakyClamp, self).__init__()
        self.floor = floor
        self.ceil = ceil

    def forward(self, input, negative_slope=0):
        if self.ceil is not None:
            input = -1*(nn.functional.leaky_relu(
                -input + self.ceil, negative_slope=negative_slope
                ) - self.ceil)
        if self.floor is not None:
            input = nn.functional.leaky_relu(input, negative_slope=negative_slope)
        return input

class LeakyThreshold(nn.Module):
    """
    A leaky threshold function
    """

    def __init__(self, x0, x1, floor=0, ceil=None):
        super(LeakyThreshold, self).__init__()
        self.x0 = x0
        self.x1 = x1
        self.offset = x0
        self.scale = 1/(x1 - x0)
        self.floor = floor
        self.ceil = ceil

    def forward(self, input, negative_slope=0):
        input = (input - self.offset)/self.scale
        if self.ceil is not None:
            input = -1*(nn.functional.leaky_relu(
                -input + self.ceil, negative_slope=negative_slope
                ) - self.ceil)
        if self.floor is not None:
            input = nn.functional.leaky_relu(input, negative_slope=negative_slope)
        return input

