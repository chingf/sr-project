import torch
import torch.nn as nn

class Clamp(nn.Module):

    def __init__(self, floor, ceil, ceil_offset=0):
        super(Clamp, self).__init__()
        self.floor = floor
        self.ceil = ceil
        self.ceil_offset = ceil_offset

    def forward(self, input):
        if torch.is_tensor(self.ceil):
            ceil = torch.abs(self.ceil)
        else:
            ceil = self.ceil
        return torch.clamp(
            input, min=self.floor,
            max=ceil+self.ceil_offset
            )

class LeakyThreshold(nn.Module):
    """
    A leaky threshold function
    """

    def __init__(self, x0=0, x1=1, floor=0, ceil=None):
        super(LeakyThreshold, self).__init__()
        self.x0 = x0
        self.x1 = x1
        self.floor = floor
        self.ceil = ceil

    def forward(self, input, negative_slope=0):
        offset = self.x0
        scale = 1/(self.x1 - self.x0)
        input = (input - offset)*scale
        input = torch.nan_to_num(input)

        if self.ceil is not None:
            input = -1*(nn.functional.leaky_relu(
                -input + self.ceil, negative_slope=negative_slope
                ) - self.ceil)
        if self.floor is not None:
            input = nn.functional.leaky_relu(input, negative_slope=negative_slope)
        return input

class TwoSidedLeakyThreshold(nn.Module):
    """
    A leaky clamp
    """

    def __init__(self, x0, x1, floor=0, ceil=None):
        super(TwoSidedLeakyThreshold, self).__init__()
        self.x0 = x0
        self.x1 = x1
        self.floor = floor
        self.ceil = ceil

    def forward(self, input, negative_slope=0):
        negative_locs = (input < 0).float()
        input = torch.abs(input)

        offset = self.x0
        scale = 1/(self.x1 - self.x0)
        input = (input - offset)*scale

        if self.ceil is not None:
            input = -1*(nn.functional.leaky_relu(
                -input + self.ceil, negative_slope=negative_slope
                ) - self.ceil)
        if self.floor is not None:
            input = nn.functional.leaky_relu(input, negative_slope=negative_slope)

        flip_back_to_neg = (-2*negative_locs) + 1
        input = input*flip_back_to_neg

        return input

