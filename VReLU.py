import torch
import torch.nn as nn


"""Variable ReLU (VReLU)"""
class VReLU(nn.Module):
    """
    VReLU: a * relu(x)
    where a is a learnable scale parameter.
    """
    def __init__(self, scale_value=1.0, scale_learnable=True, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)

