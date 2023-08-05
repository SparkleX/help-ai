import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class Regression2(nn.Module):
    def __init__(self):
        super(Regression2, self).__init__()
        self.layer1 = nn.Linear(1, 1)
        self.layer2 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.layer1(x*x)
        x = self.layer2(x)
        return x

