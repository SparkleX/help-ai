import matplotlib.pyplot as plt
import numpy as np
from torch import nn

class RegressionX1X2(nn.Module):
    def __init__(self):
        super(RegressionX1X2, self).__init__()
        self.layer1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer1(x)
        return x

