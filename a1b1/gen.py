import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

# Linear Regression Model
class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out

model = torch.load('save.model')

while True:
    print('input a number ?')
    x = int(input())
    train = np.array([x], dtype=np.float32)
    train = torch.from_numpy(train)
    y = model(train)
    print(y)