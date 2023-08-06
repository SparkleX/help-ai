import torch
from torch import nn
from model import NonLinearRegression

m = nn.Threshold(0.1, 20)
input = torch.randn(2)
output = m(input)
print(input)
print(output)