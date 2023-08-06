import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

x_train = torch.load('datasets/train.pt')
print(x_train)
y_train = torch.load('datasets/train_target.pt')
print(y_train)