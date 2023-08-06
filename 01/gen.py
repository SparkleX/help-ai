import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from model import linearRegression

model = torch.load('save.model')

while True:
    print('input a number ?')
    x = int(input())
    train = np.array([x], dtype=np.float32)
    train = torch.from_numpy(train)
    y = model(train)
    print(y)