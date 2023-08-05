import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from model import RegressionX1X2



model = torch.load('save.model')

while True:
    print('input two number ?')
    x1 = int(input())
    x2 = int(input())
    train = np.array([x1,x2], dtype=np.float32)
    train = torch.from_numpy(train)
    y = model(train)
    print(y)