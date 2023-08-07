import numpy as np
import torch

model = torch.load('save.model')

while True:
    print('input a number ?')
    x = int(input())
    train = np.array([x], dtype=np.float32)
    train = torch.from_numpy(train)
    y = model(train)
    print(y)