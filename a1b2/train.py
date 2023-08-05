import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from model import Regression2


    
x_train = np.array([[1], [2], [3], [4]], dtype=np.float32)
y_train = np.array([[1+5], [4+5], [9+5], [16+5]], dtype=np.float32)


x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


model = Regression2()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 开始训练
num_epochs = 50000
for epoch in range(num_epochs):
    inputs = x_train
    target = y_train

    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
    if (epoch+1) % 10000 == 0:
        print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        print()

model.eval()
torch.save(model, 'save.model')