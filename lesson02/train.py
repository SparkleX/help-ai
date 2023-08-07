import numpy as np
import torch
from torch import nn
from model import linearRegression

x_train = np.array([[1], [2], [3], [4],[100], [200]], dtype=np.float32)
y_train = np.array([[3], [5], [7], [9],[201], [401]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

model = linearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

num_epochs = 3
for epoch in range(num_epochs):
    inputs = x_train
    target = y_train

    # forward
    out = model(inputs)
    loss = criterion(out, target)

    print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')
    for param in model.named_parameters():
        print(param)
    
    print(out)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
