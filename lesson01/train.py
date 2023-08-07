import numpy as np
import torch
from torch import nn
from model import linearRegression
import matplotlib.pyplot as plt

x_train = np.array([[1], [2], [3], [4],[100], [200]], dtype=np.float32)
y_train = np.array([[3], [5], [7], [20],[201], [401]], dtype=np.float32)


x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

model = linearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

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
        for param in model.named_parameters():
            print(param)
        print()

model.eval()
torch.save(model, 'save.model')


fig = plt.figure(figsize=(200, 400))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), out.data.numpy(), label='Fitting Line')

plt.legend() 
plt.show()