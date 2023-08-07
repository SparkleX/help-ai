import torch
from torch import nn
from model import NonLinearRegression
import matplotlib.pyplot as plt

x_train = torch.load('datasets/train.pt')
y_train = torch.load('datasets/train_target.pt')


model = NonLinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


num_epochs = 5000
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
    
    if (epoch+1) % 1000 == 0:
        print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        print()

model.eval()
torch.save(model, 'save.model')

fig = plt.figure(figsize=(20, 10))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), out.data.numpy(), label='Fitting Line')

plt.legend() 
plt.show()