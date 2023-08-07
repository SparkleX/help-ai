import matplotlib.pyplot as plt
import numpy as np
import torch

def func(x):
    if x>0:
        return x
    return 0

x_train = []
y_train = []
for x in range(-10,10):
    y = func(x)
    x_train.append([x])
    y_train.append([y])

torch.save(x_train, 'datasets/train.pt')
torch.save(y_train, 'datasets/train_target.pt')

x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

fig = plt.figure(figsize=(10, 5))
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.legend() 
plt.show()

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

