import matplotlib.pyplot as plt
import numpy as np
import torch

x_train = []
y_train = []
for x in range(1,50):
    y = x * 2 + 1
    x_train.append([x])
    y_train.append([y])

x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

fig = plt.figure(figsize=(10, 5))
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.legend() 
plt.show()

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

torch.save(x_train, 'datasets/train.pt')
torch.save(y_train, 'datasets/train_target.pt')