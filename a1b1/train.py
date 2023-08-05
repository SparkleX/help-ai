import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

x_train = np.array([[1], [2], [3], [4],[100], [200]], dtype=np.float32)

y_train = np.array([[3], [5], [7], [9],[201], [401]], dtype=np.float32)


x_train = torch.from_numpy(x_train)

y_train = torch.from_numpy(y_train)


# Linear Regression Model
class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


model = linearRegression()
# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

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
'''
with torch.no_grad():
    predict = model(x_train)
predict = predict.data.numpy()
'''
'''
while True:
    print('input a number ?')
    x = int(input())
    train = np.array([x], dtype=np.float32)
    train = torch.from_numpy(train)
    y = model(train)
    print(y)
'''

'''
fig = plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
#plt.plot(x_train.numpy(), predict, 'ro', label='Fitting Line')
# 显示图例
plt.legend() 
plt.show()
'''
# 保存模型
#torch.save(model.state_dict(), './linear.pth')