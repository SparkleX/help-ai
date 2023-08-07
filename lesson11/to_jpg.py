
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

use_gpu = torch.cuda.is_available()

train_dataset = datasets.MNIST(root='./datasets', train=True)

print(train_dataset)

i = 0
for data in train_dataset:
    train_image_zero, train_target_zero = data
    train_image_zero.save("images/{i}-{target}.png".format(i=i,target=train_target_zero))
    i = i+ 1
    if i>=100:
        break
