from model import neuralNetwork
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

use_gpu = torch.cuda.is_available()

dataset = datasets.MNIST(root='./datasets', train=False, transform=transforms.ToTensor())

model = torch.load('save.model')

dataset_size = len(dataset)
print(dataset_size)
fail = 0
for i in range(0,dataset_size):
    data=dataset[i]
    image, target = data
    image = image.view(image.size(0), -1)
    out = model(image)
    _, pred = torch.max(out, 1)
    #print(out)
    if pred!=target:
        print(pred, target)
        fail=fail+1
print(fail/dataset_size)
