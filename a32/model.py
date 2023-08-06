import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

class neuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(neuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, 10))
        #self.layer4 =nn.Linear(10, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
       # x = self.layer4(x)
        return x