from torch import nn
import torch
class NonLinearRegression(nn.Module):
    def __init__(self):
        super(NonLinearRegression, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(False))
        self.layer1 = nn.Sequential(            
            nn.Linear(2, 1),
            nn.ReLU(False))
        #self.layer2 = nn.Sequential(
        #    nn.Linear(2, 1))   
        
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        #x = self.layer2(x)
        return x