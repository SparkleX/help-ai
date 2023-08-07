from torch import nn
import torch
class NonLinearRegression(nn.Module):
    def __init__(self):
        super(NonLinearRegression, self).__init__()
        self.layer1 = nn.Sequential(                        
            nn.Linear(1, 2),
            nn.ReLU(False))
        self.layer2 = nn.Sequential(
            nn.Linear(2, 1),
            nn.ReLU(False))
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
'''
class NonLinearRegression(nn.Module):
    def __init__(self):
        super(NonLinearRegression, self).__init__()
        self.layer1 = nn.Sequential(            
            nn.Linear(1, 1),
            nn.ReLU(False))
        self.layer2 = nn.Sequential(            
            nn.Linear(1, 2),
            nn.ReLU(False))   
        self.layer3 = nn.Sequential(            
            nn.Linear(2, 1))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
'''