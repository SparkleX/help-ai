from torch import nn

class NonLinearRegression(nn.Module):
    def __init__(self):
        super(NonLinearRegression, self).__init__()
        self.layer1 = nn.Sequential(            
            nn.ReLU(False),
            nn.Linear(1, 1))
        
    def forward(self, x):
        x = self.layer1(x)
        return x