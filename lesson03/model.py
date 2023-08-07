from torch import nn

class NonLinearRegression(nn.Module):
    def __init__(self):
        super(NonLinearRegression, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU(True))
        
    def forward(self, x):
        x = self.layer1(x)
        return x