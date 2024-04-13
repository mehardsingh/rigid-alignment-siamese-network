import torch

class Proj(torch.nn.Module):
    def __init__(self):
        super(Proj, self).__init__()

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x