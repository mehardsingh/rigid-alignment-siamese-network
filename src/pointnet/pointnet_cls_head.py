import torch
import torch.nn.functional as F

class ClsHead(torch.nn.Module):
    def __init__(self, k=40):
        super(ClsHead, self).__init__()

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class ClsHeadProj(torch.nn.Module):
    def __init__(self, k=40):
        super(ClsHeadProj, self).__init__()

        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, k)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x