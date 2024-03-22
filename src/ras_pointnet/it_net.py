import torch
import torch.nn as nn
import torch.nn.functional as F
from quaternion_to_matrix import quaternion_to_matrix

class ITNet(nn.Module):
    def __init__(self, channel):
        super(ITNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        device = x.device

        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) # (B x 7)

        quaternions = x[:, :4] # (batch x 4) 
        norms = torch.norm(quaternions, dim=1, keepdim=True)
        normalized_quaternions = quaternions / norms

        rotation = quaternion_to_matrix(normalized_quaternions).to(device) # (batchx3x3)
        translation = x[:, 4:7] # (batch x 3)

        transform = torch.eye(4).repeat(batchsize, 1, 1).to(device) # (batchx4x4)
        transform[:, :3, :3] = rotation
        transform[:, :3, 3] = translation

        return transform, rotation, translation