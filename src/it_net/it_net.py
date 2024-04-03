import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("src/pretrain_utils/")
from transforms import quaternion_to_matrix, apply_tfm, compose_tfms

class RigidTNet(nn.Module):
    def __init__(self, channel):
        super(RigidTNet, self).__init__()
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

        quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(batchsize, 1).to(device)
        quaternions += x[:, :4] # (batch x 4) 

        norms = torch.norm(quaternions, dim=1, keepdim=True)
        normalized_quaternions = quaternions / norms

        rotation = quaternion_to_matrix(normalized_quaternions).to(device) # (batchx3x3)
        translation = x[:, 4:7] # (batch x 3)

        transform = torch.eye(4).repeat(batchsize, 1, 1).to(device) # (batchx4x4)
        transform[:, :3, :3] = rotation
        transform[:, :3, 3] = translation

        return transform, rotation, translation
    
class ITNet(nn.Module):
    def __init__(self, channel, num_iters):
        super(ITNet, self).__init__()
        self.rigid_tnet = RigidTNet(channel=channel)
        self.num_iters = num_iters

    def forward(self, points):
        B, D, N = points.size()
        device = points.device

        T = torch.zeros(self.num_iters+1,B,4,4).to(device)
        T = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)

        T_deltas = list()

        for i in range(self.num_iters):
            transformed_points = apply_tfm(points, T)
            transformed_points = transformed_points.detach()
            T_delta, rotation, translation = self.rigid_tnet(transformed_points)
            T_deltas.append(T_delta)
            T = compose_tfms(T, T_delta)

        transformed_points = apply_tfm(points, T)
        return transformed_points, T, T_deltas