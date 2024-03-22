# SOURCE: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from it_net import ITNet

def apply_tfm(x, transform):
    device = x.device
    B = x.shape[0]
    N = x.shape[2]

    x = x.transpose(2, 1) # B x N x D=3
    add_ones = torch.ones(B, N, 1) # B x N x 1
    add_ones = add_ones.to(device)
    
    x = torch.cat((x, add_ones), dim=2) # B x N x 4
    x = torch.bmm(x, transform.transpose(2,1)) # B x N x 4
    x = x[:, :, :3] # B x N x 3
    x = x.transpose(2,1)

    return x

def compose_tfms(tfm1, tfm2):
    R1 = tfm1[:, :3, :3]
    t1 = tfm1[:, :3, 3].unsqueeze(-1)

    R2 = tfm2[:, :3, :3]
    t2 = tfm2[:, :3, 3].unsqueeze(-1)

    R_composed = torch.matmul(R2, R1)
    t_composed = torch.matmul(R2, t1) + t2

    composed_matrix = torch.cat((
        torch.cat((R_composed, t_composed), dim=2),
        torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32, device=tfm1.device).repeat(tfm1.size(0), 1, 1)
    ), dim=1)

    return composed_matrix

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

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
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        iden = iden.to(device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class RASNetEncoder(nn.Module):
    def __init__(self, num_iters=5, global_feat=True, feature_transform=False, channel=3):
        super(RASNetEncoder, self).__init__()
        self.num_iters = num_iters
        self.it_net = ITNet(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        device = x.device

        all_tfms = torch.zeros(self.num_iters+1,B,4,4).to(device)
        all_tfms[0] = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)
        for i in range(1, self.num_iters+1):
            curr_transform, curr_rotation, curr_translation = self.it_net(x)
            x = apply_tfm(x, curr_transform)
            all_tfms[i] = compose_tfms(all_tfms[i-1], curr_transform)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, all_tfms, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), all_tfms, trans_feat

def feature_transform_reguliarzer(trans):
    device = trans.device
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    I = I.to(device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss