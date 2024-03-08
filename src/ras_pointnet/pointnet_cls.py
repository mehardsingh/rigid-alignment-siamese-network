# SOURCE: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_cls.py

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from pointnet_cls_head import ClsHead

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()

        if normal_channel:
            channel = 6
        else:
            channel = 3
        
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.cls_head = ClsHead(k=k)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x) # (BxNx3) -> (BxC)
        logits = self.cls_head(x) # (BxC) -> (Bxk)

        return logits, trans, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss