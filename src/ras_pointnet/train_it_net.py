import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from it_net import ITNet
from quaternion_to_matrix import quaternion_to_matrix

sys.path.append("src/get_modelnet40")
from load_data import get_train_test_dls

class PLoss(nn.Module):
    def __init__(self):
        super(PLoss, self).__init__()

    def forward(self, pc1, pc2):
        pointwise_distance = torch.sqrt(torch.sum((pc1 - pc2)**2, dim=1)) # BxN
        pc_mean_distance = torch.mean(pointwise_distance, dim=1) # B
        loss = torch.mean(pc_mean_distance)
        return loss

def get_random_rigid_tfm(batch_size, translation_mean, translation_std):
    quat = torch.normal(mean=torch.zeros(batch_size, 4), std=torch.ones(batch_size, 4))
    norm = torch.norm(quat, dim=1, keepdim=True)
    norm_quat = quat / norm
    rotation = quaternion_to_matrix(norm_quat)

    translation = torch.normal(
        mean=translation_mean*torch.ones(batch_size, 3), 
        std=translation_std*torch.ones(batch_size, 3)
    )

    rand_transform = torch.eye(4).repeat(batch_size, 1, 1)
    rand_transform[:, :3, :3] = rotation
    rand_transform[:, :3, 3] = translation
    
    return rand_transform

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

device = "mps"
num_epochs = 10
batch_size = 32
lr = 1e-4
wd = 1e-1
translation_mean = 0
translation_std = 1

train_dl, val_dl, test_dl = get_train_test_dls(batch_size)

it_net = ITNet(channel=3, num_iters=5).to(device)
optimizer = torch.optim.AdamW(it_net.parameters(), lr=lr, weight_decay=wd)
criterion = PLoss()

for epoch in range(1, num_epochs+1):

    pbar = tqdm(train_dl, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()
        
        pointcloud = batch["pointcloud"].to(torch.float32).to(device)
        labels = batch["category"].to(device)
        pc1 = pointcloud.transpose(2, 1)

        random_rigid = get_random_rigid_tfm(batch_size, translation_mean, translation_std).to(device)
        pc2 = apply_tfm(pc1, random_rigid)

        pc1_tfm, _ = it_net(pc1)
        pc2_tfm, _ = it_net(pc2)

        loss = criterion(pc1_tfm, pc2_tfm)
        loss.backward()
        optimizer.step()

        # compute loss wrt. pc1_tfm, pc2_tfm
        

