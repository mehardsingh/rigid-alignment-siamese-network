import torch
import torch.nn as nn
from rasnet_utils import RASNetEncoder

class RASNet(nn.Module):
    def __init__(self, num_iters=5, global_feat=True, feature_transform=False, channel=3):
        super(RASNet, self).__init__()
        self.ras_net_encoder = RASNetEncoder(
            num_iters, 
            global_feat, 
            feature_transform, 
            channel
        )

    def forward(self, pc1, pc2):
        pc1_f, pc1_tfs, pc1_feat_tfm = self.ras_net_encoder(pc1)
        pc2_f, pc2_tfs, pc2_feat_tfm = self.ras_net_encoder(pc2)

        return (pc1_f, pc2_f), (pc1_tfs, pc2_tfs), (pc1_feat_tfm, pc2_feat_tfm)

device = "mps"
model = RASNet().to(device)
pc1 = torch.rand(32, 3, 1024).to(device)
std = 1e-6
noise = torch.normal(mean=torch.zeros(32, 3, 1024), std=torch.ones(32, 3, 1024)*std).to(device)
pc2 = pc1.clone() + noise

# print(pc1[0,:,0])
# print(pc2[0,:,0])

fts, tfms, ft_tfm = model(pc1, pc2)

print(fts[0][0])
print(fts[1][0])
        
        
