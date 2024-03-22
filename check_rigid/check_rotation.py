import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys
from get_modelnet40 import read_off

sys.path.append("src/ras_pointnet")
from rasnet_utils import RASNetEncoder

def compute_pairwise_dist(points):
    squared_distances = torch.sum((points[:, None] - points) ** 2, dim=-1)
    return squared_distances

model = RASNetEncoder(num_iters=5, global_feat=True, feature_transform=False, channel=3)

with open("check_rigid/laptop_0156.off", 'r') as f:
    P, _ = read_off(f)
P = torch.from_numpy(np.array(P)).float()
batched_input = P.unsqueeze(0).repeat(4, 1, 1)

single_input = batched_input[0]
input_pairwise = compute_pairwise_dist(single_input)

_, all_tfms, _ = model(batched_input.permute(0,2,1))
T_3 = all_tfms[-1]

add_ones = torch.ones(batched_input.shape[0], batched_input.shape[1], 1)
batched_input_ones = torch.cat((batched_input, add_ones), dim=2) # B x N x 4

for i in range(3):
    print("==============================================")
    if i == 0:
        permuted_batched_input_ones = batched_input_ones.permute(0, 2, 1)
        transformed_points = torch.bmm(T_3, permuted_batched_input_ones)
        batched_output = transformed_points.permute(0, 2, 1)
    elif i == 1:
        batched_output = torch.bmm(batched_input_ones, T_3.transpose(2,1))
    else:
        rand_rot = torch.rand(4, 3, 3)
        R_rand = rand_rot + torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
        batched_output = torch.bmm(batched_input, R_rand.transpose(2,1))

    batched_output = batched_output[:, :, :3]
    single_output = batched_output[0]

    print(single_output[0][:10])

    output_pairwise = compute_pairwise_dist(single_output)

    print(input_pairwise[0][:10])
    print(output_pairwise[0][:10])

    print(torch.allclose(input_pairwise, output_pairwise, atol=1e-2))