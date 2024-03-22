import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys
from get_modelnet40 import read_off

sys.path.append("src/ras_pointnet")
from rasnet_utils import RASNetEncoder

def is_rigid(rotation_matrix):
    # Check if the matrix is orthogonal
    print(np.dot(rotation_matrix.T, rotation_matrix))
    is_orthogonal = np.allclose(np.dot(rotation_matrix.T, rotation_matrix), np.eye(3), atol=1e-3)
    
    # Check if the determinant is 1
    determinant = np.linalg.det(rotation_matrix)
    is_determinant_one = np.isclose(determinant, 1)

    print(is_orthogonal, is_determinant_one)
    
    return is_orthogonal and is_determinant_one

with open("check_rigid/laptop_0156.off", 'r') as f:
    P, _ = read_off(f)
P = np.array(P)
P.shape

# Extract x, y, and z coordinates from the point cloud
plt.ion()

x = P[:, 0]
y = P[:, 1]
z = P[:, 2]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Visualization')

min_axis_size = 0
max_axis_size = 250
# ax.set_xlim(-min_axis_size, max_axis_size)
# ax.set_ylim(-min_axis_size, max_axis_size)
# ax.set_zlim(-min_axis_size, max_axis_size)

# Show plot
plt.show()

# Keep the plot window open
plt.show(block=True)

P = torch.from_numpy(P).float()
P = P.unsqueeze(0).repeat(4, 1, 1)
model = RASNetEncoder(num_iters=5, global_feat=True, feature_transform=False, channel=3)
_, all_tfms, _ = model(P.transpose(2, 1))
T_3 = all_tfms[-1]
# print(T_3[0,:,3])
T_3[:,:3,3] *= 100

add_ones = torch.ones(P.shape[0], P.shape[1], 1) # B x N x 1

P_ones = torch.cat((P, add_ones), dim=2) # B x N x 4
P_trans = torch.bmm(P_ones, T_3.transpose(2,1)) # B x N x 4
print(T_3[0][:,3])
P_trans = P_trans[:, :, :3] # B x N x 3
P_trans_0_numpy = P_trans[0].detach().numpy()

P_rot = torch.bmm(P, T_3[:,:3,:3].transpose(2,1)) # B x N x 4
P_rot_0_numpy = P_rot[0].detach().numpy()

plt.ion()

x = P_rot_0_numpy[:, 0]
y = P_rot_0_numpy[:, 1]
z = P_rot_0_numpy[:, 2]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Visualization')

# ax.set_xlim(-min_axis_size, max_axis_size)
# ax.set_ylim(-min_axis_size, max_axis_size)
# ax.set_zlim(-min_axis_size, max_axis_size)

# Show plot
plt.show()



plt.ion()

x = P_trans_0_numpy[:, 0]
y = P_trans_0_numpy[:, 1]
z = P_trans_0_numpy[:, 2]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Visualization')

# ax.set_xlim(-axis_size, axis_size)
# ax.set_ylim(-axis_size, axis_size)
# ax.set_zlim(-axis_size, axis_size)

# Show plot
plt.show()



# Keep the plot window open
plt.show(block=True)

rand_rot = torch.rand(3, 3)
R_rand = rand_rot + torch.eye(3)
P_rot = P[0] @ R_rand

print(is_rigid(R_rand.detach().numpy()))
print("==============")

plt.ion()

x = P_rot[:, 0]
y = P_rot[:, 1]
z = P_rot[:, 2]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Visualization')

# ax.set_xlim(-min_axis_size, max_axis_size)
# ax.set_ylim(-min_axis_size, max_axis_size)
# ax.set_zlim(-min_axis_size, max_axis_size)

# Show plot
plt.show()

# Keep the plot window open
plt.show(block=True)