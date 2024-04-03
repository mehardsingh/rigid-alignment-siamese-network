import torch

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    # SOURCE: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

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

def tfm_from_pose(pose):
    device = pose.device
    batchsize = pose.shape[0]

    quaternions = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(batchsize, 1).to(device)
    quaternions += pose[:, :4] # (batch x 4) 

    norms = torch.norm(quaternions, dim=1, keepdim=True)
    normalized_quaternions = quaternions / norms

    rotation = quaternion_to_matrix(normalized_quaternions).to(device) # (batchx3x3)
    translation = pose[:, 4:7] # (batch x 3)

    transform = torch.eye(4).repeat(batchsize, 1, 1).to(device) # (batchx4x4)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation

    return transform