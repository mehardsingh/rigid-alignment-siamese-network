import torch
from transforms import quaternion_to_matrix, tfm_from_pose
import numpy as np

# SOURCE: https://github.com/vinits5/learning3d/

def tfm_from_rand_pose(batch_size, translation_mean, translation_std):
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

def deg_to_rad(deg):
	return np.pi / 180 * deg

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack(
        (np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1
    )
    ry = np.stack(
        (np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1
    )
    rz = np.stack(
        (np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1
    )

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)

def create_random_transform(batch_size, max_rotation_deg, max_translation, dtype):
    max_rotation = deg_to_rad(max_rotation_deg)
    rot = np.random.uniform(-max_rotation, max_rotation, [batch_size, 3])
    trans = np.random.uniform(-max_translation, max_translation, [batch_size, 3])
    quat = euler_to_quaternion(rot, "xyz")

    vec = np.concatenate([quat, trans], axis=1)
    vec = torch.tensor(vec, dtype=dtype)

    rand_tfms = tfm_from_pose(vec)
    return rand_tfms

def jitter_pointcloud(pointcloud, sigma=0.04, clip=0.05):
	# N, C = pointcloud.shape
	sigma = 0.04*np.random.random_sample()
	pointcloud += torch.empty(pointcloud.shape).normal_(mean=0, std=sigma).clamp(-clip, clip)
	return pointcloud
