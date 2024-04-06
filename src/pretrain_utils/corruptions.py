import torch
from transforms import quaternion_to_matrix, tfm_from_pose
import numpy as np
import random
import scipy
import copy

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

class Jitter():
    def __init__(self, sigma=0.04, clip=0.05) -> None:
        self.sigma = sigma
        self.clip = clip    

    def __call__(self, pointcloud):
        pointcloud = pointcloud.transpose(1,2)
        sigma = self.sigma*np.random.random_sample()
        pointcloud += torch.empty(pointcloud.shape).normal_(mean=0, std=sigma).clamp(-self.clip, self.clip).to(pointcloud.device)
        pointcloud = pointcloud.transpose(1,2)
        return pointcloud
    
class Scale():
    def __init__(self, same=True, min_scale=0.75, max_scale=1.25) -> None:
        self.same = same
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, pointcloud):
        # b x n x 3
        scales = list()
        for i in range(pointcloud.shape[0]):
            xyz = (self.min_scale - self.max_scale) * torch.rand(3) + self.max_scale
            if not self.same:
                scale = torch.Tensor([[xyz[0], 0, 0], [0, xyz[1], 0], [0, 0, xyz[2]]]).unsqueeze(0)
            else:
                scale = torch.Tensor([[xyz[0], 0, 0], [0, xyz[0], 0], [0, 0, xyz[0]]]).unsqueeze(0)

            scales.append(scale)
        scales = torch.cat(scales, dim=0) #bx3x3
        scales = scales.to(pointcloud.device)
        return torch.bmm(pointcloud, scales)

# SOURCE: https://torch-points3d.readthedocs.io/en/latest/src/api/transforms.html
class ElasticDistortionSingle:
    """Apply elastic distortion on sparse coordinate space. First projects the position onto a 
    voxel grid and then apply the distortion to the voxel grid.

    Parameters
    ----------
    granularity: List[float]
        Granularity of the noise in meters
    magnitude:List[float]
        Noise multiplier in meters
    Returns
    -------
    data: Data
        Returns the same data object with distorted grid
    """

    def __init__(
        self, apply_distorsion: bool = True, granularity=[0.2, 0.8], magnitude=[0.4, 1.6],
    ):
        assert len(magnitude) == len(granularity)
        self._apply_distorsion = apply_distorsion
        self._granularity = granularity
        self._magnitude = magnitude

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        coords = coords.numpy()
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords = coords + interp(coords) * magnitude
        return torch.tensor(coords).float()

    def __call__(self, data):
        # coords = data.pos / self._spatial_resolution
        if self._apply_distorsion:
            if random.random() < 0.95:
                for i in range(len(self._granularity)):
                    data = ElasticDistortionSingle.elastic_distortion(data, self._granularity[i], self._magnitude[i],)
        return data

    def __repr__(self):
        return "{}(apply_distorsion={}, granularity={}, magnitude={})".format(
            self.__class__.__name__, self._apply_distorsion, self._granularity, self._magnitude,
        )

class ElasticDistortion():
    def __init__(
        self, apply_distorsion: bool = True, granularity=[0.2, 0.8], magnitude=[0.4, 1.6],
    ):
        self.ed = ElasticDistortionSingle(apply_distorsion, granularity, magnitude)

    def __call__(self, data):
        all_distorted = list()
        for i in range(data.shape[0]):
            ed_input = data[i].detach().cpu()
            distorted = self.ed(ed_input)
            all_distorted.append(distorted.unsqueeze(0))
        all_distorted = torch.cat(all_distorted, dim=0).to(data.device)
        return all_distorted
    
class Dropout():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        N = data.shape[1]
        all_data = list()
        for i in range(data.shape[0]):
            remaining_points = list() # Mx3
            for j in range(N):
                if random.random() > self.p:
                    remaining_points.append(data[i][j].unsqueeze(0))

            if len(remaining_points) == 0:
                break

            last = remaining_points[-1]
            while not len(remaining_points) == N:
                remaining_points.append(last.clone()) #Nx3
                
            remaining_points = torch.cat(remaining_points, dim=0) # Nx3
            all_data.append(remaining_points.unsqueeze(0))

        all_data = torch.cat(all_data, dim=0)
        all_data = all_data.to(data.device)
        return all_data
    
def apply_corruptions(pointcloud):
    corruptions = [Jitter(), Scale(same=True, min_scale=0.5, max_scale=2), ElasticDistortion(), Dropout(p=0.5)]
    corrupted = copy.deepcopy(pointcloud)
    for corruption in corruptions:
        corrupted = corruption(corrupted)
    return corrupted

def apply_jitter(pointcloud):
    corruptions = [Jitter()]
    corrupted = copy.deepcopy(pointcloud)
    for corruption in corruptions:
        corrupted = corruption(corrupted)
    return corrupted

# pointcloud = torch.randn(32, 1024, 3)
# dropout = Dropout()
# print(dropout(pointcloud).shape)
