# SOURCE: https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-ii-pytorch

import os
import numpy as np
import math, random
# random.seed = 42

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def read_off(file):
    off_header = file.readline().strip()
    if 'OFF' == off_header:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        
        return sampled_points
    
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud
    
class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)
    
def default_transforms():
    return transforms.Compose([
        PointSampler(1024),
        Normalize(),
        ToTensor()
    ])

class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, dir))]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = os.path.join(root_dir, category, folder)
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = os.path.join(new_dir, file)
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud, 
                'category': self.classes[category]}

def get_dls_for_viz(batch_size=32):
    train_transforms = transforms.Compose([
        PointSampler(1024),
        Normalize(),
        ToTensor()
    ])

    train_ds = PointCloudData("ModelNet40", transform=train_transforms)
    train_num = int(len(train_ds)*0.8)
    val_num = len(train_ds) - train_num
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_num, val_num])
    # test_ds = PointCloudData("ModelNet40", valid=True, folder='test', transform=train_transforms)

    num_workers = 10
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader

def get_train_test_dls(batch_size=32):
    train_transforms = transforms.Compose([
        PointSampler(1024),
        Normalize(),
        # RandRotation_z(),
        # RandomNoise(),
        ToTensor()
    ])

    train_ds = PointCloudData("ModelNet40", transform=train_transforms)
    train_num = int(len(train_ds)*0.8)
    val_num = len(train_ds) - train_num
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_num, val_num])
    # test_ds = PointCloudData("ModelNet40", valid=True, folder='test', transform=train_transforms)
    test_ds = PointCloudData("ModelNet40", folder='test', transform=train_transforms)

    num_workers = 10
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def get_train_test_dls_cls(batch_size=32):
    train_transforms = transforms.Compose([
        PointSampler(1024),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        ToTensor()
    ])

    train_ds = PointCloudData("ModelNet40", transform=train_transforms)
    train_num = int(len(train_ds)*0.8)
    val_num = len(train_ds) - train_num
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_num, val_num])
    test_ds = PointCloudData("ModelNet40", valid=True, folder='test', transform=train_transforms)
    # test_ds = PointCloudData("ModelNet40", folder='test', transform=train_transforms)

    num_workers = 10
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_train_test_dls()