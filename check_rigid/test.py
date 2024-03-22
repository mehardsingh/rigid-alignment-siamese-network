import torch

def transform_points_approach1(points, transformation):
    points = points.permute(0, 2, 1)
    transformed_points = torch.bmm(transformation, points)
    transformed_points = transformed_points.permute(0, 2, 1)
    return transformed_points

def transform_points_approach2(points, transformation):
    transformed_points = torch.bmm(points, transformation.transpose(2, 1))
    return transformed_points

# Example usage
B = 1  # Batch size
N = 3  # Number of points
points = torch.tensor([[[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 1]]], dtype=torch.float32)
transformation = torch.tensor([[[1, 0, 0, 2],
                                [0, 1, 0, 3],
                                [0, 0, 1, 4],
                                [0, 0, 0, 1]]], dtype=torch.float32)

transformed_points_approach1 = transform_points_approach1(points, transformation)
transformed_points_approach2 = transform_points_approach2(points, transformation)

# Check if outputs are the same
print(torch.allclose(transformed_points_approach1, transformed_points_approach2))