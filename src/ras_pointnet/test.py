from pointnet_cls import get_model
import torch

model = get_model(k=40, normal_channel=False)

batch_size = 32
num_points = 1024
input_dim = 3

P = torch.rand(batch_size, input_dim, num_points)
print("input shape", P.shape)

logits, T_3, T_64 = model(P)
predictions = torch.exp(logits)

print(logits[0])
print(predictions[0])
print(torch.sum(predictions[0]))

print()
print(T_3.shape)
print(T_64.shape)