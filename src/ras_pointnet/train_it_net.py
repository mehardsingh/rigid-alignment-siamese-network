import torch
import torch.nn as nn
import sys
import json
from tqdm import tqdm
from it_net import ITNet
from quaternion_to_matrix import quaternion_to_matrix
import numpy as np
import pandas as pd
import os
import argparse

sys.path.append("src/get_modelnet40")
from load_data import get_train_test_dls

class PLoss(nn.Module):
    def __init__(self):
        super(PLoss, self).__init__()

    def forward(self, pc1, pc2):
        pointwise_distance = torch.sum((pc1 - pc2)**2, dim=1) # BxN
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

def evaluate(it_net, val_dl, criterion, config):
    it_net.eval()

    val_loss = 0
    step = 0

    with torch.no_grad():
        pbar = tqdm(val_dl, desc="Validating")
        for batch in pbar:
            pointcloud = batch["pointcloud"].to(torch.float32).to(config.device)
            pc1 = pointcloud.transpose(2, 1)

            random_rigid = get_random_rigid_tfm(pc1.shape[0], config.translation_mean, config.translation_std).to(config.device)
            pc2 = apply_tfm(pc1, random_rigid)

            pc1_tfm, T, T_deltas = it_net(pc1)
            pc2_tfm, T, T_deltas = it_net(pc2)

            val_loss += criterion(pc1_tfm, pc2_tfm).item()
            step += 1

    return val_loss / step

# def train(config):
#     train_dl, val_dl, test_dl = get_train_test_dls(config.batch_size)

#     it_net = ITNet(channel=3, num_iters=5).to(config.device)
#     # load the model
#     # model.load_state_dict(torch.load(PATH))
#     optimizer = torch.optim.AdamW(it_net.parameters(), lr=config.lr, weight_decay=config.wd)
#     criterion = PLoss()

#     progress_dict = {
#         "step": list(),
#         "train_loss": list(),
#         "val_loss": list()
#     }
#     step = 0

#     all_training_loss = list()
#     best_val_loss = 5e5

#     for epoch in range(config.num_epochs):
#         pbar = tqdm(train_dl, desc="Training")

#         for batch in pbar:
#             optimizer.zero_grad()
            
#             pointcloud = batch["pointcloud"].to(torch.float32).to(config.device)
#             labels = batch["category"].to(config.device)
#             pc1 = pointcloud.transpose(2, 1)

#             random_rigid = get_random_rigid_tfm(pc1.shape[0], config.translation_mean, config.translation_std).to(config.device)
#             pc2 = apply_tfm(pc1, random_rigid)

#             pc1_tfm, T, T_deltas = it_net(pc1)
#             pc2_tfm, T, T_deltas = it_net(pc2)

#             loss = criterion(pc1_tfm, pc2_tfm)
#             loss.backward()
#             optimizer.step()

#             all_training_loss.append(loss.item())

#             if step % config.eval_every == 0:
#                 avg_train_loss = all_training_loss[-config.eval_every:]
#                 avg_train_loss = sum(avg_train_loss) / len(avg_train_loss)

#                 val_loss = eval(it_net, val_dl, criterion, config)

#                 progress_dict["step"].append(step)
#                 progress_dict["train_loss"].append(avg_train_loss)
#                 progress_dict["val_loss"].append(val_loss)

#                 progress_df = pd.DataFrame(progress_dict)
#                 progress_df.to_csv(os.path.join(config.save_dir, "progress.csv"))

#                 if val_loss <= best_val_loss:
#                     torch.save(it_net.state_dict(), os.path.join(config.save_dir, "model.pt"))
#                     best_val_loss = val_loss

#             if step % config.lr_decay_every == 0:
#                 for g in optimizer.param_groups:
#                     g['lr'] *= config.lr_decay
            
#             step += 1

def train(config):
    train_dl, val_dl, test_dl = get_train_test_dls(config.batch_size)

    it_net = ITNet(channel=3, num_iters=5).to(config.device)
    if config.load == "True":
        it_net.load_state_dict(torch.load(os.path.join(config.save_dir, "model.pt")))

        # geting the progress dict until the best model version
        progress_dict = pd.read_csv(os.path.join(config.save_dir, "progress.csv"))
        min_index = progress_dict["val_loss"].idxmin()
        progress_dict = progress_dict.iloc[:min_index + 1]
        print(progress_dict)

        step = list(progress_dict["step"])[-1] + 1
        all_training_loss = list()
        best_val_loss = list(progress_dict["val_loss"])[-1]

        print(step, all_training_loss, best_val_loss)

    else:
        progress_dict = {
            "step": list(),
            "train_loss": list(),
            "val_loss": list()
        }
        step = 0
        all_training_loss = list()
        best_val_loss = 5e5

    it_net.train()

    optimizer = torch.optim.AdamW(it_net.parameters(), lr=config.lr, weight_decay=config.wd)
    criterion = PLoss()

    while step < config.num_steps:
        pbar = tqdm(train_dl, desc="Training")

        for batch in pbar:
            optimizer.zero_grad()
            
            pointcloud = batch["pointcloud"].to(torch.float32).to(config.device)
            labels = batch["category"].to(config.device)
            pc1 = pointcloud.transpose(2, 1)

            random_rigid = get_random_rigid_tfm(pc1.shape[0], config.translation_mean, config.translation_std).to(config.device)
            pc2 = apply_tfm(pc1, random_rigid)

            pc1_tfm, T, T_deltas = it_net(pc1)
            pc2_tfm, T, T_deltas = it_net(pc2)

            loss = criterion(pc1_tfm, pc2_tfm)
            loss.backward()
            optimizer.step()

            all_training_loss.append(loss.item())

            if step % config.eval_every == 0:
                avg_train_loss = all_training_loss[-config.eval_every:]
                avg_train_loss = sum(avg_train_loss) / len(avg_train_loss)

                val_loss = evaluate(it_net, val_dl, criterion, config)

                progress_dict["step"].append(step)
                progress_dict["train_loss"].append(avg_train_loss)
                progress_dict["val_loss"].append(val_loss)

                progress_df = pd.DataFrame(progress_dict)
                progress_df.to_csv(os.path.join(config.save_dir, "progress.csv"))

                if val_loss <= best_val_loss:
                    torch.save(it_net.state_dict(), os.path.join(config.save_dir, "model.pt"))
                    best_val_loss = val_loss

            if step % config.lr_decay_every == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= config.lr_decay
            
            step += 1


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--device", type=str, default="mps")
#     parser.add_argument("--num_epochs", type=int, default=100)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--lr", type=float, default=0.001)
#     parser.add_argument("--wd", type=float, default=1e-1)
#     parser.add_argument("--translation_mean", type=float, default=0)
#     parser.add_argument("--translation_std", type=float, default=0.1)
#     parser.add_argument("--eval_every", type=int, default=50)
#     parser.add_argument("--lr_decay_every", type=int, default=2000)
#     parser.add_argument("--lr_decay", type=float, default=0.7)
#     parser.add_argument("--save_dir", type=str, default="results/it_net")
#     parser.add_argument("--load", type=str, default="False")
#     config = parser.parse_args()

#     train(config)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=1e-1)
    parser.add_argument("--translation_mean", type=float, default=0)
    parser.add_argument("--translation_std", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--lr_decay_every", type=int, default=2000)
    parser.add_argument("--lr_decay", type=float, default=0.7)
    parser.add_argument("--save_dir", type=str, default="results/it_net")
    parser.add_argument("--load", type=str, default="True")
    config = parser.parse_args()

    train(config)

