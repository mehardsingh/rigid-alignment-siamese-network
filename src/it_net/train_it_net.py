import torch
import torch.nn as nn
import sys
import json
from tqdm import tqdm
from it_net import ITNet
import numpy as np
import pandas as pd
import os
import argparse

sys.path.append("src/get_modelnet40")
from load_data import get_train_test_dls

sys.path.append("src/pretrain_utils/")
from transforms import quaternion_to_matrix, apply_tfm, compose_tfms
from corruptions import tfm_from_rand_pose, create_random_transform

sys.path.append("src/pretrain_utils")
from corruptions import apply_jitter

class PLoss(nn.Module):
    def __init__(self):
        super(PLoss, self).__init__()

    def forward(self, pc1, pc2):
        pointwise_distance = torch.sum((pc1 - pc2)**2, dim=1) # BxN
        pc_mean_distance = torch.mean(pointwise_distance, dim=1) # B
        loss = torch.mean(pc_mean_distance)
        return loss

def evaluate(it_net, val_dl, criterion, config):
    val_loss = 0
    step = 0

    with torch.no_grad():
        pbar = tqdm(val_dl, desc="Validating")
        for batch in pbar:
            pointcloud = batch["pointcloud"].to(torch.float32).to(config.device)
            pc1 = pointcloud.transpose(2, 1)

            # random_rigid = get_random_rigid_tfm(pc1.shape[0], config.translation_mean, config.translation_std).to(config.device)
            random_rigid = create_random_transform(pc1.shape[0], max_rotation_deg=config.max_rotation_deg, max_translation=config.max_translation, dtype=pc1.dtype).to(config.device)
            pc2 = apply_tfm(pc1, random_rigid)

            pc1_tfm, T, T_deltas = it_net(pc1)
            pc2_tfm, T, T_deltas = it_net(pc2)

            val_loss += criterion(pc1_tfm, pc2_tfm).item()
            step += 1

    return val_loss / step

def train(config):
    train_dl, val_dl, test_dl = get_train_test_dls(config.batch_size)

    it_net = ITNet(channel=3, num_iters=5).to(config.device)

    if config.load == "True":
        checkpoint = torch.load(os.path.join(config.save_dir, "checkpoint.pth"))
        it_net.load_state_dict(checkpoint["model"])
        
        lr = checkpoint["lr"]
        optimizer = torch.optim.AdamW(it_net.parameters(), lr=lr, weight_decay=config.wd)
        optimizer.load_state_dict(checkpoint["optimizer"])

        step = checkpoint["step"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        progress_df = pd.read_csv(os.path.join(config.save_dir, "checkpoint_progress.csv"))
        progress_dict = progress_df.to_dict(orient='list')

        all_training_loss = checkpoint["all_training_loss"]

        print(f"Loaded: step={step}, lr={lr}, best_val_loss={best_val_loss}")

    else:
        lr = config.lr
        optimizer = torch.optim.AdamW(it_net.parameters(), lr=lr, weight_decay=config.wd)
        progress_dict = {
            "step": list(),
            "train_loss": list(),
            "val_loss": list()
        }
        step = 0
        best_val_loss = 5e5
        all_training_loss = list()

    it_net.train()
    criterion = PLoss()

    while step < config.num_steps:
        pbar = tqdm(train_dl, desc="Training")

        for batch in pbar:
            optimizer.zero_grad()
            
            pointcloud = batch["pointcloud"].to(torch.float32).to(config.device)

            pc1 = pointcloud.transpose(2, 1)
            # random_rigid = tfm_from_rand_pose(pc1.shape[0], config.translation_mean, config.translation_std).to(config.device)
            random_rigid = create_random_transform(pc1.shape[0], max_rotation_deg=config.max_rotation_deg, max_translation=config.max_translation, dtype=pc1.dtype).to(config.device)
            pc2 = apply_tfm(pc1, random_rigid)

            # pointcloud = apply_jitter(pointcloud)
            # pointcloud = pointcloud.transpose(2, 1)

            # T1 = create_random_transform(pointcloud.shape[0], max_rotation_deg=config.max_rotation_deg, max_translation=config.max_translation, dtype=pointcloud.dtype).to(config.device)
            # T2 = create_random_transform(pointcloud.shape[0], max_rotation_deg=config.max_rotation_deg, max_translation=config.max_translation, dtype=pointcloud.dtype).to(config.device)

            # pc1 = apply_tfm(pointcloud, T1)
            # pc2 = apply_tfm(pointcloud, T2)

            pc1_tfm, T, T_deltas = it_net(pc1)
            pc2_tfm, T, T_deltas = it_net(pc2)

            loss = criterion(pc1_tfm, pc2_tfm)
            loss.backward()
            optimizer.step()

            all_training_loss.append(loss.item())

            if step % config.eval_every == 0:
                avg_train_loss = all_training_loss[-config.eval_every:]
                avg_train_loss = sum(avg_train_loss) / len(avg_train_loss)

                it_net = it_net.eval()
                val_loss = evaluate(it_net, val_dl, criterion, config)
                it_net = it_net.train()

                progress_dict["step"].append(step)
                progress_dict["train_loss"].append(avg_train_loss)
                progress_dict["val_loss"].append(val_loss)

                progress_df = pd.DataFrame(progress_dict)
                progress_df.to_csv(os.path.join(config.save_dir, "progress.csv"), index=False)

                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        "model": it_net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr": lr,
                        "best_val_loss": best_val_loss,
                        "step": step,
                        "all_training_loss": all_training_loss
                    }
                    torch.save(checkpoint, os.path.join(config.save_dir, "checkpoint.pth"))

                    progress_df = pd.DataFrame(progress_dict)
                    progress_df.to_csv(os.path.join(config.save_dir, "checkpoint_progress.csv"), index=False)

            if step % config.lr_decay_every == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= config.lr_decay
                lr *= config.lr_decay
            
            step += 1
          
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=1e-1)
    parser.add_argument("--translation_mean", type=float, default=0)
    parser.add_argument("--translation_std", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--lr_decay_every", type=int, default=2000)
    parser.add_argument("--lr_decay", type=float, default=0.7)
    parser.add_argument("--max_rotation_deg", type=float, default=30)
    parser.add_argument("--max_translation", type=float, default=1)
    parser.add_argument("--save_dir", type=str, default="results/it_net")
    parser.add_argument("--load", type=str, default="False")
    config = parser.parse_args()

    train(config)

