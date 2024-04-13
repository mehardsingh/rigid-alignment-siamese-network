import torch
import torch.nn as nn
import sys
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import argparse

from lightly.loss.ntx_ent_loss import NTXentLoss

sys.path.append("src/get_modelnet40")
from load_data import get_train_test_dls, get_train_test_dls_cls

sys.path.append("src/pretrain_utils/")
from transforms import quaternion_to_matrix, apply_tfm, compose_tfms
from corruptions import tfm_from_rand_pose, create_random_transform

sys.path.append("src/it_net")
from it_net import ITNet

sys.path.append("src/pointnet")
from pointnet_utils import PointNetEncoder

sys.path.append("src/pretrain_utils")
from corruptions import apply_corruptions

sys.path.append("src/pointnet")
from pointnet_cls_head import ClsHead, ClsHeadProj
from projection import Proj

class Classifier(torch.nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.encoder = PointNetEncoder(
            global_feat=True, 
            init_transform=False, 
            feature_transform=False, 
            channel=3
        ).to(config.device)
        self.proj = Proj().to(config.device)
        self.head = ClsHeadProj().to(config.device)

        encoder_checkpoint = torch.load(os.path.join(config.encoder_dir, "checkpoint.pth"))
        self.encoder.load_state_dict(encoder_checkpoint["encoder"])
        self.proj.load_state_dict(encoder_checkpoint["projection"])

        for param in self.encoder.parameters():
            param.requires_grad = False

        # for param in self.proj.parameters():
        #     param.requires_grad = False

        self.encoder.eval()
        # self.proj.eval()


    def forward(self, x):
        x, _, _ = self.encoder(x)
        x = self.proj(x)
        x = self.head(x)

        return x
    
class IT_Classifier(torch.nn.Module):
    def __init__(self, config):
        super(IT_Classifier, self).__init__()

        self.it_net = ITNet(3, num_iters=2).to(config.device)
        it_net_checkpoint = torch.load(os.path.join(config.it_net_dir, "checkpoint.pth"))
        self.it_net.load_state_dict(it_net_checkpoint["model"])
        for param in self.it_net.parameters():
            param.requires_grad = False
        self.it_net.eval()
        
        self.encoder = PointNetEncoder(
            global_feat=True, 
            init_transform=False, 
            feature_transform=False, 
            channel=3
        ).to(config.device)
        self.proj = Proj().to(config.device)
        self.head = ClsHeadProj().to(config.device)

        encoder_checkpoint = torch.load(os.path.join(config.encoder_dir, "checkpoint.pth"))
        self.encoder.load_state_dict(encoder_checkpoint["encoder"])
        self.proj.load_state_dict(encoder_checkpoint["projection"])

        for param in self.encoder.parameters():
            param.requires_grad = False

        # for param in self.proj.parameters():
        #     param.requires_grad = False

        self.encoder.eval()
        # self.proj.eval()

    def forward(self, x):
        x, _, _ = self.it_net(x)
        x, _, _ = self.encoder(x)
        x = self.proj(x)
        x = self.head(x)

        return x

def evaluate(model, val_dl, criterion, config):
    val_loss = 0
    step = 0

    with torch.no_grad():
        pbar = tqdm(val_dl, desc="Validating")
        for batch in pbar:
            pointcloud = batch["pointcloud"].to(torch.float32).to(config.device)
            labels = batch["category"].to(config.device)
            pointcloud = pointcloud.transpose(2, 1)

            pred = model(pointcloud)
            # print("pred", pred)
            # print("labels", labels)
            curr_loss = criterion(pred, labels).item()
            # print("curr_loss", curr_loss)
            val_loss += curr_loss
            step += 1

    return val_loss / step

def train(config):
    train_dl, val_dl, test_dl = get_train_test_dls_cls(config.batch_size)

    if "compose" in config.encoder_dir:
        model = IT_Classifier(config)
    else:
        model = Classifier(config)

    if config.load == "True":
        checkpoint = torch.load(os.path.join(config.save_dir, "checkpoint.pth"))
        model.load_state_dict(checkpoint["model"])
        
        lr = checkpoint["lr"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.wd)
        optimizer.load_state_dict(checkpoint["optimizer"])

        step = checkpoint["step"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        progress_df = pd.read_csv(os.path.join(config.save_dir, "checkpoint_progress.csv"))
        progress_dict = progress_df.to_dict(orient='list')

        all_training_loss = checkpoint["all_training_loss"]

        print(f"Loaded: step={step}, lr={lr}, best_val_loss={best_val_loss}")

    else:
        lr = config.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.wd)
        progress_dict = {
            "step": list(),
            "train_loss": list(),
            "val_loss": list()
        }
        step = 0
        best_val_loss = 5e5
        all_training_loss = list()

    # model.encoder.train()
    model.proj.train()
    model.head.train()
    criterion = torch.nn.CrossEntropyLoss()

    while step < config.num_steps:
        pbar = tqdm(train_dl, desc="Training")

        for batch in pbar:
            optimizer.zero_grad()
            
            pointcloud = batch["pointcloud"].to(torch.float32).to(config.device)
            labels = batch["category"].to(config.device)
            pointcloud = pointcloud.transpose(2, 1)

            pred = model(pointcloud)
            loss = criterion(pred, labels)
            
            loss.backward()
            optimizer.step()

            all_training_loss.append(loss.item())

            if step % config.eval_every == 0:
                avg_train_loss = all_training_loss[-config.eval_every:]
                avg_train_loss = sum(avg_train_loss) / len(avg_train_loss)

                # model.encoder.eval()
                model.proj.eval()
                model.head.eval()
                val_loss = evaluate(model, val_dl, criterion, config)
                # model.encoder.train()
                model.proj.train()
                model.head.train()

                progress_dict["step"].append(step)
                progress_dict["train_loss"].append(avg_train_loss)
                progress_dict["val_loss"].append(val_loss)

                progress_df = pd.DataFrame(progress_dict)
                progress_df.to_csv(os.path.join(config.save_dir, "progress.csv"), index=False)

                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        "model": model.state_dict(),
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
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--translation_mean", type=float, default=0)
    parser.add_argument("--translation_std", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--lr_decay_every", type=int, default=2000)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--max_rotation_deg", type=float, default=30)
    parser.add_argument("--max_translation", type=float, default=1)
    parser.add_argument("--save_dir", type=str, default="results/classifiers/compose_contrast")
    parser.add_argument("--encoder_dir", type=str, default="results/compose_contrast")
    parser.add_argument("--it_net_dir", type=str, default="results/it_net_continue")
    parser.add_argument("--load", type=str, default="False")
    config = parser.parse_args()

    train(config)