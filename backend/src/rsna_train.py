#src/rsna_train.py

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from rsna_dataset import RSNADataset, SUBTYPES
from rsna_model import create_model, NUM_CLASSES


def compute_auc(y_true, y_pred):
    """
    y_true: (N, C) numpy
    y_pred: (N, C) numpy
    returns dict of per-class AUC + mean AUC
    """
    aucs = {}
    for i, name in enumerate(SUBTYPES):
        try:
            aucs[name] = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            aucs[name] = np.nan
    valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
    aucs["mean_auc"] = float(np.mean(valid_aucs)) if valid_aucs else np.nan
    return aucs


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path("data/rsna")
    train_csv = data_dir / "rsna_train.csv"
    val_csv = data_dir / "rsna_val.csv"

    out_dir = Path("outputs/rsna_models")
    out_dir.mkdir(parents=True, exist_ok=True)

    #Tuned for RTX 3080 12GB
    image_size = 512
    batch_size = 32
    num_epochs = 10
    lr = 1e-4

    #Datasets
    train_ds = RSNADataset(train_csv, augment=True, image_size=image_size)
    val_ds = RSNADataset(val_csv, augment=False, image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    #Model
    model = create_model(device=device)

    pos_weight = torch.tensor(
        [5, 10, 5, 8, 5, 5], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_auc = -np.inf
    best_ckpt_path = out_dir / "rsna_hemorrhage_best.pth"

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        #TRAIN
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc="Train", ncols=100)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_ds)
        print(f"Train loss: {train_loss:.4f}")

        #VALIDATE
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Val  ", ncols=100)
            for imgs, labels in pbar:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(logits)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(probs.cpu().numpy())

        val_loss /= len(val_ds)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        aucs = compute_auc(all_labels, all_preds)

        print(f"Val loss: {val_loss:.4f}")
        for k, v in aucs.items():
            print(f"AUC[{k}]: {v:.4f}")

        if aucs["mean_auc"] > best_val_auc:
            best_val_auc = aucs["mean_auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_auc": aucs,
                },
                best_ckpt_path,
            )
            print(
                f"New best mean AUC {best_val_auc:.4f}, saved to {best_ckpt_path}")

    print("Training complete.")
    print(f"Best mean AUC: {best_val_auc:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    train()
