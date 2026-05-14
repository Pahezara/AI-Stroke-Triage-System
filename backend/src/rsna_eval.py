#src/rsna_eval.py

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm

from rsna_dataset import RSNADataset, SUBTYPES
from rsna_model import create_model


def compute_auc(y_true, y_pred):
    """
    y_true: (N, C)
    y_pred: (N, C)
    """
    aucs = {}
    for i, name in enumerate(SUBTYPES):
        try:
            aucs[name] = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            aucs[name] = np.nan
    valid = [v for v in aucs.values() if not np.isnan(v)]
    aucs["mean_auc"] = float(np.mean(valid)) if valid else np.nan
    return aucs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path("data/rsna")
    val_csv = data_dir / "rsna_val.csv"
    ckpt_path = Path("outputs/rsna_models/rsna_hemorrhage_best.pth")

    assert val_csv.exists(
    ), f"{val_csv} not found (run rsna_build_meta.py & train)"
    assert ckpt_path.exists(
    ), f"{ckpt_path} not found (train RSNA model first)"

    #Dataset & loader
    val_ds = RSNADataset(csv_path=val_csv, augment=False, image_size=512)
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    #Load model
    model = create_model(device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_labels = []
    all_probs = []

    print("\nRunning RSNA validation inference...")
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="RSNA Val", ncols=100):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(imgs)
                probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    #AUCs
    aucs = compute_auc(all_labels, all_probs)
    print("\n=== RSNA Validation AUCs ===")
    for k, v in aucs.items():
        print(f"{k:16s}: {v:.4f}")

    all_preds_bin = (all_probs >= 0.5).astype(int)

    print("\n=== RSNA Slice-level Classification Report (threshold=0.5) ===")
    for i, name in enumerate(SUBTYPES):
        print(f"\n--- {name.upper()} ---")
        print(
            classification_report(
                all_labels[:, i],
                all_preds_bin[:, i],
                target_names=["neg", "pos"],
                digits=4,
                zero_division=0,
            )
        )


if __name__ == "__main__":
    main()
