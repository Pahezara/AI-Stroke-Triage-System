#src/rsna_calibrate_any.py

import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from tqdm import tqdm

from rsna_dataset import RSNADataset
from rsna_model import create_model


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_ds = RSNADataset("data/rsna/rsna_val.csv",
                         augment=False, image_size=512)
    loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    #Load model
    model = create_model(device=device)
    ckpt = torch.load(
        "outputs/rsna_models/rsna_hemorrhage_best.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"]
                          if "model_state_dict" in ckpt else ckpt)
    model.eval()

    logits_list = []
    y_list = []

    print("Collecting logits for validation set (this may take a few minutes)...")
    t0 = time.time()

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Calib logits", ncols=100):
            x = x.to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
            logits_list.append(logits[:, 0].detach().cpu())  # 'any'
            y_list.append(y[:, 0].detach().cpu())

    logits_any = torch.cat(logits_list).to(device)
    y_true = torch.cat(y_list).numpy().astype(np.float32)
    y_true_t = torch.tensor(y_true, device=device, dtype=torch.float32)

    t1 = time.time()
    print(f"Logits collected in {t1 - t0:.1f}s. Optimizing temperature...")

    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = bce(scaler(logits_any), y_true_t)
        loss.backward()
        return loss

    optimizer.step(closure)

    T = float(scaler.temperature.detach().cpu().item())

    with torch.no_grad():
        probs_cal = torch.sigmoid(logits_any / T).cpu().numpy()

    ll = log_loss(y_true, probs_cal)

    out = {"temperature_any": T, "val_logloss_any": float(ll)}

    out_path = Path("outputs/rsna_temperature.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    t2 = time.time()
    print(f"Done in {t2 - t0:.1f}s total.")
    print("Saved:", out_path)
    print(out)


if __name__ == "__main__":
    main()
