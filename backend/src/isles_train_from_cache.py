#src/isles_train_from_cache.py

import json
from pathlib import Path
import random
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from monai.data.utils import list_data_collate
from monai.transforms import Compose, RandCropByPosNegLabeld, RandFlipd, RandRotate90d
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class IslesCachedPTDataset(Dataset):
    """
    Loads precomputed .pt files (image + label) and applies MONAI transforms per-sample.
    This avoids applying RandCropByPosNegLabeld on a batch (which breaks).
    """

    def __init__(self, items, cache_dir, transform=None):
        self.items = items
        self.cache_dir = Path(cache_dir)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        case_id = self.items[idx]["case_id"]
        p = self.cache_dir / f"{case_id}.pt"

        pack = torch.load(p, map_location="cpu", weights_only=False)

        image = pack["image"].to(torch.float32)      #(C,H,W,D) C=3
        label = pack["label"].to(torch.int64)        #(1,H,W,D) 0/1
        image1 = image[0:1, ...]                     #(1,H,W,D)

        data = {
            "case_id": case_id,
            "image": image,
            "label": label,
            "image1": image1, 
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


def main():
    set_seed(42)
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = "data/isles/cache_pt"
    train_items = load_list("data/isles/isles_train_list.json")
    val_items = load_list("data/isles/isles_val_list.json")

    PATCH = (80, 80, 80)
    NUM_SAMPLES = 2 
    BATCH_SIZE = 2
    EPOCHS = 80
    VAL_EVERY = 5
    SW_BATCH = 4

    train_tf = Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH,
            pos=2,
            neg=1,
            num_samples=NUM_SAMPLES,
            image_key="image1",
            allow_smaller=True,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    ])

    train_ds = IslesCachedPTDataset(train_items, cache_dir, transform=train_tf)
    val_ds = IslesCachedPTDataset(val_items, cache_dir, transform=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=list_data_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    #Model
    net = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    dice = DiceMetric(include_background=False, reduction="mean")

    out_dir = Path("outputs/isles_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "isles_unet_cached_best.pth"
    best = -1.0

    for epoch in range(1, EPOCHS + 1):
        #Train
        net.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{EPOCHS}", ncols=100)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = net(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        print("Train loss:", running / max(1, len(train_loader)))

        #Validate
        if epoch % VAL_EVERY != 0 and epoch != EPOCHS:
            continue

        net.eval()
        dice.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", ncols=100):
                x = batch["image"].to(device, non_blocking=True) 
                y = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = sliding_window_inference(
                        x, roi_size=PATCH, sw_batch_size=SW_BATCH, predictor=net, overlap=0.5
                    )
                    preds = torch.argmax(torch.softmax(logits, 1), 1, keepdim=True)

                dice(preds, y)

        d = float(dice.aggregate().item())
        print(f"Val Dice: {d:.4f}")

        if d > best:
            best = d
            torch.save({"epoch": epoch, "model_state_dict": net.state_dict(), "val_dice": best}, best_path)
            print("Saved best:", best_path)

    print("Done. Best Dice:", best)


if __name__ == "__main__":
    main()