#src/isles_eval_dice_hd95.py

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import UNet

#Silence torch.load future warning noise (not an error)
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r"You are using `torch\.load`.*")


def load_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class IslesCachedVal(Dataset):
    """
    Loads cached tensors created by isles_precompute_cache.py:
      data/isles/cache_pt/<case_id>.pt
    Each .pt has:
      image: (C,H,W,D) float16/float32
      label: (1,H,W,D) uint8/long (0/1)
    """

    def __init__(self, items, cache_dir: str):
        self.items = items
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        case_id = self.items[idx]["case_id"]
        p = self.cache_dir / f"{case_id}.pt"
        pack = torch.load(p, map_location="cpu", weights_only=False)

        image = pack["image"].to(torch.float32)              #(C,H,W,D)
        label = pack["label"].to(torch.int64)                #(1,H,W,D) 0/1
        return {"case_id": case_id, "image": image, "label": label}


def build_net(device):
    net = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return net


def to_onehot_2(y01: torch.Tensor) -> torch.Tensor:
    """
    y01: (B,1,H,W,D) with values 0/1
    returns one-hot: (B,2,H,W,D)
    """
    y01 = y01.squeeze(1).long()  #(B,H,W,D)
    oh = F.one_hot(y01, num_classes=2)  #(B,H,W,D,2)
    oh = oh.permute(0, 4, 1, 2, 3).contiguous()  #(B,2,H,W,D)
    return oh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_list", type=str,
                        default="data/isles/isles_val_list.json")
    parser.add_argument("--cache_dir", type=str, default="data/isles/cache_pt")
    parser.add_argument(
        "--ckpt", type=str, default="outputs/isles_models/isles_unet_cached_best.pth")
    parser.add_argument("--roi", type=int, nargs=3,
                        default=(80, 80, 80), help="Sliding window ROI size")
    parser.add_argument("--sw_batch", type=int, default=4,
                        help="Sliding window batch size (reduce if OOM)")
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_items = load_list(args.val_list)
    ds = IslesCachedVal(val_items, args.cache_dir)

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=True)

    net = build_net(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95, reduction="none")

    dices = []
    hd95s = []
    case_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", ncols=100):
            case_id = batch["case_id"][0]
            x = batch["image"].to(device, non_blocking=True)     #(1,C,H,W,D)
            y = batch["label"].to(device, non_blocking=True)     #(1,1,H,W,D)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = sliding_window_inference(
                    x,
                    roi_size=tuple(args.roi),
                    sw_batch_size=args.sw_batch,
                    predictor=net,
                    overlap=args.overlap,
                )
                pred = torch.argmax(torch.softmax(
                    logits, dim=1), dim=1, keepdim=True)  # (1,1,H,W,D)

            #convert to one-hot with 2 channels
            pred_oh = to_onehot_2(pred)
            y_oh = to_onehot_2(y)

            d = dice_metric(pred_oh, y_oh)   #(B, foreground_channels)
            h = hd95_metric(pred_oh, y_oh)

            #take mean across foreground channels
            d_val = float(torch.nanmean(d).item())
            h_val = float(torch.nanmean(h).item())

            dices.append(d_val)
            hd95s.append(h_val)
            case_ids.append(case_id)

    dices = np.array(dices, dtype=np.float64)
    hd95s = np.array(hd95s, dtype=np.float64)

    print("\n=== ISLES Validation Metrics (Cached) ===")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Cache dir : {args.cache_dir}")
    print(
        f"ROI       : {args.roi}, SW batch: {args.sw_batch}, overlap: {args.overlap}")
    print(f"Mean Dice : {np.nanmean(dices):.4f} ± {np.nanstd(dices):.4f}")
    print(f"Mean HD95 : {np.nanmean(hd95s):.4f} ± {np.nanstd(hd95s):.4f}")

    #Opt: print worst 5 cases
    worst_idx = np.argsort(dices)[:5]
    print("\nWorst 5 Dice cases:")
    for i in worst_idx:
        print(f"  {case_ids[i]}  Dice={dices[i]:.4f}  HD95={hd95s[i]:.4f}")


if __name__ == "__main__":
    main()
