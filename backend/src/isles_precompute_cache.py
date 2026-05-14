import json
from pathlib import Path

import torch
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    NormalizeIntensityd, CropForegroundd, ResizeWithPadOrCropd,
    ConcatItemsd, EnsureTyped
)
from monai.data import Dataset, DataLoader


def load_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    #I used 128^3 as standard size.
    STANDARD_SIZE = (128, 128, 128)
    PIXDIM = (1.5, 1.5, 1.5)

    train_list = load_list("data/isles/isles_train_list.json")
    val_list = load_list("data/isles/isles_val_list.json")

    all_items = train_list + val_list

    cache_dir = Path("data/isles/cache_pt")
    cache_dir.mkdir(parents=True, exist_ok=True)

    tf = Compose([
        LoadImaged(keys=["dwi", "adc", "flair", "label"]),
        EnsureChannelFirstd(keys=["dwi", "adc", "flair", "label"]),
        Spacingd(keys=["dwi", "adc", "flair", "label"], pixdim=PIXDIM,
                 mode=("bilinear", "bilinear", "bilinear", "nearest")),
        Orientationd(keys=["dwi", "adc", "flair", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["dwi", "adc", "flair"],
                            nonzero=True, channel_wise=True),
        CropForegroundd(keys=["dwi", "adc", "flair", "label"],
                        source_key="dwi", allow_smaller=True),
        ResizeWithPadOrCropd(
            keys=["dwi", "adc", "flair", "label"], spatial_size=STANDARD_SIZE),
        ConcatItemsd(keys=["dwi", "adc", "flair"], name="image", dim=0),
        EnsureTyped(keys=["image", "label"]),
    ])

    ds = Dataset(all_items, transform=tf)

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for item, batch in tqdm(zip(all_items, loader), total=len(all_items), desc="Caching", ncols=100):
        case_id = item["case_id"]
        out_path = cache_dir / f"{case_id}.pt"

        #image: (1,C,H,W,D) -> (C,H,W,D)
        image = batch["image"][0].contiguous()
        label = batch["label"][0].contiguous()  #(1,H,W,D)

        #Save compact: image float16, label uint8
        pack = {
            "case_id": case_id,
            "image": image.to(torch.float16).cpu(),
            "label": (label > 0).to(torch.uint8).cpu(),
            "pixdim": PIXDIM,
            "standard_size": STANDARD_SIZE,
        }
        torch.save(pack, out_path)

    print(f"\nDone. Cached {len(all_items)} cases to: {cache_dir}")


if __name__ == "__main__":
    main()
