#src/isles_infer_case.py

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ResizeWithPadOrCropd,
    NormalizeIntensityd,
    ConcatItemsd,
    EnsureTyped,
)


#MODEL LOAD

def build_net(device="cuda"):
    net = UNet(
        spatial_dims=3,
        in_channels=3,   #DWI, ADC, FLAIR
        out_channels=2,  #background, lesion
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return net


def load_model(ckpt_path, device="cuda"):
    net = build_net(device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


#TRANSFORMS

def get_case_transforms(patch_size=(128, 128, 128)):
    """
    Same preprocessing as your val_transforms in isles_train.py,
    but only for dwi/adc/flair (no label).
    """
    return Compose(
        [
            LoadImaged(keys=["dwi", "adc", "flair"]),
            EnsureChannelFirstd(keys=["dwi", "adc", "flair"]),
            Spacingd(
                keys=["dwi", "adc", "flair"],
                pixdim=(1.5, 1.5, 1.5),
                mode=("bilinear", "bilinear", "bilinear"),
            ),
            Orientationd(keys=["dwi", "adc", "flair"], axcodes="RAS"),
            ResizeWithPadOrCropd(
                keys=["dwi", "adc", "flair"],
                spatial_size=patch_size,
            ),
            NormalizeIntensityd(
                keys=["dwi", "adc", "flair"],
                nonzero=True,
                channel_wise=True,
            ),
            ConcatItemsd(keys=["dwi", "adc", "flair"], name="image", dim=0),
            EnsureTyped(keys=["image"]),
        ]
    )


#INFERENCE CORE

def infer_case(
    dwi_path,
    adc_path,
    flair_path,
    ckpt_path="outputs/isles_models/isles_unet_best.pth",
    out_dir="outputs/isles_pred",
    patch_size=(128, 128, 128),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model(ckpt_path, device=device)
    transforms = get_case_transforms(patch_size=patch_size)

    dwi_path = Path(dwi_path)
    adc_path = Path(adc_path)
    flair_path = Path(flair_path)
    assert dwi_path.exists(), f"{dwi_path} not found"
    assert adc_path.exists(), f"{adc_path} not found"
    assert flair_path.exists(), f"{flair_path} not found"

    case_dict = {
        "dwi": str(dwi_path),
        "adc": str(adc_path),
        "flair": str(flair_path),
    }

    data = transforms(case_dict)
    #(1, 3, D, H, W) = (1,3,128,128,128)
    image = data["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = net(image)
            probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)  #(1, D, H, W)
        pred_mask = preds.cpu().numpy().astype(np.uint8)[0]  #0/1

    voxel_mm3 = 1.5 * 1.5 * 1.5
    lesion_voxels = int(pred_mask.sum())
    lesion_volume_ml = lesion_voxels * voxel_mm3 / 1000.0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4, dtype=np.float32)
    out_nii = nib.Nifti1Image(pred_mask, affine=affine)
    stem = dwi_path.stem.replace("_dwi", "")
    out_path = out_dir / f"{stem}_pred_msk.nii.gz"
    nib.save(out_nii, str(out_path))

    print(f"Saved predicted lesion mask to: {out_path}")
    print(f"Lesion voxels: {lesion_voxels}")
    print(f"Estimated lesion volume: {lesion_volume_ml:.2f} ml")

    return {
        "mask_path": str(out_path),
        "lesion_voxels": lesion_voxels,
        "lesion_volume_ml": lesion_volume_ml,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dwi", type=str, required=True,
                        help="Path to *_dwi.nii.gz")
    parser.add_argument("--adc", type=str, required=True,
                        help="Path to *_adc.nii.gz")
    parser.add_argument("--flair", type=str, required=True,
                        help="Path to *_FLAIR.nii.gz")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/isles_models/isles_unet_best.pth",
        help="Path to trained ISLES checkpoint",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/isles_pred",
        help="Output folder for predicted masks",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        help="Target size used in training, e.g. 128 128 128",
    )

    args = parser.parse_args()

    infer_case(
        dwi_path=args.dwi,
        adc_path=args.adc,
        flair_path=args.flair,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        patch_size=tuple(args.patch_size),
    )


if __name__ == "__main__":
    main()
