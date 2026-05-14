#src/rsna_infer_exam.py

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

from rsna_model import create_model
from rsna_dataset import load_dicom_as_float, SUBTYPES

torch.backends.cudnn.benchmark = True 


class ExamSliceDataset(Dataset):
    """
    Dataset for a single exam (folder of DICOM slices).
    Loads & preprocesses one slice at a time.
    """

    def __init__(self, dicom_dir, image_size=512):
        self.dicom_dir = Path(dicom_dir)
        self.files = sorted(self.dicom_dir.glob("*.dcm"))
        if not self.files:
            raise RuntimeError(f"No DICOM files found in {self.dicom_dir}")
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = load_dicom_as_float(str(p))  # H x W in [0,1]
        img = cv2.resize(img, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_LINEAR)
        img3 = np.stack([img, img, img], axis=0).astype(np.float32)  # C,H,W
        return torch.from_numpy(img3)


def load_model(ckpt_path, device="cuda"):
    model = create_model(device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_exam(
    dicom_dir,
    ckpt_path,
    image_size=512,
    batch_size=32,
    num_workers=4,
    device="cuda",
):
    """
    Fast batched inference over all slices in an exam.
    Memory-safe: only keeps one batch of images on GPU/RAM at a time.
    """
    dicom_dir = Path(dicom_dir)
    assert dicom_dir.is_dir(), f"{dicom_dir} is not a directory"

    #Model
    model = load_model(ckpt_path, device=device)

    #Dataset & DataLoader
    ds = ExamSliceDataset(dicom_dir, image_size=image_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_probs = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Infer {dicom_dir.name}", ncols=100)
        for imgs in pbar:
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(imgs)
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0) 

    #Simple aggregation
    exam_probs = all_probs.max(axis=0)

    result = {
        "stroke_present": bool(exam_probs[0] > 0.3),
        "hemorrhage": {
            "subtypes": {
                st: float(exam_probs[i])
                for i, st in enumerate(SUBTYPES)
            }
        },
        "meta": {
            "num_slices": int(all_probs.shape[0]),
        },
    }
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dicom_dir",
        type=str,
        required=True,
        help="Folder with DICOM slices of a single exam/patient",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/rsna_models/rsna_hemorrhage_best.pth",
        help="Path to trained checkpoint",
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = predict_exam(
        dicom_dir=args.dicom_dir,
        ckpt_path=args.ckpt,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    print("\n=== Exam-level prediction ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
