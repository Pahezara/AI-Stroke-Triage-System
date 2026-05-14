#src/rsna_dataset.py

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
import cv2

SUBTYPES = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


def load_dicom_as_float(path, window_min=-100, window_max=200):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32, copy=False)

    slope = np.float32(getattr(ds, "RescaleSlope", 1.0))
    intercept = np.float32(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept  

    img = np.clip(img, window_min, window_max).astype(np.float32, copy=False)
    img = (img - np.float32(window_min)) / np.float32(window_max - window_min)
    return img.astype(np.float32, copy=False)


class RSNADataset(Dataset):
    def __init__(self, csv_path, augment=False, image_size=512, max_retry=5):
        """
        max_retry: how many times we try to resample a different index
                   if a DICOM is corrupted/bad.
        """
        self.df = pd.read_csv(csv_path)
        self.augment = augment
        self.image_size = image_size
        self.max_retry = max_retry

    def __len__(self):
        return len(self.df)

    def _augment(self, img):
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
        if np.random.rand() < 0.5:
            img = np.flipud(img)
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-10, 10)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(
                img,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        return img

    def _load_item(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]

        img = load_dicom_as_float(filepath)
        img = cv2.resize(img, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_LINEAR)

        if self.augment:
            img = self._augment(img)

        img3 = np.stack([img, img, img], axis=0)  # C,H,W
        labels = row[SUBTYPES].values.astype(np.float32)

        return torch.from_numpy(img3), torch.from_numpy(labels)

    def __getitem__(self, idx):
        """
        Try to load this index; if the DICOM is corrupted, print a warning and
        sample another random index, up to max_retry times.
        """
        for _ in range(self.max_retry):
            try:
                return self._load_item(idx)
            except Exception as e:
                bad_path = self.df.iloc[idx]["filepath"]
                print(f"[WARN] Skipping bad DICOM {bad_path}: {e}")
                idx = np.random.randint(0, len(self.df))

        raise RuntimeError("Too many consecutive bad DICOMs, aborting.")
