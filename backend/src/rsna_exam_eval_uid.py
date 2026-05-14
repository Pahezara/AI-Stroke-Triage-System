#src/rsna_exam_eval_uid.py

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pydicom
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from rsna_model import create_model
from rsna_dataset import load_dicom_as_float


def get_study_uid(dicom_path: str) -> str:
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    uid = getattr(ds, "StudyInstanceUID", None)
    if uid is None:
        uid = getattr(ds, "SeriesInstanceUID", "UNKNOWN_UID")
    return str(uid)


def load_model(ckpt_path: str, device: str):
    model = create_model(device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_csv = Path("data/rsna/rsna_val.csv")
    ckpt_path = Path("outputs/rsna_models/rsna_hemorrhage_best.pth")
    assert val_csv.exists(), "Missing data/rsna/rsna_val.csv"
    assert ckpt_path.exists(), "Missing RSNA checkpoint"

    df = pd.read_csv(val_csv)

    #group by StudyInstanceUID
    print("Reading StudyInstanceUIDs (this takes a bit)...")
    study_uids = []
    for fp in tqdm(df["filepath"].tolist(), desc="UID scan", ncols=100):
        study_uids.append(get_study_uid(fp))
    df["study_uid"] = study_uids

    #Load model once
    model = load_model(str(ckpt_path), device=device)

    #Inference per slice, store any_prob + label_any
    any_probs = []
    for fp in tqdm(df["filepath"].tolist(), desc="Slice infer", ncols=100):
        img = load_dicom_as_float(fp)  
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(
            np.stack([img, img, img], 0)).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
            prob_any = torch.sigmoid(logits)[0, 0].item()
        any_probs.append(prob_any)

    df["prob_any"] = any_probs

    #Build exam-level labels and predictions
    #Label = max(any label per slice in exam)
    #Pred = max(prob_any per slice in exam)
    exam = df.groupby("study_uid").agg(
        y_true_any=("any", "max"),
        y_score_any=("prob_any", "max"),
        n_slices=("prob_any", "count"),
    ).reset_index()

    y_true = exam["y_true_any"].values.astype(int)
    y_score = exam["y_score_any"].values.astype(float)

    #metrics
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    print("\n=== RSNA Exam-level metrics (StudyInstanceUID grouping) ===")
    print(f"Num exams: {len(exam)}")
    print(f"Exam AUC(any):   {auc:.4f}")
    print(f"Exam PR-AUC(any): {ap:.4f}")

    #triage threshold
    thr = 0.30
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)

    print(f"\nThreshold: {thr:.2f}")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(cm)
    print(f"Sensitivity (recall pos): {sens:.4f}")
    print(f"Specificity:              {spec:.4f}")

    print("\nExam slice count stats:")
    print(exam["n_slices"].describe())


if __name__ == "__main__":
    main()
