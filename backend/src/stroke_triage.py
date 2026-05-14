import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from src.rsna_dataset import load_dicom_as_float, SUBTYPES
from src.rsna_model import create_model as create_rsna_model
from src.isles_infer_case import infer_case as isles_infer_case


#RSNA

def load_rsna_model(ckpt_path, device="cuda"):
    model = create_rsna_model(device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict_hemorrhage_exam(
    dicom_dir,
    ckpt_path="outputs/rsna_models/rsna_hemorrhage_best.pth",
    image_size=512,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_rsna_model(ckpt_path, device=device)

    dicom_dir = Path(dicom_dir)
    files = sorted(dicom_dir.glob("*.dcm"))
    assert files, f"No DICOM files found in {dicom_dir}"

    import cv2
    all_probs = []

    with torch.no_grad():
        for p in tqdm(files, desc=f"Infer {dicom_dir.name}", ncols=100):
            img = load_dicom_as_float(str(p))  
            img = cv2.resize(img, (image_size, image_size),
                             interpolation=cv2.INTER_LINEAR)
            img3 = np.stack([img, img, img], axis=0) 
            x = torch.from_numpy(img3).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            all_probs.append(probs)

    all_probs = np.stack(all_probs, axis=0)
    exam_probs = all_probs.max(axis=0)

    subtype_dict = {name: float(exam_probs[i])
                    for i, name in enumerate(SUBTYPES)}
    meta = {"num_slices": len(files)}

    return subtype_dict, meta


#TRIAGE

def triage_decision(hem_probs, isc_volume_ml):
    any_p = hem_probs.get("any", 0.0)
    ivh = hem_probs.get("intraventricular", 0.0)
    iph = hem_probs.get("intraparenchymal", 0.0)

    hem_present = any_p >= 0.30
    isc_present = isc_volume_ml >= 1.0

    if not hem_present and not isc_present:
        stroke_type = "none"
    elif hem_present and not isc_present:
        stroke_type = "hemorrhagic"
    elif isc_present and not hem_present:
        stroke_type = "ischemic"
    else:
        stroke_type = "mixed/uncertain"

    score = 0
    reasons = []

    if hem_present:
        score += 2
        reasons.append(f"Hemorrhage probability(any)={any_p:.2f}")
        if ivh >= 0.30:
            score += 2
            reasons.append(f"High IVH probability={ivh:.2f}")
        if iph >= 0.30:
            score += 2
            reasons.append(f"High IPH probability={iph:.2f}")

    if isc_present:
        reasons.append(f"Ischemic lesion volume={isc_volume_ml:.2f} ml")
        if isc_volume_ml >= 70:
            score += 3
            reasons.append("Large ischemic core (>=70 ml)")
        elif isc_volume_ml >= 30:
            score += 2
            reasons.append("Moderate ischemic core (>=30 ml)")
        elif isc_volume_ml >= 5:
            score += 1
            reasons.append("Small ischemic lesion (>=5 ml)")
        else:
            reasons.append("Very small lesion (possible noise)")

    if score >= 5:
        severity = "high"
    elif score >= 2:
        severity = "moderate"
    else:
        severity = "low"

    borderline = (0.25 <= any_p <= 0.45) or (0.8 <= isc_volume_ml <= 2.0)

    if severity == "high":
        review_priority = "urgent"
    elif borderline:
        review_priority = "needs_radiologist_confirmation"
    else:
        review_priority = "standard"

    return {
        "stroke_present": bool(hem_present or isc_present),
        "stroke_type": stroke_type,
        "severity": severity,
        "review_priority": review_priority,
        "reasons": reasons,
        "confidence_summary": {
            "hem_any": float(any_p),
            "isc_volume_ml": float(isc_volume_ml),
            "borderline": bool(borderline),
        },
        "hemorrhage_present": hem_present,
        "ischemia_present": isc_present,
    }


#MAIN

def run_triage(
    ct_dir=None,
    mri_dwi=None,
    mri_adc=None,
    mri_flair=None,
    rsna_ckpt="outputs/rsna_models/rsna_hemorrhage_best.pth",
    isles_ckpt="outputs/isles_models/isles_unet_cached_best.pth",
):
    #Hemorrhage (CT)
    if ct_dir:
        hem_subtypes, hem_meta = predict_hemorrhage_exam(
            dicom_dir=ct_dir, ckpt_path=rsna_ckpt, image_size=512
        )
    else:
        hem_subtypes = {name: 0.0 for name in SUBTYPES}
        hem_meta = {"num_slices": 0}

    #Ischemia (MRI)
    if mri_dwi and mri_adc and mri_flair:
        isc_result = isles_infer_case(
            dwi_path=mri_dwi,
            adc_path=mri_adc,
            flair_path=mri_flair,
            ckpt_path=isles_ckpt,
            out_dir="outputs/isles_pred",
            patch_size=(128, 128, 128),
        )
        isc_volume_ml = float(isc_result["lesion_volume_ml"])
    else:
        isc_result = {"mask_path": None,
                      "lesion_voxels": 0, "lesion_volume_ml": 0.0}
        isc_volume_ml = 0.0

    decision = triage_decision(hem_subtypes, isc_volume_ml)

    result = {
        **decision,
        "hemorrhage": {
            "subtypes": hem_subtypes,
            "meta": hem_meta,
        },
        "ischemia": {
            "lesion_volume_ml": isc_volume_ml,
            "mask_path": isc_result.get("mask_path"),
            "lesion_voxels": isc_result.get("lesion_voxels"),
        },
    }

    print(json.dumps(result, indent=2))
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ct_dir", type=str, default=None)
    parser.add_argument("--mri_dwi", type=str, default=None)
    parser.add_argument("--mri_adc", type=str, default=None)
    parser.add_argument("--mri_flair", type=str, default=None)
    parser.add_argument("--rsna_ckpt", type=str,
                        default="outputs/rsna_models/rsna_hemorrhage_best.pth")
    parser.add_argument("--isles_ckpt", type=str,
                        default="outputs/isles_models/isles_unet_cached_best.pth")

    args = parser.parse_args()

    run_triage(
        ct_dir=args.ct_dir,
        mri_dwi=args.mri_dwi,
        mri_adc=args.mri_adc,
        mri_flair=args.mri_flair,
        rsna_ckpt=args.rsna_ckpt,
        isles_ckpt=args.isles_ckpt,
    )


if __name__ == "__main__":
    main()
