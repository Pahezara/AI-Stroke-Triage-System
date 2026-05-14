#src/gradcam_utils.py

from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.rsna_model import create_model as create_rsna_model
from src.rsna_dataset import load_dicom_as_float

GRADCAM_DIR = Path("outputs/rsna_gradcam")


def _load_rsna_model(ckpt_path: str, device: str = "cuda"):
    model = create_rsna_model(device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def generate_ct_gradcam(
    dicom_path: str,
    ckpt_path: str = "outputs/rsna_models/rsna_hemorrhage_best.pth",
    class_idx: int = 0,
    image_size: int = 512,
) -> str:
    """
    Generate a Grad-CAM heatmap for a single CT DICOM slice.

    Returns a relative path like 'rsna_gradcam/ID_xxx_c0.png'
    which can be served via FastAPI static files.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dicom_path = Path(dicom_path)
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM not found: {dicom_path}")

    model = _load_rsna_model(ckpt_path, device=device)

    target_layers = [model.backbone.conv_head]

    #Prepare image
    img = load_dicom_as_float(str(dicom_path))
    img = cv2.resize(img, (image_size, image_size),
                     interpolation=cv2.INTER_LINEAR)
    rgb_img = np.stack([img, img, img], axis=-1).astype(np.float32)  #H,W,3

    input_tensor = (
        torch.from_numpy(rgb_img.transpose(2, 0, 1))
        .unsqueeze(0)
        .to(device)
    )

    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  #H,W

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{dicom_path.stem}_c{class_idx}.png"
    out_rel = Path("rsna_gradcam") / out_name
    out_path = GRADCAM_DIR / out_name

    #Save as PNG
    cv2.imwrite(str(out_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    return str(out_rel).replace("\\", "/")
