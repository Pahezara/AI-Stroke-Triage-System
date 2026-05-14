#src/rsna_gradcam.py

import torch
import numpy as np
import cv2
from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from rsna_model import create_model
from rsna_dataset import load_dicom_as_float


def load_model(ckpt_path, device="cuda"):
    model = create_model(device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def generate_gradcam(dicom_path, ckpt_path, class_idx=0, image_size=512, device="cuda"):
    dicom_path = Path(dicom_path)
    assert dicom_path.exists(), f"{dicom_path} not found"

    #Load model
    model = load_model(ckpt_path, device=device)

    target_layers = [model.backbone.conv_head]  

    cam = GradCAM(model=model, target_layers=target_layers)

    #Prepare input
    img = load_dicom_as_float(str(dicom_path))
    img = cv2.resize(img, (image_size, image_size),
                     interpolation=cv2.INTER_LINEAR)

    rgb_img = np.stack([img, img, img], axis=-1)  # H,W,3 in [0,1]
    rgb_img_float = rgb_img.astype(np.float32)

    input_tensor = torch.from_numpy(
        rgb_img_float.transpose(2, 0, 1)).unsqueeze(0).to(device)

    #Run Grad-CAM
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # H,W

    #Overlay heatmap
    visualization = show_cam_on_image(
        rgb_img_float, grayscale_cam, use_rgb=True)

    out_dir = Path("outputs/rsna_gradcam")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dicom_path.stem}_gradcam_class{class_idx}.png"

    cv2.imwrite(str(out_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM visualization to: {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom", type=str, required=True,
                        help="Path to DICOM slice")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/rsna_models/rsna_hemorrhage_best.pth",
    )
    parser.add_argument(
        "--class_idx",
        type=int,
        default=0,
        help="0:any,1:epidural,2:intraparenchymal,3:intraventricular,4:subarachnoid,5:subdural",
    )
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generate_gradcam(
        dicom_path=args.dicom,
        ckpt_path=args.ckpt,
        class_idx=args.class_idx,
        image_size=args.image_size,
        device=device,
    )


if __name__ == "__main__":
    main()
