#src/api.py

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.stroke_triage import run_triage
from src.gradcam_utils import generate_ct_gradcam
from src.pdf_report import build_triage_pdf_bytes
from src.auth import AdminLoginRequest, login_admin, require_admin


app = FastAPI(
    title="Stroke AI Triage API",
    description="AI-powered stroke triage combining CT (RSNA) + MRI (ISLES).",
    version="0.3.0",
)

#CORS CONFIG

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#STATIC FILE SERVING

app.mount(
    "/static",
    StaticFiles(directory="outputs"),
    name="static",
)


#PUBLIC ROUTES

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Stroke AI Triage API",
        "version": "0.3.0",
    }


@app.post("/auth/login")
def auth_login(req: AdminLoginRequest):
    """
    Admin login endpoint.

    Default demo credentials are controlled in src/auth.py:
    username: admin
    password: admin123

    Change these through environment variables before final deployment.
    """
    return login_admin(req)


@app.get("/auth/me")
def auth_me(admin=Depends(require_admin)):
    """
    Verify current admin token.
    """
    return {
        "authenticated": True,
        "user": admin.get("sub"),
        "role": admin.get("role"),
        "exp": admin.get("exp"),
    }


#REQUEST MODELS

class TriageRequest(BaseModel):
    ct_dir: Optional[str] = None
    mri_dwi: Optional[str] = None
    mri_adc: Optional[str] = None
    mri_flair: Optional[str] = None

    rsna_ckpt: Optional[str] = "outputs/rsna_models/rsna_hemorrhage_best.pth"
    isles_ckpt: Optional[str] = "outputs/isles_models/isles_unet_cached_best.pth"


class CTGradCAMRequest(BaseModel):
    dicom_path: str
    class_idx: int = 0
    rsna_ckpt: Optional[str] = "outputs/rsna_models/rsna_hemorrhage_best.pth"


class TriagePdfRequest(BaseModel):
    ct_dir: Optional[str] = None
    mri_dwi: Optional[str] = None
    mri_adc: Optional[str] = None
    mri_flair: Optional[str] = None

    patient_id: Optional[str] = None
    study_id: Optional[str] = None
    generated_by: Optional[str] = "Stroke AI Triage System"

    include_gradcam: bool = False
    gradcam_dicom_path: Optional[str] = None
    gradcam_class_idx: int = 0

    rsna_ckpt: Optional[str] = "outputs/rsna_models/rsna_hemorrhage_best.pth"
    isles_ckpt: Optional[str] = "outputs/isles_models/isles_unet_cached_best.pth"


#HELPERS

def _check_path(p: Optional[str], kind: str):
    if p is None:
        return

    path = Path(p)

    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"{kind} path not found: {p}",
        )


def _validate_triage_inputs(
    ct_dir: Optional[str],
    mri_dwi: Optional[str],
    mri_adc: Optional[str],
    mri_flair: Optional[str],
):
    has_ct = bool(ct_dir)
    has_complete_mri = bool(mri_dwi and mri_adc and mri_flair)

    if not has_ct and not has_complete_mri:
        raise HTTPException(
            status_code=400,
            detail="Provide at least CT (ct_dir) or complete MRI inputs (mri_dwi, mri_adc, mri_flair).",
        )

    has_partial_mri = bool(
        mri_dwi or mri_adc or mri_flair) and not has_complete_mri

    if has_partial_mri:
        raise HTTPException(
            status_code=400,
            detail="MRI input is incomplete. Provide all three MRI paths: mri_dwi, mri_adc, and mri_flair.",
        )


#PROTECTED ROUTES

@app.post("/triage/paths")
def triage_from_paths(req: TriageRequest, admin=Depends(require_admin)):
    """
    Run triage using local filesystem paths.

    Requires admin Authorization header:
    Authorization: Bearer <token>
    """

    _validate_triage_inputs(
        ct_dir=req.ct_dir,
        mri_dwi=req.mri_dwi,
        mri_adc=req.mri_adc,
        mri_flair=req.mri_flair,
    )

    _check_path(req.ct_dir, "CT directory")
    _check_path(req.mri_dwi, "MRI DWI")
    _check_path(req.mri_adc, "MRI ADC")
    _check_path(req.mri_flair, "MRI FLAIR")
    _check_path(req.rsna_ckpt, "RSNA checkpoint")
    _check_path(req.isles_ckpt, "ISLES checkpoint")

    try:
        result = run_triage(
            ct_dir=req.ct_dir,
            mri_dwi=req.mri_dwi,
            mri_adc=req.mri_adc,
            mri_flair=req.mri_flair,
            rsna_ckpt=req.rsna_ckpt,
            isles_ckpt=req.isles_ckpt,
        )

        result["requested_by"] = admin.get("sub")
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Triage failed: {e}",
        )


@app.post("/gradcam/ct_slice")
def ct_slice_gradcam(req: CTGradCAMRequest, admin=Depends(require_admin)):
    """
    Generate Grad-CAM for one CT DICOM slice.

    Requires admin Authorization header.
    """

    _check_path(req.dicom_path, "DICOM")
    _check_path(req.rsna_ckpt, "RSNA checkpoint")

    try:
        rel_path = generate_ct_gradcam(
            dicom_path=req.dicom_path,
            ckpt_path=req.rsna_ckpt,
            class_idx=req.class_idx,
            image_size=512,
        )

        gradcam_url = f"/static/{rel_path}"

        return {
            "gradcam_url": gradcam_url,
            "class_idx": req.class_idx,
            "requested_by": admin.get("sub"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM error: {e}",
        )


@app.post("/report/pdf")
def report_pdf(req: TriagePdfRequest, admin=Depends(require_admin)):
    """
    Generate a PDF triage report and return it as a downloadable file.

    Requires admin Authorization header.
    """

    _validate_triage_inputs(
        ct_dir=req.ct_dir,
        mri_dwi=req.mri_dwi,
        mri_adc=req.mri_adc,
        mri_flair=req.mri_flair,
    )

    _check_path(req.ct_dir, "CT directory")
    _check_path(req.mri_dwi, "MRI DWI")
    _check_path(req.mri_adc, "MRI ADC")
    _check_path(req.mri_flair, "MRI FLAIR")
    _check_path(req.rsna_ckpt, "RSNA checkpoint")
    _check_path(req.isles_ckpt, "ISLES checkpoint")

    gradcam_png_abs = None

    if req.include_gradcam:
        if not req.gradcam_dicom_path:
            raise HTTPException(
                status_code=400,
                detail="include_gradcam=True but gradcam_dicom_path was not provided.",
            )

        _check_path(req.gradcam_dicom_path, "Grad-CAM DICOM")

        try:
            rel_path = generate_ct_gradcam(
                dicom_path=req.gradcam_dicom_path,
                ckpt_path=req.rsna_ckpt,
                class_idx=req.gradcam_class_idx,
                image_size=512,
            )

            gradcam_png_abs = str(Path("outputs") / rel_path)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Grad-CAM generation failed for PDF: {e}",
            )

    try:
        result = run_triage(
            ct_dir=req.ct_dir,
            mri_dwi=req.mri_dwi,
            mri_adc=req.mri_adc,
            mri_flair=req.mri_flair,
            rsna_ckpt=req.rsna_ckpt,
            isles_ckpt=req.isles_ckpt,
        )

        result["requested_by"] = admin.get("sub")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Triage failed: {e}",
        )

    try:
        pdf_bytes = build_triage_pdf_bytes(
            triage_result=result,
            patient_id=req.patient_id,
            study_id=req.study_id,
            generated_by=req.generated_by or "Stroke AI Triage System",
            gradcam_png_path=gradcam_png_abs,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF build failed: {e}",
        )

    safe_patient = (req.patient_id or "patient").replace(" ", "_")
    safe_study = (req.study_id or "study").replace(" ", "_")
    filename = f"stroke_triage_report_{safe_patient}_{safe_study}.pdf"

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )
