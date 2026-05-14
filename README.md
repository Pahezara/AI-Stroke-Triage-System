\# AI-Powered Stroke Emergency Triage System



This project is an AI-assisted emergency stroke triage prototype using CT and MRI medical imaging.



\## Main Features



\- CT hemorrhage classification using RSNA Intracranial Hemorrhage dataset

\- MRI ischemic lesion segmentation using ISLES 2022 dataset

\- Dual-model triage logic for stroke type and severity

\- Grad-CAM explainability for CT predictions

\- MRI lesion mask and lesion volume estimation

\- FastAPI backend

\- React + Tailwind dashboard

\- Admin login

\- PDF report generation



\## Models



\- CT model: EfficientNet-B0 multi-label classifier

\- MRI model: 3D U-Net segmentation model



\## Evaluation Results



RSNA validation mean AUC: 0.9878  

ISLES validation Dice: 0.5524 ± 0.2721  

ISLES HD95: 23.4937 ± 19.3994



\## Important Notes



Full RSNA and ISLES datasets are not included in this repository due to size and access restrictions.



Model weights are provided separately through the submission OneDrive link.




This is a research prototype and must not be used for real clinical diagnosis without formal validation and regulatory approval.



\## Backend Setup



```bash

cd backend

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

python -m uvicorn src.api:app --reload --port 8000

