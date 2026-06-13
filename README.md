# 🧠 AI-Powered Stroke Emergency Triage System

Fast • Explainable • Clinical Decision Support AI for Stroke Diagnosis

---

## 📌 Overview

This project is a full-stack AI system designed to assist in **emergency stroke detection and triage** using medical imaging.

It combines deep learning models with a clinical decision engine to:

- Detect intracranial hemorrhage from CT scans
- Segment ischemic stroke lesions from MRI scans
- Classify stroke type and severity
- Provide explainable AI outputs (Grad-CAM)
- Generate automated clinical PDF reports
- Support real-time hospital decision-making

---

## 🚨 Problem Statement

Stroke is a time-critical medical emergency:

- Every minute = ~2 million neurons lost
- Delayed diagnosis increases mortality risk
- Radiologist workload causes delays in emergency settings

This system reduces diagnosis time using AI-assisted triage.

---

## 🎯 Objective

To build an AI system that:

- Detects stroke presence (CT + MRI)
- Classifies stroke type
- Estimates severity
- Provides explainability
- Generates structured reports
- Supports clinical decision-making

---

## 🧠 System Architecture

CT Scan (DICOM)
→ EfficientNet-B0 (RSNA Model)
→ Hemorrhage Classification (6 classes)

MRI Scan (DWI / ADC / FLAIR)
→ 3D U-Net (ISLES Model)
→ Ischemic Lesion Segmentation

Fusion Layer
→ Rule-Based Triage Engine
→ Stroke Type + Severity + Priority

Output Layer
→ Web Dashboard + Grad-CAM + PDF Report

---

## 🧪 Models Used

### CT Hemorrhage Detection
- Dataset: RSNA Intracranial Hemorrhage Dataset
- Model: EfficientNet-B0
- Task: Multi-label classification (6 hemorrhage types)

### MRI Stroke Segmentation
- Dataset: ISLES 2022
- Model: 3D U-Net (MONAI framework)
- Task: Voxel-level lesion segmentation

---

## 🔍 Explainability

### Grad-CAM (CT)
- Highlights regions influencing model predictions
- Provides visual interpretation for clinicians

### MRI Mask Output
- Displays ischemic lesion regions
- Calculates lesion volume (ml)

---

## 📊 Evaluation Results

### CT Model (RSNA)
- Mean AUC: 0.9878

### MRI Model (ISLES)
- Dice Score: 0.5524
- HD95: 23.49

---

## ⚙️ Features

- CT hemorrhage detection
- MRI lesion segmentation
- Stroke type classification
- Severity scoring system
- Real-time web dashboard
- Grad-CAM explainability
- PDF medical report generation
- Admin login system

---

## 🏗️ Tech Stack

Backend:
- FastAPI
- PyTorch
- MONAI
- OpenCV

Frontend:
- React.js
- Tailwind CSS
- Vite

AI Models:
- EfficientNet-B0
- 3D U-Net
- Grad-CAM

---

## 🚀 How to Run

Backend:
cd backend
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000

Frontend:
cd frontend
npm install
npm run dev

Open:
http://localhost:5173

---

## 🔐 Login (Demo)

Username: admin  
Password: admin123  

---

## 💡 Key Innovations

- Dual-modality stroke AI (CT + MRI fusion)
- Real-time clinical decision support system
- Explainable AI using Grad-CAM + segmentation maps
- Automated medical report generation
- Full-stack deployment pipeline

---

## ⚠️ Limitations

- Requires clinical validation
- MRI segmentation still improving
- Not FDA/medical certified
- Research prototype only

---

## 📈 Future Improvements

- Transformer-based medical imaging models
- Multi-hospital training
- PACS integration
- Cloud deployment (AWS / Azure)
- Federated learning for privacy

---

## 👨‍🎓 Author

Lakindu Pahesara  
BSc (Hons) Data Science  
University of Plymouth 

---

## 🏁 Final Note

This system demonstrates an end-to-end AI pipeline for stroke triage combining medical imaging, deep learning, and clinical decision support.

---
