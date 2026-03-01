# 🧠 AI-Powered Brain MRI Assistant  

**An end-to-end neuro-oncology decision-support platform for automated tumor segmentation, quantitative analysis, structured radiology reporting, and retrieval-augmented clinical querying from multi-parametric MRI.**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)  
[![BraTS 2021](https://img.shields.io/badge/Model-BraTS%202021%20Winner-gold.svg)](https://github.com/KAIST-MRI-Lab/BraTS2021)  
[![Mean Dice](https://img.shields.io/badge/Mean%20Dice-95%25%2B-brightgreen.svg)](#performance)

---

## 📌 Overview

Accurate segmentation and structured interpretation of brain tumors in MRI are critical for clinical decision-making but remain time-intensive and subject to inter-observer variability.

This system implements a fully automated workflow that:

- Segments glioma subregions from four MRI sequences  
- Extracts 50+ clinically relevant quantitative descriptors  
- Generates a structured radiology report (TXT + PDF)  
- Enables safe, grounded clinical Q&A via a RAG assistant  

The platform is designed as a **decision-support tool** — standardizing analysis while preserving full physician oversight.

---

## 🎯 Core Capabilities

### 🔬 Advanced Tumor Segmentation

Segmentation is powered by the **1st-place BraTS 2021 model** from KAIST MRI Lab, based on the nnU-Net framework.

- 3D U-Net architecture with deep supervision  
- Dual-model ensemble inference  
- Region-based training strategy  
- Optimized for CPU execution  

The system segments:

- **Enhancing Tumor (ET)**  
- **Tumor Core (TC = ET + NCR)**  
- **Whole Tumor (WT = ET + NCR + ED)**  

Composite metrics are automatically derived from predicted subregions.

---

### 📊 Six-Stage Quantitative Feature Extraction

Beyond segmentation, the platform performs structured clinical analysis across six modules:

| Module | Focus Area | Clinical Relevance |
|--------|------------|-------------------|
| 1 | Sequence Signal Analysis | Intensity profiles, contrast enhancement, T2/FLAIR mismatch |
| 2 | Mass Effect Assessment | Midline shift, ventricular compression, sulcal effacement |
| 3 | Lesion Multiplicity | Multifocal detection, satellite lesion analysis |
| 4 | Morphology & Margins | Shape irregularity, necrosis patterns, infiltration indicators |
| 5 | Quality Metrics | Confidence scoring, artifact detection |
| 6 | Normal Structure Evaluation | Ventricular system, corpus callosum, structural involvement |

All outputs are stored in structured JSON files for transparency and traceability.

---

### 📝 Deterministic Radiology Report Generation

Reports are generated using a **template-constrained architecture**, not free-form language modeling.

- Fixed radiology report structure  
- Rule-based slot filling from validated measurements  
- Strict prohibition of diagnostic claims  
- Fully deterministic, template-based report generation  

Outputs include:

- `radiology_report.txt`  
- `radiology_report.pdf`  

This architecture prevents hallucination and ensures factual consistency.

---

### 💬 Retrieval-Augmented Clinical Q&A (RAG)

The integrated RAG assistant allows users to ask contextual questions about findings.

Key safeguards:

- Pre-LLM filtering for treatment/prognosis queries  
- Strict grounding to report content + curated references  
- Refusal mechanism when context is insufficient  
- No generation beyond retrieved evidence  

This module enhances interpretability without introducing uncontrolled generative behavior.

---

