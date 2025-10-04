
## 🎯 Project Overview

This is an **AI-assisted diagnostic system** for brain tumor segmentation using pretrained nnU-Net models. The system analyzes multi-modal MRI scans (T1, T1CE, T2, FLAIR) and automatically identifies tumor regions to assist radiologists in diagnosis.

### What This MVP Does:
✅ Loads a **pretrained nnU-Net model** (no training required)  
✅ Segments brain tumors into 3 regions: NCR, ED, ET  
✅ Calculates tumor volumes automatically  
✅ Generates **visual overlays** showing segmentation results  
✅ Provides structured output for integration into reports  

### Tumor Segmentation Labels:
- **NCR (Label 1)**: Necrotic/Non-enhancing Tumor Core
- **ED (Label 2)**: Peritumoral Edema
- **ET (Label 3)**: Enhancing Tumor

### Clinical Metrics:
- **Whole Tumor (WT)**: NCR + ED + ET (all tumor regions)
- **Tumor Core (TC)**: NCR + ET (solid tumor parts)
- **Enhancing Tumor (ET)**: Active tumor region

---

## 📁 Project Files

```
AI-Powered Brain MRI Assistant/
├── requirements.txt              # Python dependencies
├── setup_nnunet.py              # Creates directory structure
├── download_pretrained_model.py # Downloads pretrained nnU-Net
├── inference_nnunet.py          # Runs segmentation & visualization
├── validate_setup.py            # Checks if setup is correct
└── README.md                    # This file
```

---

## 🚀 Complete Setup Guide

### **Step 1: Install Python Packages** (10-15 minutes)

Create a virtual environment (recommended):
```cmd
python -m venv venv
venv\Scripts\activate
```

Install dependencies:
```cmd
pip install -r requirements.txt
```

**What gets installed:**
- `nnunetv2` - The segmentation framework
- `torch` - Deep learning engine
- `SimpleITK` - Medical image processing
- `matplotlib` - Visualization
- `numpy`, `scipy` - Numerical processing

---

### **Step 2: Setup Directories** (1 minute)

Run the setup script:
```cmd
python setup_nnunet.py
```

**What this does:**
- Creates `nnUNet_raw/` folder
- Creates `nnUNet_preprocessed/` folder  
- Creates `nnUNet_results/` folder
- Sets environment variables automatically

---

### **Step 3: Download Pretrained Model** (5-10 minutes)

Run the download script:
```cmd
python download_pretrained_model.py
```

**What this does:**
- Downloads a pretrained nnU-Net model trained on BraTS data
- Places it in `nnUNet_results/`
- Model is ready to use immediately (no training needed!)

**Model source:** You can use models from:
- [nnU-Net Model Zoo](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/model_zoo.md)
- [BraTS pretrained models](https://zenodo.org/communities/brats)

---

### **Step 4: Validate Setup** (1 minute)

Check if everything is configured correctly:
```cmd
python validate_setup.py
```

**Expected output:**
```
✓ Directories: PASS
✓ Dataset JSON: PASS  
✓ Python Packages: PASS
✓ CUDA: PASS
✓ All checks passed!
```

---

## 🧠 Running Segmentation

### **Single Patient MRI Scan**

```cmd
python inference_nnunet.py ^
    --t1 "path\to\patient_t1.nii.gz" ^
    --t1ce "path\to\patient_t1ce.nii.gz" ^
    --t2 "path\to\patient_t2.nii.gz" ^
    --flair "path\to\patient_flair.nii.gz" ^
    --output "segmentation_result.nii.gz" ^
    --visualize ^
    --plot_output "visualization.png"
```

**What you get:**
1. `segmentation_result.nii.gz` - 3D segmentation mask
2. `visualization.png` - Visual overlay showing tumor regions
3. Console output with tumor volumes

**Example output:**
```
✓ Segmentation saved to: segmentation_result.nii.gz

Tumor volumes:
  Whole Tumor (WT): 45823.45 mm³
  Tumor Core (TC): 23451.23 mm³
  Enhancing Tumor (ET): 12345.67 mm³
  Necrotic Core (NCR): 11105.56 mm³
  Edema (ED): 22372.22 mm³
```

---

## 📊 Understanding the Output

### **1. Segmentation Mask** (`segmentation_result.nii.gz`)
- 3D NIfTI file with the same dimensions as input
- Each voxel labeled: 0 (background), 1 (NCR), 2 (ED), 3 (ET)
- Can be loaded in medical imaging software (3D Slicer, ITK-SNAP)

### **2. Visualization** (`visualization.png`)
Three panels:
- **Left:** Original T1CE MRI slice
- **Middle:** Segmentation mask (color-coded)
- **Right:** Overlay (mask on top of MRI)

Color coding:
- 🔵 Blue = NCR (Necrotic core)
- 🟢 Green = ED (Edema)
- 🔴 Red = ET (Enhancing tumor)

### **3. Volume Measurements**
Printed to console and can be saved to file for report generation

---

### **Integration Ideas:**

1. **Automated Report Generation:**
   - Parse the volume output from `inference_nnunet.py`
   - Generate structured reports using templates
   - Include visualizations in PDF reports

2. **Batch Processing:**
   - Modify script to process multiple patients
   - Store results in database
   - Generate comparison reports

3. **Quality Control:**
   - Visual review of segmentations
   - Flag cases with unusual volumes
   - Export for radiologist confirmation

---

## ⚙️ System Requirements

### **Minimum (for testing):**
- GPU: NVIDIA GTX 1060 (6GB VRAM)
- RAM: 8GB
- Storage: 20GB free space
- OS: Windows 10/11

### **Recommended (for clinical use):**
- GPU: NVIDIA RTX 3060/4060 (12GB+ VRAM)
- RAM: 16GB+
- Storage: 50GB SSD
- OS: Windows 10/11 with latest updates

### **Inference Speed:**
- ~30-60 seconds per patient (with GPU)
- ~5-10 minutes per patient (CPU only)


## 📚 References

- **nnU-Net Paper:** [Nature Methods 2021](https://www.nature.com/articles/s41592-020-01008-z)
- **nnU-Net GitHub:** https://github.com/MIC-DKFZ/nnUNet
- **BraTS Challenge:** https://www.synapse.org/brats

---



