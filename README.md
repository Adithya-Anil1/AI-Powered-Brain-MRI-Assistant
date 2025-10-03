# AI-Powered Brain MRI Assistant

This project implements brain tumor segmentation using nnU-Net on the BraTS 2024 dataset.

## Overview

nnU-Net (no-new-Net) is a self-configuring deep learning framework for medical image segmentation. This implementation is specifically configured for brain tumor segmentation on BraTS 2024 data, which includes multi-modal MRI scans (T1, T1CE, T2, FLAIR).

### Segmentation Labels
- **Label 0**: Background
- **Label 1**: NCR (Necrotic tumor core)
- **Label 2**: ED (Peritumoral edematous/invaded tissue)
- **Label 3**: ET (Enhancing tumor)

### Composite Regions
- **Whole Tumor (WT)**: NCR + ED + ET
- **Tumor Core (TC)**: NCR + ET
- **Enhancing Tumor (ET)**: Label 3

## Project Structure

```
AI-Powered Brain MRI Assistant/
├── setup_nnunet.py           # Initial setup and directory creation
├── convert_brats_data.py     # Convert BraTS data to nnU-Net format
├── train_nnunet.py           # Training script
├── inference_nnunet.py       # Inference and visualization
├── requirements.txt          # Python dependencies
├── nnUNet_raw/              # Raw dataset (created by setup)
│   └── Dataset001_BraTS2024/
│       ├── imagesTr/        # Training images
│       ├── labelsTr/        # Training labels
│       ├── imagesTs/        # Test images
│       └── dataset.json     # Dataset configuration
├── nnUNet_preprocessed/     # Preprocessed data (created during training)
└── nnUNet_results/          # Model checkpoints and results
```

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup nnU-Net Directories

```bash
python setup_nnunet.py
```

This will:
- Create the required directory structure
- Set up environment variables
- Create the dataset.json configuration file

## Data Preparation

### 1. Download BraTS 2024 Dataset

Download the BraTS 2024 dataset from the official source:
https://www.synapse.org/#!Synapse:syn53708249/wiki/

### 2. Convert Data to nnU-Net Format

```bash
python convert_brats_data.py --brats_path "path/to/BraTS2024" --nnunet_raw "./nnUNet_raw"
```

This script will:
- Convert BraTS naming convention to nnU-Net format
- Copy all 4 modalities (T1, T1CE, T2, FLAIR) for each case
- Copy segmentation masks
- Update dataset.json with the number of training cases

### Expected BraTS Data Structure

```
BraTS2024/
├── BraTS-GLI-00000-000/
│   ├── BraTS-GLI-00000-000-t1n.nii.gz
│   ├── BraTS-GLI-00000-000-t1c.nii.gz
│   ├── BraTS-GLI-00000-000-t2w.nii.gz
│   ├── BraTS-GLI-00000-000-t2f.nii.gz
│   └── BraTS-GLI-00000-000-seg.nii.gz
├── BraTS-GLI-00001-001/
│   └── ...
└── ...
```

## Training

### 1. Preprocessing

First, run preprocessing and planning:

```bash
python train_nnunet.py --mode preprocess --dataset_id 1
```

Or use the nnU-Net command directly:

```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

### 2. Train the Model

Train using the automated script:

```bash
python train_nnunet.py --mode train --dataset_id 1 --config 3d_fullres --fold 0 --gpu 0
```

Or use nnU-Net commands directly:

```bash
# Train a single fold
nnUNetv2_train 001 3d_fullres 0

# Train all folds (5-fold cross-validation)
nnUNetv2_train 001 3d_fullres all
```

### Configuration Options

- **2d**: 2D U-Net
- **3d_fullres**: 3D U-Net at full resolution (recommended for BraTS)
- **3d_lowres**: 3D U-Net at lower resolution
- **3d_cascade_fullres**: Cascade of low-res and full-res (for very high-resolution data)

### Training Tips

- Training on BraTS data typically takes 24-48 hours on a modern GPU (RTX 3090/4090)
- The 3d_fullres configuration is recommended for BraTS
- Use all 5 folds for best performance, then ensemble predictions
- Monitor training with TensorBoard logs in nnUNet_results

## Inference

### Single Case Prediction

```bash
python inference_nnunet.py \
    --t1 "path/to/t1.nii.gz" \
    --t1ce "path/to/t1ce.nii.gz" \
    --t2 "path/to/t2.nii.gz" \
    --flair "path/to/flair.nii.gz" \
    --output "path/to/output_seg.nii.gz" \
    --visualize \
    --plot_output "path/to/visualization.png"
```

### Batch Prediction

```bash
python train_nnunet.py --mode predict \
    --input_folder "path/to/test/images" \
    --output_folder "path/to/predictions" \
    --dataset_id 1 \
    --config 3d_fullres \
    --fold 0
```

Or use nnU-Net command:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 001 -c 3d_fullres -f 0
```

### Ensemble Prediction (Best Performance)

To get the best results, ensemble predictions from all 5 folds:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 001 -c 3d_fullres -f all
```

## Evaluation

Evaluate predictions against ground truth:

```bash
nnUNetv2_evaluate_folder GT_FOLDER PREDICTION_FOLDER -djfile nnUNet_raw/Dataset001_BraTS2024/dataset.json
```

## Model Performance

nnU-Net typically achieves state-of-the-art performance on BraTS:
- **Whole Tumor (WT) Dice**: ~90-92%
- **Tumor Core (TC) Dice**: ~85-88%
- **Enhancing Tumor (ET) Dice**: ~80-85%

## Environment Variables

The following environment variables are automatically set by the scripts:

```python
nnUNet_raw = "c:/Users/adith/OneDrive/Desktop/AI-Powered Brain MRI Assistant/nnUNet_raw"
nnUNet_preprocessed = "c:/Users/adith/OneDrive/Desktop/AI-Powered Brain MRI Assistant/nnUNet_preprocessed"
nnUNet_results = "c:/Users/adith/OneDrive/Desktop/AI-Powered Brain MRI Assistant/nnUNet_results"
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ free space for dataset and results
- **CUDA**: Version 11.0 or higher

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in nnUNetTrainerV2 (modify plans file)
- Use 3d_lowres configuration
- Use a GPU with more VRAM

### Dataset Integrity Errors
- Verify all images have the same spacing and orientation
- Check that all 4 modalities exist for each case
- Ensure file naming follows nnU-Net convention exactly

### Training Crashes
- Check GPU drivers and CUDA installation
- Verify sufficient disk space
- Monitor GPU temperature and usage

## Citation

If you use this code, please cite:

```bibtex
@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## References

- [nnU-Net GitHub](https://github.com/MIC-DKFZ/nnUNet)
- [BraTS 2024 Challenge](https://www.synapse.org/#!Synapse:syn53708249/wiki/)
- [nnU-Net Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)

## License

This project uses nnU-Net, which is licensed under Apache License 2.0.
