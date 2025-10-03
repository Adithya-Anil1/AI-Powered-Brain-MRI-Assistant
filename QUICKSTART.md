# AI-Powered Brain MRI Assistant - Quick Start Guide

## Step-by-Step Setup

### 1. Install Dependencies (5 minutes)

First, create a Python virtual environment (recommended):
```cmd
python -m venv venv
venv\Scripts\activate
```

Then install all required packages:
```cmd
pip install -r requirements.txt
```

**Note**: This will install PyTorch, nnU-Net, and all dependencies. Installation may take 10-15 minutes.

### 2. Initialize Project (1 minute)

Run the setup script to create directory structure:
```cmd
python setup_nnunet.py
```

This creates:
- `nnUNet_raw/Dataset001_BraTS2024/` - for raw data
- `nnUNet_preprocessed/` - for preprocessed data
- `nnUNet_results/` - for trained models

### 3. Prepare BraTS 2024 Data (10-30 minutes)

Download BraTS 2024 from: https://www.synapse.org/#!Synapse:syn53708249/wiki/

Then convert to nnU-Net format:
```cmd
python convert_brats_data.py --brats_path "C:\path\to\BraTS2024" --nnunet_raw ".\nnUNet_raw"
```

### 4. Train the Model (1-2 days on GPU)

First, preprocess the data:
```cmd
python train_nnunet.py --mode preprocess --dataset_id 1
```

Then start training:
```cmd
python train_nnunet.py --mode train --dataset_id 1 --config 3d_fullres --fold 0 --gpu 0
```

For best results, train all 5 folds:
```cmd
for /L %i in (0,1,4) do python train_nnunet.py --mode train --dataset_id 1 --config 3d_fullres --fold %i --gpu 0
```

### 5. Run Inference

On a single case:
```cmd
python inference_nnunet.py ^
    --t1 "path\to\t1.nii.gz" ^
    --t1ce "path\to\t1ce.nii.gz" ^
    --t2 "path\to\t2.nii.gz" ^
    --flair "path\to\flair.nii.gz" ^
    --output "output_segmentation.nii.gz" ^
    --visualize
```

## Quick Commands Reference

### Training
```cmd
# Full pipeline (preprocess + train)
python train_nnunet.py --mode all --dataset_id 1 --config 3d_fullres --fold 0

# Just preprocessing
python train_nnunet.py --mode preprocess --dataset_id 1

# Just training
python train_nnunet.py --mode train --dataset_id 1 --config 3d_fullres --fold 0
```

### Inference
```cmd
# Single case with visualization
python inference_nnunet.py --t1 t1.nii.gz --t1ce t1ce.nii.gz --t2 t2.nii.gz --flair flair.nii.gz --output seg.nii.gz --visualize

# Batch prediction
python train_nnunet.py --mode predict --input_folder "test_images" --output_folder "predictions"
```

### Evaluation
```cmd
# Evaluate predictions
python evaluate_predictions.py --pred_folder "predictions" --gt_folder "ground_truth" --output "results.json"
```

## File Naming Convention

nnU-Net expects specific file names:

**Training images**: `BraTS2024_00000_0000.nii.gz` to `BraTS2024_00000_0003.nii.gz`
- `_0000`: T1
- `_0001`: T1CE
- `_0002`: T2
- `_0003`: FLAIR

**Training labels**: `BraTS2024_00000.nii.gz`

The `convert_brats_data.py` script handles this automatically.

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use `3d_lowres` configuration

### Issue: "No module named 'nnunetv2'"
**Solution**: Run `pip install nnunetv2`

### Issue: "Environment variables not set"
**Solution**: Run `set_env.bat` or use the Python scripts which set them automatically

### Issue: Training is very slow
**Solution**: 
- Check GPU usage with `nvidia-smi`
- Ensure CUDA is properly installed
- Use smaller dataset for testing

## Hardware Requirements

**Minimum**:
- GPU: NVIDIA GTX 1080 Ti (11GB VRAM)
- RAM: 16GB
- Storage: 50GB

**Recommended**:
- GPU: NVIDIA RTX 3090/4090 (24GB VRAM)
- RAM: 32GB+
- Storage: 100GB+ SSD

## Expected Training Time

- **Preprocessing**: 1-3 hours (one-time)
- **Training single fold**: 24-48 hours
- **Training all 5 folds**: 5-10 days
- **Inference per case**: 30-60 seconds

## Expected Performance (Dice Scores)

- **Whole Tumor (WT)**: ~90-92%
- **Tumor Core (TC)**: ~85-88%
- **Enhancing Tumor (ET)**: ~80-85%

## Next Steps

1. **Monitor Training**: Check logs in `nnUNet_results/Dataset001_BraTS2024/`
2. **Validate Results**: Use `evaluate_predictions.py` to compute Dice scores
3. **Fine-tune**: Adjust hyperparameters if needed
4. **Ensemble**: Use all 5 folds for best performance

## Support

For issues with:
- **nnU-Net**: https://github.com/MIC-DKFZ/nnUNet/issues
- **BraTS Dataset**: https://www.synapse.org/#!Synapse:syn53708249/discussion
- **This project**: Check README.md for detailed documentation
