# Scripts

Utility scripts for the AI-Powered Brain MRI Assistant.

## Evaluation Scripts

- **evaluate_segmentation.py** - Calculate Dice, IoU, sensitivity, specificity
- **compare_segmentations.py** - Create visual comparisons
- **check_labels.py** - Check what labels are in a segmentation file
- **visualize_segmentation.py** - Visualize segmentation results

## Setup Scripts

- **setup_brats2021_model.py** - Download and setup BraTS 2021 model
- **setup_nnunet.py** - Setup nnU-Net environment
- **download_pretrained_model.py** - Download pre-trained models
- **download_sample_data.py** - Download sample data
- **download_more_brats_data.py** - Guide for downloading more BraTS data

## Utility Scripts

- **check_compatibility.py** - Check system compatibility
- **validate_setup.py** - Validate installation
- **extract_sample.py** - Extract sample cases

## Usage

Run scripts from the project root directory:
```bash
python scripts/evaluate_segmentation.py --pred <prediction> --gt <ground_truth>
```
