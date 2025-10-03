"""
Convert BraTS 2024 dataset to nnU-Net format
This script converts BraTS data structure to the format required by nnU-Net
"""
import os
import json
import shutil
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def convert_brats_to_nnunet(brats_data_path, nnunet_raw_path):
    """
    Convert BraTS 2024 dataset to nnU-Net format
    
    Args:
        brats_data_path: Path to the original BraTS 2024 dataset
        nnunet_raw_path: Path to nnUNet_raw directory
    """
    brats_path = Path(brats_data_path)
    dataset_path = Path(nnunet_raw_path) / "Dataset001_BraTS2024"
    
    # Create output directories
    images_tr = dataset_path / "imagesTr"
    labels_tr = dataset_path / "labelsTr"
    images_ts = dataset_path / "imagesTs"
    
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)
    
    # BraTS modality suffixes
    modalities = ['t1n', 't1c', 't2w', 't2f']  # T1, T1CE, T2, FLAIR
    
    # Process training data
    print("Converting training data...")
    training_cases = list(brats_path.glob("BraTS-GLI-*")) + list(brats_path.glob("BraTS-MEN-*")) + \
                     list(brats_path.glob("BraTS-MET-*")) + list(brats_path.glob("BraTS-PED-*"))
    
    case_counter = 0
    
    for case_folder in tqdm(training_cases, desc="Processing cases"):
        if not case_folder.is_dir():
            continue
            
        case_name = case_folder.name
        case_id = f"{case_counter:05d}"
        
        # Check if all required files exist
        has_all_modalities = True
        modality_files = {}
        
        for idx, mod in enumerate(modalities):
            # BraTS 2024 naming: BraTS-XXX-XXXXX-XXX-t1n.nii.gz
            mod_file = case_folder / f"{case_name}-{mod}.nii.gz"
            if not mod_file.exists():
                has_all_modalities = False
                break
            modality_files[idx] = mod_file
        
        # Check for segmentation file
        seg_file = case_folder / f"{case_name}-seg.nii.gz"
        has_segmentation = seg_file.exists()
        
        if not has_all_modalities:
            continue
        
        # Copy and rename modality files
        for mod_idx, mod_file in modality_files.items():
            output_name = f"BraTS2024_{case_id}_{mod_idx:04d}.nii.gz"
            output_path = images_tr / output_name
            shutil.copy2(mod_file, output_path)
        
        # Copy segmentation if available
        if has_segmentation:
            output_seg_name = f"BraTS2024_{case_id}.nii.gz"
            output_seg_path = labels_tr / output_seg_name
            shutil.copy2(seg_file, output_seg_path)
        
        case_counter += 1
    
    print(f"Converted {case_counter} training cases")
    
    # Update dataset.json with the number of training cases
    dataset_json_path = dataset_path / "dataset.json"
    if dataset_json_path.exists():
        with open(dataset_json_path, 'r') as f:
            dataset_json = json.load(f)
        
        dataset_json['numTraining'] = case_counter
        
        with open(dataset_json_path, 'w') as f:
            json.dump(dataset_json, f, indent=4)
        
        print(f"Updated dataset.json with {case_counter} training cases")
    
    return case_counter


def verify_conversion(nnunet_raw_path):
    """
    Verify that the conversion was successful
    """
    dataset_path = Path(nnunet_raw_path) / "Dataset001_BraTS2024"
    images_tr = dataset_path / "imagesTr"
    labels_tr = dataset_path / "labelsTr"
    
    image_files = list(images_tr.glob("*.nii.gz"))
    label_files = list(labels_tr.glob("*.nii.gz"))
    
    print("\n" + "=" * 60)
    print("Verification:")
    print(f"Total image files: {len(image_files)}")
    print(f"Total label files: {len(label_files)}")
    print(f"Expected: {len(label_files) * 4} images for {len(label_files)} cases (4 modalities each)")
    
    if len(image_files) == len(label_files) * 4:
        print("✓ Conversion successful!")
    else:
        print("⚠ Warning: Number of files doesn't match expected count")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert BraTS 2024 to nnU-Net format")
    parser.add_argument("--brats_path", type=str, required=True,
                       help="Path to BraTS 2024 dataset directory")
    parser.add_argument("--nnunet_raw", type=str, default="./nnUNet_raw",
                       help="Path to nnUNet_raw directory")
    
    args = parser.parse_args()
    
    print("Converting BraTS 2024 dataset to nnU-Net format...")
    print("=" * 60)
    
    num_cases = convert_brats_to_nnunet(args.brats_path, args.nnunet_raw)
    verify_conversion(args.nnunet_raw)
    
    print("\nConversion complete!")
    print("\nNext step: Run preprocessing")
    print("Command: nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity")
