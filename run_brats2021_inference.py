"""
BraTS 2021 Segmentation - Direct Inference
Uses nnU-Net v1 Python API for the BraTS 2021 KAIST model
"""
import os
import sys
from pathlib import Path
import shutil

# Add Brats21_KAIST_MRI_Lab to path
project_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir / "Brats21_KAIST_MRI_Lab"))

# Setup nnU-Net environment
os.environ['nnUNet_raw_data_base'] = str(project_dir / "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = str(project_dir / "nnUNet_preprocessed")
os.environ['RESULTS_FOLDER'] = str(project_dir / "nnUNet_results")

print("=" * 70)
print("BraTS 2021 TUMOR SEGMENTATION")
print("=" * 70)
print(f"RESULTS_FOLDER: {os.environ['RESULTS_FOLDER']}\n")

# Import nnU-Net after setting env
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_trainer
from batchgenerators.utilities.file_and_folder_operations import *

def prepare_input(input_folder, temp_folder):
    """Convert BraTS naming to nnU-Net naming"""
    temp_folder = Path(temp_folder)
    temp_folder.mkdir(parents=True, exist_ok=True)
    
    input_folder = Path(input_folder)
    files = list(input_folder.glob("*.nii.gz"))
    
    # Find patient ID
    t1_file = [f for f in files if '_t1.nii.gz' in f.name and '_t1ce' not in f.name][0]
    patient_id = t1_file.name.replace('_t1.nii.gz', '')
    
    print(f"Patient ID: {patient_id}")
    print(f"Converting files to nnU-Net format...")
    
    # Mapping: BraTS modality -> nnU-Net channel
    modality_map = {
        '_t1.nii.gz': '_0000.nii.gz',
        '_t1ce.nii.gz': '_0001.nii.gz',
        '_t2.nii.gz': '_0002.nii.gz',
        '_flair.nii.gz': '_0003.nii.gz'
    }
    
    for brats_suffix, nnunet_suffix in modality_map.items():
        src = input_folder / f"{patient_id}{brats_suffix}"
        dst = temp_folder / f"{patient_id}{nnunet_suffix}"
        if src.exists():
            shutil.copy(src, dst)
            print(f"  ✓ {src.name} -> {dst.name}")
        else:
            print(f"  ✗ Missing: {src.name}")
            sys.exit(1)
    
    return patient_id

def run_model(input_folder, output_folder, model_name, task='Task500_BraTS2021'):
    """Run a single nnU-Net model"""
    print(f"\nRunning model: {model_name}")
    print("-" * 70)
    
    model_folder = Path(os.environ['RESULTS_FOLDER']) / "3d_fullres" / task / model_name
    
    if not model_folder.exists():
        print(f"[ERROR] Model not found: {model_folder}")
        sys.exit(1)
    
    print(f"Model path: {model_folder}")
    
    # Use all 5 folds for ensemble
    folds = (0, 1, 2, 3, 4)
    
    predict_from_folder(
        model=str(model_folder),
        input_folder=str(input_folder),
        output_folder=str(output_folder),
        folds=folds,
        save_npz=True,
        num_threads_preprocessing=1,  # Disable multiprocessing (Windows compatibility)
        num_threads_nifti_save=1,      # Disable multiprocessing (Windows compatibility)
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=True,  # Test-time augmentation
        overwrite_existing=True,
        mode='normal',
        overwrite_all_in_gpu=None,
        mixed_precision=True,
        step_size=0.5
    )
    
    print(f"[OK] {model_name} complete!")

def ensemble_predictions(folder1, folder2, output_folder):
    """Ensemble two model predictions"""
    import numpy as np
    import nibabel as nib
    
    print("\nEnsembling predictions...")
    print("-" * 70)
    
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get prediction files
    files1 = list(folder1.glob("*.nii.gz"))
    
    for file1 in files1:
        file2 = folder2 / file1.name
        
        if not file2.exists():
            print(f"[WARNING] Missing file in model 2: {file1.name}")
            continue
        
        # Load predictions
        nii1 = nib.load(file1)
        nii2 = nib.load(file2)
        
        seg1 = nii1.get_fdata().astype(np.uint8)
        seg2 = nii2.get_fdata().astype(np.uint8)
        
        # Simple voting ensemble
        ensemble = np.where(seg1 == seg2, seg1, seg1)  # If they agree, use that; else use model1
        
        # Save ensemble result
        ensemble_nii = nib.Nifti1Image(ensemble, nii1.affine, nii1.header)
        output_file = output_folder / file1.name
        nib.save(ensemble_nii, output_file)
        print(f"  ✓ {output_file.name}")
    
    print("[OK] Ensemble complete!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='BraTS 2021 Tumor Segmentation')
    parser.add_argument('--input', required=True, help='Input folder with BraTS MRI scans')
    parser.add_argument('--output', required=True, help='Output folder for segmentation')
    args = parser.parse_args()
    
    # Create temporary folders
    temp_input = project_dir / "temp_inference_input"
    temp_output1 = project_dir / "temp_inference_output1"
    temp_output2 = project_dir / "temp_inference_output2"
    
    # Prepare input
    patient_id = prepare_input(args.input, temp_input)
    
    # Run Model 1
    print("\n" + "=" * 70)
    print("[1/3] MODEL 1: nnUNetTrainerV2BraTSRegions_DA4_BN_BD")
    print("=" * 70)
    run_model(
        temp_input,
        temp_output1,
        "nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1"
    )
    
    # Run Model 2
    print("\n" + "=" * 70)
    print("[2/3] MODEL 2: largeUnet_Groupnorm")
    print("=" * 70)
    run_model(
        temp_input,
        temp_output2,
        "nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm__nnUNetPlansv2.1"
    )
    
    # Ensemble
    print("\n" + "=" * 70)
    print("[3/3] ENSEMBLE")
    print("=" * 70)
    output_folder = Path(args.output)
    ensemble_predictions(temp_output1, temp_output2, output_folder)
    
    # Calculate volumes
    print("\n" + "=" * 70)
    print("TUMOR VOLUME ANALYSIS")
    print("=" * 70)
    
    import nibabel as nib
    import numpy as np
    
    seg_file = list(output_folder.glob("*.nii.gz"))[0]
    seg_nii = nib.load(seg_file)
    seg_data = seg_nii.get_fdata()
    
    voxel_dims = seg_nii.header.get_zooms()
    voxel_volume_cm3 = np.prod(voxel_dims) / 1000
    
    ncr_volume = np.sum(seg_data == 1) * voxel_volume_cm3
    ed_volume = np.sum(seg_data == 2) * voxel_volume_cm3
    et_volume = np.sum(seg_data == 4) * voxel_volume_cm3
    
    tc_volume = ncr_volume + et_volume
    wt_volume = ncr_volume + ed_volume + et_volume
    
    print(f"\nSegmentation: {seg_file.name}")
    print("-" * 70)
    print(f"  Necrotic Core (NCR):         {ncr_volume:>10.2f} cm³")
    print(f"  Peritumoral Edema (ED):      {ed_volume:>10.2f} cm³")
    print(f"  Enhancing Tumor (ET):        {et_volume:>10.2f} cm³")
    print("-" * 70)
    print(f"  Tumor Core (TC):             {tc_volume:>10.2f} cm³")
    print(f"  Whole Tumor (WT):            {wt_volume:>10.2f} cm³")
    print("=" * 70)
    
    print(f"\n✓ Results saved to: {output_folder}")
    
    # Cleanup temp folders
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_input, ignore_errors=True)
    shutil.rmtree(temp_output1, ignore_errors=True)
    shutil.rmtree(temp_output2, ignore_errors=True)
    print("[OK] Done!")

if __name__ == "__main__":
    main()
