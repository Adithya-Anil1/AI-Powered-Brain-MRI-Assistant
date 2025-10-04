"""
BraTS 2021 Brain Tumor Segmentation - Inference Script
Uses the winning BraTS 2021 KAIST model for tumor segmentation

Usage: python run_inference.py --input <patient_folder> --output <output_folder>
"""
import os
import sys
from pathlib import Path
import argparse
import subprocess


def setup_env():
    """Setup nnU-Net environment"""
    project_dir = Path(__file__).parent.absolute()
    os.environ['nnUNet_raw'] = str(project_dir / "nnUNet_raw")
    os.environ['nnUNet_preprocessed'] = str(project_dir / "nnUNet_preprocessed")
    os.environ['nnUNet_results'] = str(project_dir / "nnUNet_results")
    
    print("=" * 70)
    print("BraTS 2021 TUMOR SEGMENTATION")
    print("=" * 70)
    print(f"nnUNet_results: {os.environ['nnUNet_results']}")


def prepare_input_folder(patient_folder):
    """
    Prepare input folder in nnU-Net format
    BraTS format: *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz
    nnU-Net format: *_0000.nii.gz, *_0001.nii.gz, *_0002.nii.gz, *_0003.nii.gz
    """
    patient_folder = Path(patient_folder)
    
    # Find BraTS files
    files = list(patient_folder.glob("*.nii.gz"))
    
    # Check if files exist
    t1 = [f for f in files if '_t1.nii.gz' in f.name and '_t1ce' not in f.name]
    t1ce = [f for f in files if '_t1ce.nii.gz' in f.name]
    t2 = [f for f in files if '_t2.nii.gz' in f.name]
    flair = [f for f in files if '_flair.nii.gz' in f.name]
    
    if not all([t1, t1ce, t2, flair]):
        print("\n[ERROR] Missing MRI modalities!")
        print(f"Found files: {[f.name for f in files]}")
        print("\nRequired files:")
        print("  - *_t1.nii.gz")
        print("  - *_t1ce.nii.gz")
        print("  - *_t2.nii.gz")
        print("  - *_flair.nii.gz")
        sys.exit(1)
    
    print("\n[OK] Found all 4 MRI modalities:")
    print(f"  T1:    {t1[0].name}")
    print(f"  T1CE:  {t1ce[0].name}")
    print(f"  T2:    {t2[0].name}")
    print(f"  FLAIR: {flair[0].name}")
    
    return patient_folder


def run_segmentation(input_folder, output_folder):
    """
    Run brain tumor segmentation using BraTS 2021 winning model
    
    The model uses ensemble of 2 trainers with 5-fold cross-validation
    """
    setup_env()
    
    input_folder = prepare_input_folder(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Create temporary folders for each model's output
    temp_output1 = output_folder / "temp_model1"
    temp_output2 = output_folder / "temp_model2"
    temp_output1.mkdir(exist_ok=True)
    temp_output2.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("RUNNING ENSEMBLE SEGMENTATION (2 Models)")
    print("=" * 70)
    
    # Model 1: nnUNetTrainerV2BraTSRegions_DA4_BN_BD
    print("\n[1/3] Running Model 1: nnUNetTrainerV2BraTSRegions_DA4_BN_BD")
    print("-" * 70)
    
    cmd1 = [
        "nnUNet_predict",
        "-i", str(input_folder),
        "-o", str(temp_output1),
        "-t", "500",
        "-m", "3d_fullres",
        "-tr", "nnUNetTrainerV2BraTSRegions_DA4_BN_BD",
        "--save_npz"
    ]
    
    print(f"Command: {' '.join(cmd1)}")
    result1 = subprocess.run(cmd1, capture_output=False, text=True)
    
    if result1.returncode != 0:
        print("\n[ERROR] Model 1 failed!")
        sys.exit(1)
    
    print("[OK] Model 1 complete!")
    
    # Model 2: nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm
    print("\n[2/3] Running Model 2: largeUnet_Groupnorm")
    print("-" * 70)
    
    cmd2 = [
        "nnUNet_predict",
        "-i", str(input_folder),
        "-o", str(temp_output2),
        "-t", "500",
        "-m", "3d_fullres",
        "-tr", "nnUNetTrainerV2BraTSRegions_DA4_BN_BD_largeUnet_Groupnorm",
        "--save_npz"
    ]
    
    print(f"Command: {' '.join(cmd2)}")
    result2 = subprocess.run(cmd2, capture_output=False, text=True)
    
    if result2.returncode != 0:
        print("\n[ERROR] Model 2 failed!")
        sys.exit(1)
    
    print("[OK] Model 2 complete!")
    
    # Ensemble the results
    print("\n[3/3] Ensembling predictions...")
    print("-" * 70)
    
    cmd_ensemble = [
        "nnUNet_ensemble",
        "-f", str(temp_output1), str(temp_output2),
        "-o", str(output_folder)
    ]
    
    print(f"Command: {' '.join(cmd_ensemble)}")
    result_ensemble = subprocess.run(cmd_ensemble, capture_output=False, text=True)
    
    if result_ensemble.returncode != 0:
        print("\n[ERROR] Ensemble failed!")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("SEGMENTATION COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_folder}")
    print("\nOutput files:")
    for f in output_folder.glob("*.nii.gz"):
        print(f"  - {f.name}")
    
    # Calculate volumes
    try:
        calculate_volumes(output_folder)
    except Exception as e:
        print(f"\n[INFO] Volume calculation skipped: {e}")


def calculate_volumes(output_folder):
    """Calculate tumor volumes from segmentation mask"""
    import nibabel as nib
    import numpy as np
    
    output_folder = Path(output_folder)
    seg_files = list(output_folder.glob("*.nii.gz"))
    
    if not seg_files:
        print("[WARNING] No segmentation files found")
        return
    
    seg_file = seg_files[0]
    
    print("\n" + "=" * 70)
    print("TUMOR VOLUME ANALYSIS")
    print("=" * 70)
    
    # Load segmentation
    seg_nii = nib.load(seg_file)
    seg_data = seg_nii.get_fdata()
    
    # Get voxel dimensions (mm)
    voxel_dims = seg_nii.header.get_zooms()
    voxel_volume_mm3 = np.prod(voxel_dims)  # Volume of one voxel in mm³
    voxel_volume_cm3 = voxel_volume_mm3 / 1000  # Convert to cm³
    
    # Count voxels for each label
    # BraTS labels: 0=background, 1=necrotic, 2=edema, 4=enhancing
    ncr_voxels = np.sum(seg_data == 1)  # Necrotic tumor core
    ed_voxels = np.sum(seg_data == 2)   # Peritumoral edema
    et_voxels = np.sum(seg_data == 4)   # Enhancing tumor
    
    # Calculate volumes in cm³
    ncr_volume = ncr_voxels * voxel_volume_cm3
    ed_volume = ed_voxels * voxel_volume_cm3
    et_volume = et_voxels * voxel_volume_cm3
    
    # Calculate combined regions
    tc_volume = ncr_volume + et_volume  # Tumor core
    wt_volume = ncr_volume + ed_volume + et_volume  # Whole tumor
    
    print(f"\nSegmentation file: {seg_file.name}")
    print(f"Voxel size: {voxel_dims[0]:.2f} x {voxel_dims[1]:.2f} x {voxel_dims[2]:.2f} mm")
    print(f"Voxel volume: {voxel_volume_mm3:.2f} mm³")
    
    print("\n" + "-" * 70)
    print("TUMOR REGIONS:")
    print("-" * 70)
    print(f"  Necrotic Core (NCR):         {ncr_volume:>10.2f} cm³")
    print(f"  Peritumoral Edema (ED):      {ed_volume:>10.2f} cm³")
    print(f"  Enhancing Tumor (ET):        {et_volume:>10.2f} cm³")
    print("-" * 70)
    print(f"  Tumor Core (TC = NCR + ET):  {tc_volume:>10.2f} cm³")
    print(f"  Whole Tumor (WT = all):      {wt_volume:>10.2f} cm³")
    print("=" * 70)
    
    # Save results to text file
    results_file = output_folder / "volume_analysis.txt"
    with open(results_file, 'w') as f:
        f.write("BraTS 2021 Tumor Segmentation - Volume Analysis\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Segmentation file: {seg_file.name}\n")
        f.write(f"Voxel size: {voxel_dims[0]:.2f} x {voxel_dims[1]:.2f} x {voxel_dims[2]:.2f} mm\n\n")
        f.write("Tumor Regions:\n")
        f.write(f"  Necrotic Core (NCR):         {ncr_volume:>10.2f} cm³\n")
        f.write(f"  Peritumoral Edema (ED):      {ed_volume:>10.2f} cm³\n")
        f.write(f"  Enhancing Tumor (ET):        {et_volume:>10.2f} cm³\n\n")
        f.write(f"  Tumor Core (TC = NCR + ET):  {tc_volume:>10.2f} cm³\n")
        f.write(f"  Whole Tumor (WT = all):      {wt_volume:>10.2f} cm³\n")
    
    print(f"\n[OK] Volume analysis saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run brain tumor segmentation')
    parser.add_argument('--input', required=True, help='Input folder with MRI scans')
    parser.add_argument('--output', required=True, help='Output folder for segmentation')
    
    args = parser.parse_args()
    
    run_segmentation(args.input, args.output)
