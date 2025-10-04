"""
Simple inference script for brain tumor segmentation
Usage: python run_inference.py --input <patient_folder> --output <output_folder>
"""
import os
import sys
from pathlib import Path
import argparse


def setup_env():
    """Setup nnU-Net environment"""
    project_dir = Path(__file__).parent.absolute()
    os.environ['nnUNet_raw'] = str(project_dir / "nnUNet_raw")
    os.environ['nnUNet_preprocessed'] = str(project_dir / "nnUNet_preprocessed")
    os.environ['nnUNet_results'] = str(project_dir / "nnUNet_results")


def run_segmentation(input_folder, output_folder):
    """
    Run brain tumor segmentation on patient MRI scans
    
    Args:
        input_folder: Folder containing T1, T1CE, T2, FLAIR .nii.gz files
        output_folder: Where to save segmentation results
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    
    setup_env()
    
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    
    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device='cuda',
        verbose=True,
        allow_tqdm=True
    )
    
    # Load pretrained model
    model_path = Path(os.environ['nnUNet_results']) / "Dataset001_BraTS2024" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    
    predictor.initialize_from_trained_model_folder(
        str(model_path),
        use_folds=(0,),  # Use fold 0
        checkpoint_name='checkpoint_final.pth'
    )
    
    # Run prediction
    predictor.predict_from_files(
        input_folder,
        output_folder,
        save_probabilities=False,
        overwrite=True
    )
    
    print(f"[OK] Segmentation complete! Results saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run brain tumor segmentation')
    parser.add_argument('--input', required=True, help='Input folder with MRI scans')
    parser.add_argument('--output', required=True, help='Output folder for segmentation')
    
    args = parser.parse_args()
    
    run_segmentation(args.input, args.output)
