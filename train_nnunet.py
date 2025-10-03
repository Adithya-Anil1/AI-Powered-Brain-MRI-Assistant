"""
Training script for nnU-Net on BraTS 2024 dataset
"""
import os
import subprocess
import sys
from pathlib import Path


def setup_environment():
    """Setup nnU-Net environment variables"""
    project_dir = Path(__file__).parent.absolute()
    
    os.environ['nnUNet_raw'] = str(project_dir / "nnUNet_raw")
    os.environ['nnUNet_preprocessed'] = str(project_dir / "nnUNet_preprocessed")
    os.environ['nnUNet_results'] = str(project_dir / "nnUNet_results")
    
    print("Environment variables set:")
    print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results: {os.environ['nnUNet_results']}")


def plan_and_preprocess(dataset_id=1, verify_integrity=True):
    """
    Run nnU-Net planning and preprocessing
    
    Args:
        dataset_id: Dataset ID (default: 1 for Dataset001_BraTS2024)
        verify_integrity: Whether to verify dataset integrity
    """
    print("\n" + "=" * 60)
    print("Running nnU-Net planning and preprocessing...")
    print("=" * 60)
    
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "--verify_dataset_integrity" if verify_integrity else ""
    ]
    
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Planning and preprocessing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during preprocessing: {e}")
        return False


def train_model(dataset_id=1, configuration="3d_fullres", fold=0, gpu_id=0):
    """
    Train nnU-Net model
    
    Args:
        dataset_id: Dataset ID
        configuration: Model configuration (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres)
        fold: Cross-validation fold (0-4 or 'all')
        gpu_id: GPU device ID
    """
    print("\n" + "=" * 60)
    print(f"Training nnU-Net model...")
    print(f"Dataset: {dataset_id}, Configuration: {configuration}, Fold: {fold}")
    print("=" * 60)
    
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        configuration,
        str(fold)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during training: {e}")
        return False


def predict(input_folder, output_folder, dataset_id=1, configuration="3d_fullres", 
            fold=0, checkpoint="checkpoint_final.pth"):
    """
    Run inference on new data
    
    Args:
        input_folder: Folder containing input images
        output_folder: Folder to save predictions
        dataset_id: Dataset ID
        configuration: Model configuration
        fold: Fold used for training
        checkpoint: Checkpoint file to use
    """
    print("\n" + "=" * 60)
    print("Running inference...")
    print("=" * 60)
    
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_folder),
        "-o", str(output_folder),
        "-d", str(dataset_id),
        "-c", configuration,
        "-f", str(fold),
        "-chk", checkpoint
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Predictions saved to: {output_folder}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during prediction: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train nnU-Net on BraTS 2024")
    parser.add_argument("--mode", type=str, choices=["preprocess", "train", "predict", "all"],
                       default="all", help="Mode to run")
    parser.add_argument("--dataset_id", type=int, default=1, help="Dataset ID")
    parser.add_argument("--config", type=str, default="3d_fullres",
                       choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
                       help="Model configuration")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--input_folder", type=str, help="Input folder for prediction")
    parser.add_argument("--output_folder", type=str, help="Output folder for prediction")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Run requested mode
    if args.mode in ["preprocess", "all"]:
        success = plan_and_preprocess(args.dataset_id)
        if not success and args.mode == "all":
            print("Preprocessing failed. Stopping.")
            sys.exit(1)
    
    if args.mode in ["train", "all"]:
        success = train_model(args.dataset_id, args.config, args.fold, args.gpu)
        if not success:
            print("Training failed.")
            sys.exit(1)
    
    if args.mode == "predict":
        if not args.input_folder or not args.output_folder:
            print("Error: --input_folder and --output_folder required for prediction mode")
            sys.exit(1)
        predict(args.input_folder, args.output_folder, args.dataset_id, 
               args.config, args.fold)
