"""
Inference script for nnU-Net trained on BraTS 2024
Performs segmentation on new brain MRI scans
"""
import os
import sys
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def setup_environment():
    """Setup nnU-Net environment variables"""
    project_dir = Path(__file__).parent.absolute()
    
    os.environ['nnUNet_raw'] = str(project_dir / "nnUNet_raw")
    os.environ['nnUNet_preprocessed'] = str(project_dir / "nnUNet_preprocessed")
    os.environ['nnUNet_results'] = str(project_dir / "nnUNet_results")


def predict_single_case(input_files, output_path, dataset_id=1, 
                        configuration="3d_fullres", fold=0):
    """
    Run prediction on a single case
    
    Args:
        input_files: Dictionary with modality files {0: t1_path, 1: t1ce_path, 2: t2_path, 3: flair_path}
        output_path: Path to save the prediction
        dataset_id: Dataset ID
        configuration: Model configuration
        fold: Fold to use for prediction
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    
    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device='cuda',
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # Initialize the network architecture and load checkpoint
    predictor.initialize_from_trained_model_folder(
        os.path.join(os.environ['nnUNet_results'], 
                    f'Dataset{dataset_id:03d}_BraTS2024',
                    f'nnUNetTrainer__{configuration}',
                    f'fold_{fold}'),
        use_folds=(fold,),
        checkpoint_name='checkpoint_final.pth'
    )
    
    # Prepare input
    input_list = [input_files[i] for i in sorted(input_files.keys())]
    
    # Run prediction
    predictor.predict_from_files(
        [[input_list]],
        [output_path],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2
    )
    
    return output_path


def visualize_prediction(image_path, prediction_path, output_plot_path=None, slice_idx=None):
    """
    Visualize the prediction overlaid on the input image
    
    Args:
        image_path: Path to input image (e.g., T1CE)
        prediction_path: Path to prediction mask
        output_plot_path: Path to save the visualization
        slice_idx: Slice index to visualize (if None, uses middle slice)
    """
    # Load images
    image = sitk.ReadImage(image_path)
    prediction = sitk.ReadImage(prediction_path)
    
    image_array = sitk.GetArrayFromImage(image)
    pred_array = sitk.GetArrayFromImage(prediction)
    
    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = image_array.shape[0] // 2
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_array[slice_idx], cmap='gray')
    axes[0].set_title('Input Image (T1CE)')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(pred_array[slice_idx], cmap='jet', vmin=0, vmax=3)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image_array[slice_idx], cmap='gray')
    mask_overlay = np.ma.masked_where(pred_array[slice_idx] == 0, pred_array[slice_idx])
    axes[2].imshow(mask_overlay, cmap='jet', alpha=0.5, vmin=0, vmax=3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='NCR (1)'),
        Patch(facecolor='green', label='ED (2)'),
        Patch(facecolor='red', label='ET (3)')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_plot_path}")
    
    plt.show()


def calculate_tumor_volumes(prediction_path, spacing=None):
    """
    Calculate volumes of different tumor regions
    
    Args:
        prediction_path: Path to prediction mask
        spacing: Voxel spacing (if None, reads from image metadata)
    
    Returns:
        Dictionary with volumes in mm³
    """
    # Load prediction
    prediction = sitk.ReadImage(prediction_path)
    pred_array = sitk.GetArrayFromImage(prediction)
    
    # Get spacing
    if spacing is None:
        spacing = prediction.GetSpacing()
    
    voxel_volume = np.prod(spacing)  # mm³ per voxel
    
    # Calculate volumes
    volumes = {
        'NCR': np.sum(pred_array == 1) * voxel_volume,
        'ED': np.sum(pred_array == 2) * voxel_volume,
        'ET': np.sum(pred_array == 3) * voxel_volume
    }
    
    # Calculate composite regions
    volumes['WT'] = volumes['NCR'] + volumes['ED'] + volumes['ET']  # Whole tumor
    volumes['TC'] = volumes['NCR'] + volumes['ET']  # Tumor core
    
    return volumes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with nnU-Net")
    parser.add_argument("--t1", type=str, required=True, help="Path to T1 image")
    parser.add_argument("--t1ce", type=str, required=True, help="Path to T1CE image")
    parser.add_argument("--t2", type=str, required=True, help="Path to T2 image")
    parser.add_argument("--flair", type=str, required=True, help="Path to FLAIR image")
    parser.add_argument("--output", type=str, required=True, help="Output path for segmentation")
    parser.add_argument("--dataset_id", type=int, default=1, help="Dataset ID")
    parser.add_argument("--config", type=str, default="3d_fullres", help="Model configuration")
    parser.add_argument("--fold", type=int, default=0, help="Fold to use")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    parser.add_argument("--plot_output", type=str, help="Path to save visualization plot")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Prepare input files
    input_files = {
        0: args.t1,
        1: args.t1ce,
        2: args.t2,
        3: args.flair
    }
    
    print("Running inference...")
    print("=" * 60)
    
    # Run prediction
    output_path = predict_single_case(
        input_files, 
        args.output,
        args.dataset_id,
        args.config,
        args.fold
    )
    
    print(f"✓ Segmentation saved to: {output_path}")
    
    # Calculate volumes
    volumes = calculate_tumor_volumes(output_path)
    print("\nTumor volumes:")
    print(f"  Whole Tumor (WT): {volumes['WT']:.2f} mm³")
    print(f"  Tumor Core (TC): {volumes['TC']:.2f} mm³")
    print(f"  Enhancing Tumor (ET): {volumes['ET']:.2f} mm³")
    print(f"  Necrotic Core (NCR): {volumes['NCR']:.2f} mm³")
    print(f"  Edema (ED): {volumes['ED']:.2f} mm³")
    
    # Visualize if requested
    if args.visualize:
        visualize_prediction(args.t1ce, output_path, args.plot_output)
