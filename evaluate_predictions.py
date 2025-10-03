"""
Evaluation script for nnU-Net predictions on BraTS 2024
Computes Dice scores and other metrics
"""
import os
import json
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def compute_dice_score(pred, gt, label):
    """
    Compute Dice score for a specific label
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        label: Label value to compute Dice for
    
    Returns:
        Dice score (0-1)
    """
    pred_binary = (pred == label).astype(int)
    gt_binary = (gt == label).astype(int)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice


def compute_hausdorff_distance(pred, gt, label, spacing=(1.0, 1.0, 1.0)):
    """
    Compute Hausdorff distance for a specific label
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        label: Label value
        spacing: Voxel spacing
    
    Returns:
        Hausdorff distance in mm
    """
    from scipy.ndimage import distance_transform_edt
    
    pred_binary = (pred == label).astype(bool)
    gt_binary = (gt == label).astype(bool)
    
    # If either is empty, return inf
    if not np.any(pred_binary) or not np.any(gt_binary):
        return np.inf
    
    # Compute distance transforms
    pred_dist = distance_transform_edt(~pred_binary, sampling=spacing)
    gt_dist = distance_transform_edt(~gt_binary, sampling=spacing)
    
    # Hausdorff distance
    hd1 = np.max(pred_dist[gt_binary])
    hd2 = np.max(gt_dist[pred_binary])
    
    return max(hd1, hd2)


def evaluate_case(pred_path, gt_path):
    """
    Evaluate a single case
    
    Args:
        pred_path: Path to prediction
        gt_path: Path to ground truth
    
    Returns:
        Dictionary with metrics
    """
    # Load images
    pred_img = sitk.ReadImage(pred_path)
    gt_img = sitk.ReadImage(gt_path)
    
    pred = sitk.GetArrayFromImage(pred_img)
    gt = sitk.GetArrayFromImage(gt_img)
    spacing = pred_img.GetSpacing()
    
    # Compute metrics for individual labels
    metrics = {}
    
    # Individual regions
    for label, name in [(1, 'NCR'), (2, 'ED'), (3, 'ET')]:
        dice = compute_dice_score(pred, gt, label)
        metrics[f'Dice_{name}'] = dice
    
    # Composite regions
    # Whole Tumor (WT): all non-background
    pred_wt = (pred > 0).astype(int)
    gt_wt = (gt > 0).astype(int)
    metrics['Dice_WT'] = compute_dice_score(pred_wt, gt_wt, 1)
    
    # Tumor Core (TC): NCR + ET (labels 1 and 3)
    pred_tc = ((pred == 1) | (pred == 3)).astype(int)
    gt_tc = ((gt == 1) | (gt == 3)).astype(int)
    metrics['Dice_TC'] = compute_dice_score(pred_tc, gt_tc, 1)
    
    return metrics


def evaluate_folder(pred_folder, gt_folder, output_json=None):
    """
    Evaluate all predictions in a folder
    
    Args:
        pred_folder: Folder with predictions
        gt_folder: Folder with ground truth
        output_json: Path to save results JSON
    
    Returns:
        Dictionary with all results
    """
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)
    
    # Find all prediction files
    pred_files = sorted(pred_folder.glob("*.nii.gz"))
    
    results = {}
    all_metrics = []
    
    print(f"Evaluating {len(pred_files)} cases...")
    
    for pred_file in tqdm(pred_files):
        # Find corresponding ground truth
        case_name = pred_file.stem.replace('.nii', '')
        gt_file = gt_folder / pred_file.name
        
        if not gt_file.exists():
            print(f"Warning: Ground truth not found for {pred_file.name}")
            continue
        
        # Evaluate case
        try:
            metrics = evaluate_case(str(pred_file), str(gt_file))
            results[case_name] = metrics
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error evaluating {case_name}: {e}")
            continue
    
    # Compute average metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        results['average'] = avg_metrics
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    
    if 'average' in results:
        for metric, stats in results['average'].items():
            print(f"{metric}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Save results
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {output_json}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net predictions")
    parser.add_argument("--pred_folder", type=str, required=True,
                       help="Folder with prediction files")
    parser.add_argument("--gt_folder", type=str, required=True,
                       help="Folder with ground truth files")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    results = evaluate_folder(args.pred_folder, args.gt_folder, args.output)
