"""
Clean up and reorganize the AI-Powered Brain MRI Assistant workspace.
Creates a neat folder structure and moves files appropriately.
"""

import os
import shutil
from pathlib import Path


def reorganize_workspace():
    """Reorganize workspace with clean folder structure"""
    
    project_dir = Path(__file__).parent.absolute()
    
    print("=" * 80)
    print("REORGANIZING WORKSPACE")
    print("=" * 80)
    
    # Define new folder structure
    folders = {
        'docs': 'Documentation and guides',
        'scripts': 'Python scripts for inference and utilities',
        'models': 'Pre-trained model weights',
        'data': 'Input data and datasets',
        'results': 'Segmentation outputs',
        'visualizations': 'Plots and comparison images',
        'config': 'Configuration files'
    }
    
    # Create folders
    print("\n📁 Creating folder structure...")
    for folder, desc in folders.items():
        folder_path = project_dir / folder
        folder_path.mkdir(exist_ok=True)
        print(f"   ✓ {folder}/ - {desc}")
    
    # Move markdown files to docs/
    print("\n📄 Moving documentation files...")
    md_files = list(project_dir.glob("*.md"))
    for md_file in md_files:
        if md_file.name not in ['README.md']:  # Keep README in root
            dest = project_dir / 'docs' / md_file.name
            if not dest.exists():
                shutil.move(md_file, dest)
                print(f"   ✓ {md_file.name} → docs/")
    
    # Move Python scripts (keep main ones in root, move utilities to scripts/)
    print("\n🐍 Organizing Python scripts...")
    
    # Scripts to keep in root (main entry points)
    keep_in_root = [
        'reorganize_workspace.py',
        'run_brats2021_inference.py',
        'run_brats2021_inference_singlethread.py'
    ]
    
    # Move other scripts to scripts/
    py_files = list(project_dir.glob("*.py"))
    for py_file in py_files:
        if py_file.name not in keep_in_root:
            dest = project_dir / 'scripts' / py_file.name
            if not dest.exists() and py_file.name != 'reorganize_workspace.py':
                try:
                    shutil.move(py_file, dest)
                    print(f"   ✓ {py_file.name} → scripts/")
                except:
                    print(f"   ⚠️  Could not move {py_file.name}")
    
    # Move existing results to results/ (if not already there)
    print("\n📊 Organizing results...")
    # Results are already in results/ folder, so just confirm
    results_dir = project_dir / 'results'
    if results_dir.exists():
        print(f"   ✓ Results directory exists with {len(list(results_dir.iterdir()))} items")
    
    # Organize data folders
    print("\n💾 Organizing data...")
    data_folders = ['sample_data', 'temp_inference_input', 'temp_nnunet_input', 
                    'temp_inference_output1']
    for folder in data_folders:
        src = project_dir / folder
        if src.exists():
            dest = project_dir / 'data' / folder
            if not dest.exists():
                try:
                    shutil.move(src, dest)
                    print(f"   ✓ {folder}/ → data/")
                except:
                    print(f"   ⚠️  Could not move {folder}")
    
    # Move nnUNet folders to models/
    print("\n🤖 Organizing model data...")
    model_folders = ['nnUNet_results', 'downloads/trained_models']
    # Keep nnUNet_results, nnUNet_raw, nnUNet_preprocessed in root (needed by nnUNet)
    print("   ℹ️  Keeping nnUNet folders in root (required by nnUNet framework)")
    
    # Clean up temporary files
    print("\n🧹 Cleaning up temporary files...")
    temp_patterns = ['*.pyc', '__pycache__', '*.tmp', '.ipynb_checkpoints']
    for pattern in temp_patterns:
        for item in project_dir.rglob(pattern):
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
                print(f"   ✓ Removed: {item.relative_to(project_dir)}")
            except:
                pass
    
    # Create a clean README for docs folder
    docs_readme = project_dir / 'docs' / 'README.md'
    if not docs_readme.exists():
        with open(docs_readme, 'w') as f:
            f.write("""# Documentation

This folder contains all documentation and guides for the AI-Powered Brain MRI Assistant.

## Files

- **ACCURACY_EVALUATION_GUIDE.md** - Guide for evaluating segmentation accuracy
- **WHY_LOW_ACCURACY.md** - Troubleshooting low accuracy issues
- **get_real_sample_data.md** - Guide for obtaining real BraTS data
- **instructions.md** - General usage instructions
- **MULTIPROCESSING_FIX.md** - Fixing Windows multiprocessing issues
- **PYTHON_310_SETUP.md** - Python 3.10 setup guide

## Quick Links

- Main README: `../README.md`
- Sample Data: `../data/sample_data/`
- Results: `../results/`
""")
        print(f"   ✓ Created docs/README.md")
    
    # Create scripts README
    scripts_readme = project_dir / 'scripts' / 'README.md'
    if not scripts_readme.exists():
        with open(scripts_readme, 'w') as f:
            f.write("""# Scripts

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
""")
        print(f"   ✓ Created scripts/README.md")
    
    print("\n" + "=" * 80)
    print("WORKSPACE REORGANIZATION COMPLETE!")
    print("=" * 80)
    
    print("\n📋 New Structure:")
    print("""
    AI-Powered Brain MRI Assistant/
    ├── README.md (main documentation)
    ├── requirements.txt
    │
    ├── docs/ (all markdown documentation)
    │   ├── README.md
    │   ├── ACCURACY_EVALUATION_GUIDE.md
    │   ├── WHY_LOW_ACCURACY.md
    │   └── ... (other .md files)
    │
    ├── scripts/ (utility scripts)
    │   ├── README.md
    │   ├── evaluate_segmentation.py
    │   ├── compare_segmentations.py
    │   ├── check_labels.py
    │   └── ... (other scripts)
    │
    ├── data/ (input data)
    │   ├── sample_data/
    │   ├── temp_inference_input/
    │   └── ... (downloaded datasets)
    │
    ├── results/ (segmentation outputs)
    │   ├── BraTS2021_00495/
    │   └── ...
    │
    ├── models/ (for future model storage)
    │
    ├── Brats21_KAIST_MRI_Lab/ (KAIST model code)
    ├── nnUNet_results/ (model weights - kept in root)
    ├── nnUNet_raw/ (raw data - kept in root)
    ├── nnUNet_preprocessed/ (preprocessed data - kept in root)
    ├── downloads/ (downloaded files)
    │
    ├── venv310/ (Python virtual environment)
    │
    └── Main inference scripts (in root):
        ├── run_brats2021_inference.py
        └── run_brats2021_inference_singlethread.py
    """)
    
    print("\n✅ Your workspace is now organized!")
    print("\nNote: nnUNet folders (nnUNet_results, nnUNet_raw, nnUNet_preprocessed)")
    print("      must stay in root directory as required by nnU-Net framework.")


if __name__ == "__main__":
    reorganize_workspace()
