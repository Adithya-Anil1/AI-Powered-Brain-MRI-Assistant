"""
Download sample brain tumor MRI data from Medical Decathlon
Task01 - Brain Tumour dataset
"""
import os
import sys
from pathlib import Path
import urllib.request
import tarfile


def download_medical_decathlon_brain():
    """Download Medical Decathlon Brain Tumor sample data"""
    
    project_dir = Path(__file__).parent.absolute()
    download_dir = project_dir / "sample_data"
    download_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("DOWNLOADING MEDICAL DECATHLON BRAIN TUMOR SAMPLE DATA")
    print("=" * 70)
    
    # Medical Decathlon Task01 Brain Tumor
    # Note: The full dataset is large, we'll guide manual download
    
    print("\n[INFO] Medical Decathlon Brain Tumor Dataset")
    print("\nTo download sample data:")
    print("\n1. Visit: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2")
    print("   (Medical Decathlon - Task01_BrainTumour)")
    print("\n2. Download 'Task01_BrainTumour.tar' (about 5 GB)")
    print("   OR download just the 'imagesTr' folder for samples")
    print("\n3. Extract to:")
    print(f"   {download_dir}")
    print("\n4. After extraction, you'll have:")
    print("   sample_data/")
    print("     └── Task01_BrainTumour/")
    print("         ├── imagesTr/")
    print("         ├── labelsTr/")
    print("         └── dataset.json")
    
    print("\n" + "=" * 70)
    print("ALTERNATIVE: Download Single Sample Case")
    print("=" * 70)
    print("\nFor quick testing, you can also:")
    print("\n1. Visit: https://github.com/neheller/kits19")
    print("   OR any public medical imaging dataset")
    print("\n2. Download ONE case with 4 modalities (T1, T1CE, T2, FLAIR)")
    print("\n3. Place files in:")
    print(f"   {download_dir / 'test_case' / 'case_001'}/")
    print("     ├── case_001_0000.nii.gz  (T1)")
    print("     ├── case_001_0001.nii.gz  (T1CE)")
    print("     ├── case_001_0002.nii.gz  (T2)")
    print("     └── case_001_0003.nii.gz  (FLAIR)")
    
    print("\n" + "=" * 70)
    print("EASIEST OPTION: Use BraTS Repository Sample")
    print("=" * 70)
    print("\nThe Brats21_KAIST_MRI_Lab repository may have sample data:")
    
    brats_repo = project_dir / "Brats21_KAIST_MRI_Lab"
    if brats_repo.exists():
        print(f"\n✓ Repository found at: {brats_repo}")
        print("\nCheck for sample/test data in the repository")
    else:
        print("\nRepository not found. Clone it with:")
        print("  git clone https://github.com/rixez/Brats21_KAIST_MRI_Lab.git")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS AFTER DOWNLOADING DATA:")
    print("=" * 70)
    print("\n1. Place your MRI files in: sample_data/test_case/")
    print("2. Run: python test_inference.py")
    print("\nI'll create a test_inference.py script for you next!")
    
    return download_dir


if __name__ == "__main__":
    download_dir = download_medical_decathlon_brain()
    
    print(f"\n[OK] Download directory ready: {download_dir}")
    print("\nFollow the instructions above to get sample data")
    print("Then we can test the segmentation model!")
