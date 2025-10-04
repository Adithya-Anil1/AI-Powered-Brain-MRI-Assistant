"""
Download pretrained BraTS 2021 KAIST model weights from Google Drive
and organize them for nnU-Net inference
"""
import os
import sys
from pathlib import Path
import gdown


def download_pretrained_weights():
    """Download the pretrained model weights from Google Drive"""
    
    project_dir = Path(__file__).parent.absolute()
    
    # Setup environment
    nnunet_results = project_dir / "nnUNet_results"
    os.environ['nnUNet_results'] = str(nnunet_results)
    
    print("=" * 70)
    print("DOWNLOADING BraTS 2021 KAIST PRETRAINED WEIGHTS")
    print("=" * 70)
    
    # Google Drive file ID from the README
    # URL: https://drive.google.com/file/d/1HZmWG4j2zQg0vVwBsTrpnuLOmtKCpix2/view?usp=sharing
    file_id = "1HZmWG4j2zQg0vVwBsTrpnuLOmtKCpix2"
    
    download_dir = project_dir / "downloads"
    download_dir.mkdir(exist_ok=True)
    
    output_zip = download_dir / "brats2021_pretrained_models.zip"
    
    print(f"\nDownloading from Google Drive...")
    print(f"File ID: {file_id}")
    print(f"Destination: {output_zip}")
    print("\nThis may take several minutes depending on your internet speed...")
    
    try:
        # Download using gdown with fuzzy mode for large files
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_zip), quiet=False, fuzzy=True)
        print("\n[OK] Download complete!")
        return output_zip
        
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\n" + "=" * 70)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 70)
        print("\n1. Visit: https://drive.google.com/file/d/1HZmWG4j2zQg0vVwBsTrpnuLOmtKCpix2/view?usp=sharing")
        print(f"\n2. Download the file manually")
        print(f"\n3. Save it to: {output_zip}")
        print(f"\n4. Run this script again to extract and organize the files")
        return None


def extract_and_organize(zip_path):
    """Extract the downloaded weights and organize for nnU-Net"""
    
    import zipfile
    import shutil
    
    project_dir = Path(__file__).parent.absolute()
    nnunet_results = project_dir / "nnUNet_results"
    
    print("\n" + "=" * 70)
    print("EXTRACTING AND ORGANIZING MODEL FILES")
    print("=" * 70)
    
    extract_dir = project_dir / "downloads" / "extracted"
    extract_dir.mkdir(exist_ok=True)
    
    print(f"\nExtracting to: {extract_dir}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print("[OK] Extraction complete!")
    
    # The extracted folder should contain Task500_BraTS_2021
    # We need to move it to nnUNet_results
    
    task_folders = list(extract_dir.glob("Task*"))
    
    if task_folders:
        source_folder = task_folders[0]
        dest_folder = nnunet_results / source_folder.name
        
        print(f"\nMoving model files:")
        print(f"  From: {source_folder}")
        print(f"  To: {dest_folder}")
        
        if dest_folder.exists():
            shutil.rmtree(dest_folder)
        
        shutil.move(str(source_folder), str(dest_folder))
        
        print("[OK] Model files organized!")
        print(f"\nPretrained model ready at: {dest_folder}")
        
        return dest_folder
    else:
        print("[ERROR] Could not find Task folder in extracted files")
        print(f"Contents of {extract_dir}:")
        for item in extract_dir.iterdir():
            print(f"  - {item.name}")
        return None


def verify_installation():
    """Verify the model files are correctly installed"""
    
    project_dir = Path(__file__).parent.absolute()
    nnunet_results = project_dir / "nnUNet_results"
    
    print("\n" + "=" * 70)
    print("VERIFYING INSTALLATION")
    print("=" * 70)
    
    # Look for model folders
    task_folders = list(nnunet_results.glob("Task*"))
    
    if not task_folders:
        print("[ERROR] No task folders found in nnUNet_results")
        return False
    
    for task_folder in task_folders:
        print(f"\n[OK] Found: {task_folder.name}")
        
        # Check for trainer folders
        trainer_folders = list(task_folder.glob("nnUNetTrainerV2*"))
        
        for trainer_folder in trainer_folders:
            print(f"  [OK] Trainer: {trainer_folder.name}")
            
            # Check for fold folders
            fold_folders = list(trainer_folder.glob("fold_*"))
            print(f"      Folds found: {len(fold_folders)}")
            
            for fold_folder in fold_folders:
                # Check for checkpoint files
                checkpoints = list(fold_folder.glob("*.pth"))
                if checkpoints:
                    print(f"      [OK] {fold_folder.name}: {len(checkpoints)} checkpoint(s)")
    
    print("\n[OK] Model installation verified!")
    return True


if __name__ == "__main__":
    
    # Check if gdown is installed
    try:
        import gdown
    except ImportError:
        print("[INFO] Installing gdown for Google Drive downloads...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    project_dir = Path(__file__).parent.absolute()
    zip_path = project_dir / "downloads" / "brats2021_pretrained_models.zip"
    
    # Check if already downloaded
    if zip_path.exists():
        print(f"[OK] Model file already downloaded: {zip_path}")
        print("Skipping download, proceeding to extraction...")
    else:
        # Download
        zip_path = download_pretrained_weights()
        
        if not zip_path:
            print("\n[ERROR] Could not download automatically.")
            print("Please download manually and rerun this script.")
            sys.exit(1)
    
    # Extract and organize
    model_folder = extract_and_organize(zip_path)
    
    if model_folder:
        # Verify
        verify_installation()
        
        print("\n" + "=" * 70)
        print("SETUP COMPLETE!")
        print("=" * 70)
        print("\nYour BraTS 2021 KAIST pretrained model is ready to use!")
        print("\nNext steps:")
        print("1. Prepare your MRI data (T1, T1CE, T2, FLAIR)")
        print("2. Run inference using: python inference_nnunet.py")
        print("\nFor more details, see: Brats21_KAIST_MRI_Lab/readme.md")
    else:
        print("\n[ERROR] Setup incomplete. Please check the error messages above.")
