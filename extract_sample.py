"""
Extract a single sample case from Task01_BrainTumour.tar
This extracts only ONE case instead of the entire 5GB archive
"""
import tarfile
from pathlib import Path
import os


def extract_single_sample(tar_path, output_dir, num_cases=1):
    """
    Extract only a few sample cases from the tar file
    
    Args:
        tar_path: Path to Task01_BrainTumour.tar
        output_dir: Where to extract the files
        num_cases: Number of sample cases to extract (default: 1)
    """
    
    print("=" * 70)
    print("EXTRACTING SAMPLE CASE FROM TAR FILE")
    print("=" * 70)
    
    tar_path = Path(tar_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not tar_path.exists():
        print(f"[ERROR] Tar file not found: {tar_path}")
        print("\nPlease specify the correct path to Task01_BrainTumour.tar")
        return None
    
    print(f"\n[OK] Found tar file: {tar_path}")
    print(f"[OK] Output directory: {output_dir}")
    print(f"\nExtracting {num_cases} sample case(s)...")
    print("This will take 1-2 minutes...\n")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # List all files in the archive
            members = tar.getmembers()
            
            # Find image files (not labels)
            image_files = [m for m in members if 'imagesTr' in m.name and m.name.endswith('.nii.gz')]
            
            if not image_files:
                print("[ERROR] No image files found in the archive")
                print("Archive contents:")
                for m in members[:10]:
                    print(f"  - {m.name}")
                return None
            
            print(f"[OK] Found {len(image_files)} image files in archive")
            print(f"\nExtracting first {num_cases} case(s):\n")
            
            extracted_files = []
            for i, member in enumerate(image_files[:num_cases]):
                print(f"  [{i+1}/{num_cases}] Extracting: {member.name}")
                tar.extract(member, output_dir)
                extracted_files.append(output_dir / member.name)
            
            print(f"\n[OK] Extraction complete!")
            print(f"\nExtracted files:")
            for f in extracted_files:
                print(f"  - {f}")
            
            return extracted_files
            
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract sample from Medical Decathlon tar")
    parser.add_argument("--tar_path", type=str, 
                       default="downloads/Task01_BrainTumour.tar",
                       help="Path to Task01_BrainTumour.tar file")
    parser.add_argument("--output", type=str,
                       default="sample_data",
                       help="Output directory for extracted files")
    parser.add_argument("--num_cases", type=int, default=1,
                       help="Number of cases to extract (default: 1)")
    
    args = parser.parse_args()
    
    # Get the project directory
    project_dir = Path(__file__).parent.absolute()
    
    # Build full paths
    if not Path(args.tar_path).is_absolute():
        tar_path = project_dir / args.tar_path
    else:
        tar_path = Path(args.tar_path)
    
    if not Path(args.output).is_absolute():
        output_dir = project_dir / args.output
    else:
        output_dir = Path(args.output)
    
    # Extract
    files = extract_single_sample(tar_path, output_dir, args.num_cases)
    
    if files:
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("\n1. The sample case has been extracted")
        print("2. Now we need to split the 4-channel file into separate modalities")
        print("3. Then we can run segmentation!")
        print("\nRun: python prepare_sample_for_inference.py")
    else:
        print("\n[ERROR] Extraction failed. Please check the file path.")
