"""
DATASET MERGER
==============
Merges ASL data collected from multiple people into one unified dataset
Maintains person metadata for stratified splitting

USAGE:
1. Place all ZIP files in the same folder as this script
2. Run: python merge_datasets.py
3. Use the merged dataset for training
"""

import os
import numpy as np
import zipfile
import shutil
from collections import defaultdict

# ============================================
# CONFIGURATION
# ============================================

MERGED_DIR = "../dataset_merged"
TEMP_EXTRACT_DIR = "temp_extracted"

# ============================================
# EXTRACT ZIP FILES
# ============================================

def extract_all_zips(directory="."):
    """Extract all ZIP files in directory"""
    zip_files = [f for f in os.listdir(directory) if f.endswith('.zip') and 'asl_data' in f]
    
    if not zip_files:
        print("✗ No ZIP files found!")
        print("  Make sure ZIP files are in the same directory as this script")
        return []
    
    print(f"\n📦 Found {len(zip_files)} ZIP files:")
    for zf in zip_files:
        print(f"  • {zf}")
    
    extracted_dirs = []
    
    for zip_file in zip_files:
        print(f"\n📂 Extracting {zip_file}...")
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract to temp directory
            extract_path = os.path.join(TEMP_EXTRACT_DIR, zip_file.replace('.zip', ''))
            zip_ref.extractall(extract_path)
            extracted_dirs.append(extract_path)
            print(f"  ✓ Extracted to {extract_path}")
    
    return extracted_dirs

# ============================================
# MERGE DATASETS
# ============================================

def merge_datasets(extracted_dirs):
    """Merge all datasets into unified structure"""
    
    print(f"\n🔄 Merging datasets into {MERGED_DIR}...")
    
    # Create merged directory
    if os.path.exists(MERGED_DIR):
        print(f"  ⚠ {MERGED_DIR} already exists. Removing...")
        shutil.rmtree(MERGED_DIR)
    os.makedirs(MERGED_DIR)
    
    stats = defaultdict(lambda: defaultdict(int))
    
    for extract_dir in extracted_dirs:
        # Find the actual data directory (it might be nested)
        data_dir = extract_dir
        for item in os.listdir(extract_dir):
            item_path = os.path.join(extract_dir, item)
            if os.path.isdir(item_path) and 'asl_data' in item:
                data_dir = item_path
                break
        
        # Get person ID from directory name
        person_id = os.path.basename(data_dir).replace('asl_data_', '')
        print(f"\n  Processing: {person_id}")
        
        # Process each letter
        for label in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label)
            
            # Skip non-directories and info files
            if not os.path.isdir(label_path):
                continue
            
            # Create label directory in merged dataset
            merged_label_dir = os.path.join(MERGED_DIR, label)
            os.makedirs(merged_label_dir, exist_ok=True)
            
            # Copy all .npy files with person prefix
            file_count = 0
            for file in os.listdir(label_path):
                if file.endswith('.npy'):
                    src = os.path.join(label_path, file)
                    # Rename with person prefix to avoid collisions
                    dst = os.path.join(merged_label_dir, f"{person_id}_{file}")
                    shutil.copy2(src, dst)
                    file_count += 1
            
            stats[label][person_id] = file_count
            print(f"    {label}: {file_count} samples")
    
    return stats

# ============================================
# GENERATE REPORT
# ============================================

def generate_report(stats):
    """Generate detailed statistics report"""
    
    report_path = os.path.join(MERGED_DIR, "dataset_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MERGED DATASET REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        total_samples = sum(sum(persons.values()) for persons in stats.values())
        total_people = len(set(person for persons in stats.values() for person in persons.keys()))
        total_labels = len(stats)
        
        f.write(f"Total People:    {total_people}\n")
        f.write(f"Total Labels:    {total_labels}\n")
        f.write(f"Total Samples:   {total_samples:,}\n")
        f.write(f"Avg per Label:   {total_samples // total_labels if total_labels > 0 else 0}\n\n")
        
        # People summary
        f.write("-"*70 + "\n")
        f.write("CONTRIBUTORS\n")
        f.write("-"*70 + "\n")
        
        people_totals = defaultdict(int)
        for label, persons in stats.items():
            for person, count in persons.items():
                people_totals[person] += count
        
        for person, total in sorted(people_totals.items()):
            f.write(f"{person:20s}: {total:5d} samples\n")
        
        # Per-label breakdown
        f.write("\n" + "-"*70 + "\n")
        f.write("PER-LABEL BREAKDOWN\n")
        f.write("-"*70 + "\n\n")
        
        for label in sorted(stats.keys()):
            f.write(f"Letter '{label}':\n")
            total_for_label = sum(stats[label].values())
            f.write(f"  Total: {total_for_label} samples\n")
            f.write(f"  Contributors: {len(stats[label])}\n")
            for person, count in sorted(stats[label].items()):
                f.write(f"    • {person}: {count} samples\n")
            f.write("\n")
        
        # Recommendations
        f.write("="*70 + "\n")
        f.write("RECOMMENDATIONS FOR TRAINING\n")
        f.write("="*70 + "\n\n")
        
        if total_people >= 3:
            f.write("✓ GOOD: You have data from 3+ people\n")
            f.write("  Suggested split:\n")
            people_list = list(people_totals.keys())
            f.write(f"    Training:   {', '.join(people_list[:-2])}\n")
            f.write(f"    Validation: {people_list[-2]}\n")
            f.write(f"    Testing:    {people_list[-1]}\n\n")
        else:
            f.write("⚠ WARNING: Only data from {total_people} person/people\n")
            f.write("  Recommendation: Collect data from 2-3 more people\n\n")
        
        # Check balance
        label_counts = {label: sum(persons.values()) for label, persons in stats.items()}
        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 2:
            f.write("⚠ WARNING: Class imbalance detected\n")
            f.write(f"  Min samples: {min_count}, Max samples: {max_count}\n")
            f.write("  Consider collecting more data for underrepresented letters\n\n")
        else:
            f.write("✓ GOOD: Dataset is well balanced\n\n")
    
    print(f"\n📄 Report saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"Total People:    {total_people}")
    print(f"Total Labels:    {total_labels}")
    print(f"Total Samples:   {total_samples:,}")
    print(f"Avg per Person:  {total_samples // total_people if total_people > 0 else 0}")
    print("="*70)

# ============================================
# CREATE PERSON-BASED SPLITS
# ============================================

def create_split_info(stats):
    """Create file mapping samples to people for stratified splitting"""
    
    split_info_path = os.path.join(MERGED_DIR, "person_mapping.txt")
    
    with open(split_info_path, 'w', encoding='utf-8') as f:
        f.write("# Sample to Person Mapping\n")
        f.write("# Format: filename,person_id,label\n\n")
        
        for label in sorted(stats.keys()):
            for person in stats[label].keys():
                # List all files for this person/label
                label_dir = os.path.join(MERGED_DIR, label)
                for file in os.listdir(label_dir):
                    if file.startswith(person):
                        f.write(f"{file},{person},{label}\n")
    
    print(f"📋 Person mapping saved to: {split_info_path}")

# ============================================
# CLEANUP
# ============================================

def cleanup():
    """Remove temporary extraction directory"""
    if os.path.exists(TEMP_EXTRACT_DIR):
        print(f"\n🧹 Cleaning up temporary files...")
        shutil.rmtree(TEMP_EXTRACT_DIR)
        print(f"  ✓ Removed {TEMP_EXTRACT_DIR}")

# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "="*70)
    print("ASL DATASET MERGER")
    print("="*70)
    
    # Extract all ZIP files
    extracted_dirs = extract_all_zips()
    
    if not extracted_dirs:
        return
    
    # Merge datasets
    stats = merge_datasets(extracted_dirs)
    
    # Generate report
    generate_report(stats)
    
    # Create person mapping
    create_split_info(stats)
    
    # Cleanup
    cleanup()
    
    print("\n" + "="*70)
    print("✓ MERGE COMPLETE!")
    print("="*70)
    print(f"\nMerged dataset location: {MERGED_DIR}/")
    print("\nNext steps:")
    print("  1. Review dataset_report.txt for statistics")
    print("  2. Update your training script to use person-based splits")
    print("  3. Train your improved model!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup
        if os.path.exists(TEMP_EXTRACT_DIR):
            shutil.rmtree(TEMP_EXTRACT_DIR)