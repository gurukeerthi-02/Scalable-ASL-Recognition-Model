"""
DYNAMIC DATASET MERGER
======================
Merges dynamic gesture data from multiple people

USAGE:
1. Place all ZIP files in same folder as this script
2. Run: python merge_dynamic_datasets.py
3. Use merged dataset for training

OUTPUT:
- ../dataset_dynamic_merged/ - Combined dataset
- dataset_report.txt - Statistics
"""

import os
import numpy as np
import zipfile
import shutil
from collections import defaultdict
import sys
import io

# ============================================
# CONFIGURATION
# ============================================

MERGED_DIR = "../dataset_dynamic_merged"
TEMP_EXTRACT_DIR = "temp_dynamic_extracted"
EXPECTED_SHAPE = (30, 68)  # Sequence length, features

# ============================================
# EXTRACT ZIP FILES
# ============================================

def extract_all_zips(directory="./zip_files"):
    """Extract all dynamic data ZIP files"""
    
    zip_files = [f for f in os.listdir(directory) 
                 if f.endswith('.zip')]
    
    if not zip_files:
        print("[ERR] No ZIP files found!")
        return []
    
    print(f"\nFound {len(zip_files)} ZIP files:", flush=True)
    for zf in zip_files:
        print(f"  - {zf}")
    
    extracted_dirs = []
    
    for zip_file in zip_files:
        print(f"\nExtracting {zip_file}...")
        
        full_zip_path = os.path.join(directory, zip_file)
        with zipfile.ZipFile(full_zip_path, 'r') as zip_ref:
            extract_path = os.path.join(TEMP_EXTRACT_DIR, zip_file.replace('.zip', ''))
            zip_ref.extractall(extract_path)
            extracted_dirs.append(extract_path)
            print(f"  [OK] Extracted to {extract_path}")
    
    return extracted_dirs

# ============================================
# VALIDATE SEQUENCES
# ============================================

def validate_sequence(file_path, expected_shape):
    """Check if sequence file is valid"""
    try:
        sequence = np.load(file_path)
        
        if sequence.shape != expected_shape:
            return False, f"Wrong shape: {sequence.shape} (expected {expected_shape})"
        
        if np.isnan(sequence).any():
            return False, "Contains NaN values"
        
        if np.isinf(sequence).any():
            return False, "Contains infinite values"
        
        return True, "OK"
        
    except Exception as e:
        return False, str(e)

# ============================================
# MERGE DATASETS
# ============================================

def merge_datasets(extracted_dirs):
    """Merge all datasets into unified structure"""
    
    print(f"\nMerging datasets into {MERGED_DIR}...")
    
    # Create merged directory
    if os.path.exists(MERGED_DIR):
        print(f"  [WARN] {MERGED_DIR} already exists. Removing...")
        shutil.rmtree(MERGED_DIR)
    os.makedirs(MERGED_DIR)
    
    stats = defaultdict(lambda: defaultdict(int))
    errors = defaultdict(list)
    
    for extract_dir in extracted_dirs:
        # Find data directory
        data_dir = extract_dir
        for item in os.listdir(extract_dir):
            item_path = os.path.join(extract_dir, item)
            if os.path.isdir(item_path) and 'asl_dynamic' in item:
                data_dir = item_path
                break
        
        # Get person ID
        person_id = os.path.basename(data_dir).replace('asl_dynamic_', '')
        print(f"\n  Processing: {person_id}")
        
        # Process each gesture
        for gesture in os.listdir(data_dir):
            gesture_path = os.path.join(data_dir, gesture)
            
            if not os.path.isdir(gesture_path):
                continue
            
            # Create gesture directory in merged dataset
            merged_gesture_dir = os.path.join(MERGED_DIR, gesture)
            os.makedirs(merged_gesture_dir, exist_ok=True)
            
            # Copy and validate all .npy files
            valid_count = 0
            invalid_count = 0
            
            for file in os.listdir(gesture_path):
                if not file.endswith('.npy'):
                    continue
                
                src = os.path.join(gesture_path, file)
                
                # Validate before copying
                is_valid, message = validate_sequence(src, EXPECTED_SHAPE)
                
                if is_valid:
                    # Rename with person prefix
                    dst = os.path.join(merged_gesture_dir, f"{person_id}_{file}")
                    shutil.copy2(src, dst)
                    valid_count += 1
                else:
                    invalid_count += 1
                    errors[gesture].append(f"{person_id}/{file}: {message}")
            
            stats[gesture][person_id] = valid_count
            
            if invalid_count > 0:
                print(f"    {gesture}: {valid_count} valid, {invalid_count} invalid")
            else:
                print(f"    {gesture}: {valid_count} samples")
    
    return stats, errors

# ============================================
# GENERATE REPORT
# ============================================

def generate_report(stats, errors):
    """Generate detailed statistics report"""
    
    report_path = os.path.join(MERGED_DIR, "dataset_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DYNAMIC DATASET MERGE REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall stats
        total_samples = sum(sum(persons.values()) for persons in stats.values())
        total_people = len(set(person for persons in stats.values() for person in persons.keys()))
        total_gestures = len(stats)
        
        f.write(f"Total People:    {total_people}\n")
        f.write(f"Total Gestures:  {total_gestures}\n")
        f.write(f"Total Sequences: {total_samples:,}\n")
        f.write(f"Avg per Gesture: {total_samples // total_gestures if total_gestures > 0 else 0}\n\n")
        
        # People summary
        f.write("-"*70 + "\n")
        f.write("CONTRIBUTORS\n")
        f.write("-"*70 + "\n")
        
        people_totals = defaultdict(int)
        for gesture, persons in stats.items():
            for person, count in persons.items():
                people_totals[person] += count
        
        for person, total in sorted(people_totals.items()):
            f.write(f"{person:20s}: {total:5d} sequences\n")
        
        # Per-gesture breakdown
        f.write("\n" + "-"*70 + "\n")
        f.write("PER-GESTURE BREAKDOWN\n")
        f.write("-"*70 + "\n\n")
        
        for gesture in sorted(stats.keys()):
            f.write(f"Gesture '{gesture}':\n")
            total_for_gesture = sum(stats[gesture].values())
            f.write(f"  Total: {total_for_gesture} sequences\n")
            f.write(f"  Contributors: {len(stats[gesture])}\n")
            for person, count in sorted(stats[gesture].items()):
                f.write(f"    - {person}: {count} sequences\n")
            f.write("\n")
        
        # Errors
        if errors:
            f.write("="*70 + "\n")
            f.write("VALIDATION ERRORS\n")
            f.write("="*70 + "\n\n")
            
            for gesture, error_list in errors.items():
                if error_list:
                    f.write(f"{gesture}:\n")
                    for error in error_list[:5]:  # Show first 5 errors
                        f.write(f"  [ERR] {error}\n")
                    if len(error_list) > 5:
                        f.write(f"  ... and {len(error_list) - 5} more\n")
                    f.write("\n")
        
        # Recommendations
        f.write("="*70 + "\n")
        f.write("TRAINING RECOMMENDATIONS\n")
        f.write("="*70 + "\n\n")
        
        if total_people >= 3:
            f.write("[GOOD]: Data from 3+ people\n")
            f.write("  Suggested split:\n")
            people_list = list(people_totals.keys())
            f.write(f"    Training:   {', '.join(people_list[:-2])}\n")
            f.write(f"    Validation: {people_list[-2]}\n")
            f.write(f"    Testing:    {people_list[-1]}\n\n")
        else:
            f.write(f"[WARN]: Only {total_people} contributor(s)\n")
            f.write("  Recommendation: Collect from 2-3 more people\n\n")
        
        # Check balance
        gesture_counts = {gesture: sum(persons.values()) for gesture, persons in stats.items()}
        if gesture_counts:
            min_count = min(gesture_counts.values())
            max_count = max(gesture_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 2:
                f.write("[WARN]: Gesture imbalance detected\n")
                f.write(f"  Min: {min_count}, Max: {max_count}\n")
                f.write("  Consider collecting more for underrepresented gestures\n\n")
            else:
                f.write("[GOOD]: Dataset is well balanced\n\n")
        
        # Expected performance
        f.write("="*70 + "\n")
        f.write("EXPECTED MODEL PERFORMANCE\n")
        f.write("="*70 + "\n\n")
        
        if total_samples >= 400:  # 100 per gesture × 4 gestures
            f.write("[EXCELLENT]: Sufficient data for good model\n")
            f.write("  Expected test accuracy: 75-85%\n\n")
        elif total_samples >= 200:
            f.write("[GOOD]: Adequate data\n")
            f.write("  Expected test accuracy: 70-80%\n\n")
        else:
            f.write("[WARN]: Limited data\n")
            f.write("  Expected test accuracy: 60-70%\n")
            f.write("  Recommendation: Collect more samples\n\n")
    
    print(f"\nReport saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("MERGE SUMMARY")
    print("="*70)
    print(f"Total People:     {total_people}")
    print(f"Total Gestures:   {total_gestures}")
    print(f"Total Sequences:  {total_samples:,}")
    print(f"Avg per Person:   {total_samples // total_people if total_people > 0 else 0}")
    
    if errors:
        total_errors = sum(len(e) for e in errors.values())
        print(f"\n[WARN] Validation errors: {total_errors}")
        print("  Check dataset_report.txt for details")
    
    print("="*70)

# ============================================
# CLEANUP
# ============================================

def cleanup():
    """Remove temporary files"""
    if os.path.exists(TEMP_EXTRACT_DIR):
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(TEMP_EXTRACT_DIR)
        print(f"  [OK] Removed {TEMP_EXTRACT_DIR}")

# ============================================
# MAIN
# ============================================

def main():
    """Main merge workflow"""
    
    print("\n" + "="*70)
    print("DYNAMIC ASL DATASET MERGER")
    print("="*70)
    
    # Extract ZIP files
    extracted_dirs = extract_all_zips()
    
    if not extracted_dirs:
        print("\n[ERR] No data to merge!")
        print("\nMake sure you have ZIP files in the current directory:")
        print("  asl_dynamic_vishal_20260203.zip")
        print("  asl_dynamic_guru_20260203.zip")
        print("  etc.")
        return
    
    # Merge datasets
    stats, errors = merge_datasets(extracted_dirs)
    
    # Generate report
    generate_report(stats, errors)
    
    # Cleanup
    cleanup()
    
    print("\n" + "="*70)
    print("MERGE COMPLETE!")
    print("="*70)
    print(f"\nMerged dataset: {MERGED_DIR}/")
    print("\nNext steps:")
    print("  1. Review dataset_report.txt")
    print("  2. Run: python train_dynamic_model.py")
    print("  3. Test with: python test_dynamic_model.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERR] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(TEMP_EXTRACT_DIR):
            shutil.rmtree(TEMP_EXTRACT_DIR)