"""
QUICK DATASET HEALTH CHECK
===========================
Fast overview of dataset quality

USAGE:
python quick_dataset_check.py
"""

import numpy as np
import os
from collections import defaultdict

# Get absolute path to dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(os.path.join(script_dir, "../dataset_merged"))

def quick_check():
    """Quick health check of dataset"""
    
    print("\n" + "="*70)
    print("QUICK DATASET HEALTH CHECK")
    print("="*70)
    
    labels = sorted([d for d in os.listdir(DATASET_PATH) 
                    if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    data = defaultdict(lambda: defaultdict(list))
    issues = []
    
    print("\nScanning dataset...")
    
    total_files = 0
    corrupted_files = 0
    
    for label in labels:
        label_path = os.path.join(DATASET_PATH, label)
        files = [f for f in os.listdir(label_path) if f.endswith('.npy')]
        
        for file in files:
            total_files += 1
            person = file.split('_')[0]
            filepath = os.path.join(label_path, file)
            
            try:
                features = np.load(filepath)
                
                # Quick checks
                if features.shape != (68,):
                    issues.append(f"✗ {label}/{file}: Wrong shape {features.shape}")
                    corrupted_files += 1
                elif np.isnan(features).any():
                    issues.append(f"✗ {label}/{file}: Contains NaN")
                    corrupted_files += 1
                elif np.isinf(features).any():
                    issues.append(f"✗ {label}/{file}: Contains Inf")
                    corrupted_files += 1
                elif np.all(features == 0):
                    issues.append(f"✗ {label}/{file}: All zeros")
                    corrupted_files += 1
                else:
                    data[person][label].append(features)
                    
            except Exception as e:
                issues.append(f"✗ {label}/{file}: Load error - {e}")
                corrupted_files += 1
    
    print(f"✓ Scanned {total_files} files")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nTotal files: {total_files}")
    if total_files > 0:
        val_pct = (corrupted_files / total_files * 100)
    else:
        val_pct = 0.0
    print(f"Corrupted: {corrupted_files} ({val_pct:.1f}%)")
    print(f"Valid: {total_files - corrupted_files}")
    
    # Per-person summary
    print("\n" + "-"*70)
    print("PER-PERSON SUMMARY")
    print("-"*70)
    
    for person in sorted(data.keys()):
        total = sum(len(data[person][l]) for l in labels)
        letters = len([l for l in labels if data[person][l]])
        print(f"{person:15s}: {total:4d} samples, {letters:2d}/24 letters")
    
    # Per-letter summary
    print("\n" + "-"*70)
    print("PER-LETTER SUMMARY")
    print("-"*70)
    
    for label in labels:
        total = sum(len(data[p][label]) for p in data.keys())
        people = len([p for p in data.keys() if data[p][label]])
        
        if total == 0:
            status = "✗ MISSING"
        elif total < 100:
            status = f"⚠ LOW ({total})"
        else:
            status = f"✓ OK ({total})"
        
        print(f"{label:3s}: {status:20s} | {people} people")
    
    # Issues
    if issues:
        print("\n" + "="*70)
        print(f"ISSUES FOUND: {len(issues)}")
        print("="*70)
        
        for issue in issues[:20]:  # Show first 20
            print(issue)
        
        if len(issues) > 20:
            print(f"\n... and {len(issues) - 20} more issues")
        
        print("\nRun inspect_dataset_quality.py for detailed analysis")
    else:
        print("\n✓ No corruption issues found!")
    
    # Health score
    print("\n" + "="*70)
    print("HEALTH SCORE")
    print("="*70)
    
    corruption_score = 100 - (corrupted_files / total_files * 100) if total_files > 0 else 0
    
    # Check balance
    samples_per_letter = {l: sum(len(data[p][l]) for p in data.keys()) for l in labels}
    valid_letters = [s for s in samples_per_letter.values() if s > 0]
    
    if valid_letters:
        min_samples = min(valid_letters)
        max_samples = max(valid_letters)
        balance_ratio = min_samples / max_samples if max_samples > 0 else 0
        balance_score = balance_ratio * 100
    else:
        balance_score = 0
    
    # Check people coverage
    samples_per_person = {p: sum(len(data[p][l]) for l in labels) for p in data.keys()}
    people_coverage = len([p for p in samples_per_person.values() if p > 0])
    coverage_score = min(people_coverage / 3 * 100, 100)  # 3+ people ideal
    
    overall_score = (corruption_score * 0.5 + balance_score * 0.25 + coverage_score * 0.25)
    
    print(f"\nCorruption Score:  {corruption_score:.1f}% (higher is better)")
    print(f"Balance Score:     {balance_score:.1f}% (higher is better)")
    print(f"Coverage Score:    {coverage_score:.1f}% (3+ people)")
    print(f"\nOVERALL SCORE:     {overall_score:.1f}%")
    
    if overall_score >= 85:
        print("✓ EXCELLENT - Dataset is healthy")
    elif overall_score >= 70:
        print("✓ GOOD - Minor issues, usable for training")
    elif overall_score >= 50:
        print("⚠ FAIR - Some issues, review recommendations")
    else:
        print("✗ POOR - Significant issues, cleanup needed")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        quick_check()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()