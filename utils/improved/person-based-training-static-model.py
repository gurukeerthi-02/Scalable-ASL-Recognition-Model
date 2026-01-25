"""
PERSON-BASED TRAINING SCRIPT
=============================
Trains ASL model with proper person-based splitting to prevent overfitting
Ensures train/val/test sets contain different people

USAGE:
python train_person_based.py
"""

import numpy as np
import os
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import time

# ============================================
# CONFIGURATION
# ============================================

DATASET_PATH = "../dataset_merged"  # Use merged dataset
MODEL_SAVE_PATH = "../models/static_model_person_split_v1.h5"
AUGMENT_TRAINING = True  # Apply augmentation to training data only
AUGMENTATIONS_PER_SAMPLE = 2  # How many augmented versions per sample

# ============================================
# DATA AUGMENTATION
# ============================================

def augment_sample(features):
    """Apply random augmentation to a single sample"""
    aug = features.copy()
    
    # Random noise (±1.5%)
    noise = np.random.normal(0, 0.015, aug.shape)
    aug += noise
    
    # Random scale (90-110%)
    scale = np.random.uniform(0.90, 1.10)
    aug[:63] *= scale
    
    # Random rotation (-10° to +10°)
    angle = np.radians(np.random.uniform(-10, 10))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    for i in range(0, 63, 3):
        x, y = aug[i], aug[i+1]
        aug[i] = x * cos_a - y * sin_a
        aug[i+1] = x * sin_a + y * cos_a
    
    return aug

# ============================================
# LOAD DATA WITH PERSON TRACKING
# ============================================

def load_data_by_person(dataset_path):
    """Load data while tracking which person each sample belongs to"""
    
    data_by_person = defaultdict(lambda: {'X': [], 'y': []})
    labels = sorted([d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d))])
    
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    print("Loading data by person...")
    
    for label in labels:
        label_path = os.path.join(dataset_path, label)
        
        if not os.path.exists(label_path):
            continue
            
        files = [f for f in os.listdir(label_path) if f.endswith('.npy')]
        
        for file in files:
            # Extract person ID from filename (format: personID_XXX.npy)
            person_id = file.split('_')[0]
            
            # Load features
            file_path = os.path.join(label_path, file)
            features = np.load(file_path)
            
            data_by_person[person_id]['X'].append(features)
            data_by_person[person_id]['y'].append(label_map[label])
    
    # Convert lists to numpy arrays
    for person_id in data_by_person:
        data_by_person[person_id]['X'] = np.array(data_by_person[person_id]['X'])
        data_by_person[person_id]['y'] = np.array(data_by_person[person_id]['y'])
    
    return data_by_person, labels, label_map

# ============================================
# PERSON-BASED SPLIT
# ============================================

def split_by_person(data_by_person):
    """Split data ensuring different people in train/val/test"""
    
    people = list(data_by_person.keys())
    num_people = len(people)
    
    print(f"\nFound {num_people} different people: {', '.join(people)}")
    
    if num_people < 3:
        print("\n⚠ WARNING: You have data from fewer than 3 people!")
        print("  This is not ideal. Consider collecting from more people.")
        print("  For now, using random split with warning...")
        return split_randomly(data_by_person)
    
    # Assign people to splits
    # Recommendation: Use different people for each split
    # Example: If 5 people -> 3 train, 1 val, 1 test
    
    num_test = max(1, num_people // 5)  # 20% of people for test
    num_val = max(1, num_people // 5)   # 20% of people for val
    num_train = num_people - num_test - num_val
    
    # Shuffle people randomly
    import random
    random.shuffle(people)
    
    train_people = people[:num_train]
    val_people = people[num_train:num_train + num_val]
    test_people = people[num_train + num_val:]
    
    print(f"\nPerson-based split:")
    print(f"  Training:   {train_people} ({num_train} people)")
    print(f"  Validation: {val_people} ({num_val} people)")
    print(f"  Testing:    {test_people} ({num_test} people)")
    
    # Combine data from assigned people
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    for person in train_people:
        X_train.extend(data_by_person[person]['X'])
        y_train.extend(data_by_person[person]['y'])
    
    for person in val_people:
        X_val.extend(data_by_person[person]['X'])
        y_val.extend(data_by_person[person]['y'])
    
    for person in test_people:
        X_test.extend(data_by_person[person]['X'])
        y_test.extend(data_by_person[person]['y'])
    
    return (np.array(X_train), np.array(y_train),
            np.array(X_val), np.array(y_val),
            np.array(X_test), np.array(y_test))

def split_randomly(data_by_person):
    """Fallback: Random split when not enough people"""
    X_all, y_all = [], []
    
    for person in data_by_person:
        X_all.extend(data_by_person[person]['X'])
        y_all.extend(data_by_person[person]['y'])
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ============================================
# APPLY AUGMENTATION TO TRAINING SET
# ============================================

def augment_training_data(X_train, y_train):
    """Apply augmentation only to training data"""
    
    if not AUGMENT_TRAINING:
        return X_train, y_train
    
    print(f"\nApplying augmentation ({AUGMENTATIONS_PER_SAMPLE}x per sample)...")
    
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X_train)):
        # Add original
        X_augmented.append(X_train[i])
        y_augmented.append(y_train[i])
        
        # Add augmented versions
        for _ in range(AUGMENTATIONS_PER_SAMPLE):
            aug_sample = augment_sample(X_train[i])
            X_augmented.append(aug_sample)
            y_augmented.append(y_train[i])
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    print(f"  Training data: {len(X_train)} → {len(X_augmented)} samples")
    
    return X_augmented, y_augmented

# ============================================
# BUILD MODEL
# ============================================

def build_model(num_classes):
    """Build improved model with regularization"""
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(68,), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.6),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================
# MAIN TRAINING
# ============================================

def main():
    print("\n" + "="*70)
    print("PERSON-BASED ASL MODEL TRAINING")
    print("="*70)
    
    # Load data
    data_by_person, labels, label_map = load_data_by_person(DATASET_PATH)
    
    if not data_by_person:
        print("\n✗ ERROR: No data found!")
        print(f"  Make sure {DATASET_PATH} exists and contains data")
        return
    
    # Split by person
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_person(data_by_person)
    
    print(f"\nInitial split sizes:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Testing:    {len(X_test)} samples")
    
    # Augment training data
    X_train, y_train = augment_training_data(X_train, y_train)
    
    # One-hot encode labels
    num_classes = len(labels)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    
    print(f"\nFinal training set: {len(X_train)} samples")
    print(f"Number of classes: {num_classes}")
    print(f"Labels: {labels}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(num_classes)
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE - PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Training Time           : {training_time:.2f} minutes")
    print(f"Total Epochs            : {len(history.history['accuracy'])}")
    print(f"Best Epoch              : {best_epoch}")
    print("-" * 70)
    print(f"Training Accuracy       : {final_train_acc * 100:.2f}%")
    print(f"Validation Accuracy     : {final_val_acc * 100:.2f}%")
    print(f"Test Accuracy           : {test_accuracy * 100:.2f}%")
    print(f"Best Val Accuracy       : {best_val_acc * 100:.2f}%")
    print("-" * 70)
    
    # Overfitting analysis
    train_test_gap = (final_train_acc - test_accuracy) * 100
    train_val_gap = (final_train_acc - final_val_acc) * 100
    
    print(f"Train-Test Gap          : {train_test_gap:.2f}%")
    print(f"Train-Val Gap           : {train_val_gap:.2f}%")
    print("="*70)
    
    # Interpretation
    print("\nMODEL ASSESSMENT:")
    
    if test_accuracy < 0.5:
        print("  ✗ POOR: Test accuracy below 50%")
        print("    → Model is not learning properly")
        print("    → Check if data is corrupted or labels are wrong")
    elif test_accuracy < 0.7:
        print("  ⚠ WEAK: Test accuracy 50-70%")
        print("    → Model needs improvement")
        print("    → Try: more data, better augmentation, tune hyperparameters")
    elif test_accuracy < 0.85:
        print("  ✓ GOOD: Test accuracy 70-85%")
        print("    → Model is working reasonably well")
        print("    → Can improve with more diverse data")
    else:
        print("  ✓ EXCELLENT: Test accuracy 85%+")
        print("    → Model is performing very well!")
    
    print()
    
    if train_test_gap < 5:
        print("  ✓ EXCELLENT: Minimal overfitting (<5% gap)")
        print("    → Model generalizes well to new people")
    elif train_test_gap < 10:
        print("  ✓ GOOD: Slight overfitting (5-10% gap)")
        print("    → Acceptable for this task")
    elif train_test_gap < 15:
        print("  ⚠ WARNING: Moderate overfitting (10-15% gap)")
        print("    → Consider: more data, stronger regularization")
    else:
        print("  ✗ PROBLEM: Severe overfitting (>15% gap)")
        print("    → Definitely needs: more diverse training data")
        print("    → Try: increase dropout, reduce model size")
    
    print("\n" + "="*70)
    print("EXPECTED REAL-WORLD PERFORMANCE:")
    print("="*70)
    print(f"  Same person (training set):     ~{final_train_acc * 100:.0f}%")
    print(f"  Same person (different session): ~{test_accuracy * 0.95 * 100:.0f}%")
    print(f"  Different person (similar demo): ~{test_accuracy * 0.85 * 100:.0f}%")
    print(f"  Different demographics:          ~{test_accuracy * 0.70 * 100:.0f}%")
    print(f"  Different environment:           ~{test_accuracy * 0.60 * 100:.0f}%")
    print("="*70)
    
    print(f"\n✓ Model saved to: {MODEL_SAVE_PATH}")
    print("\nNext steps:")
    print("  1. Test the model with real-time prediction script")
    print("  2. Try it with different people")
    print("  3. If performance is poor, collect more diverse data")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()