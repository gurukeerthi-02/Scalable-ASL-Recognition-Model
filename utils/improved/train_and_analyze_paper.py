
"""
ASL RECOGNITION - PAPER RESULTS GENERATOR
=========================================
This script trains the model (ensuring a fixed person-based split)
and generates comprehensive statistical data and plots for research papers.

OUTPUTS (in 'paper_results' folder):
1. training_history_plot.png (Accuracy/Loss curves)
2. confusion_matrix.png (Heatmap)
3. classification_report.csv (Precision, Recall, F1 per class)
4. model_performance_summary.txt (Textual summary)
5. per_class_accuracy.png

USAGE:
python train_and_analyze_paper.py
"""

import os
import time
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# ============================================
# REPRODUCIBILITY SETUP
# ============================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================
# CONFIGURATION
# ============================================

DATASET_PATH = "../dataset_merged"
OUTPUT_DIR = "paper_results"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "final_model_paper.h5")
AUGMENT_TRAINING = True
AUGMENTATIONS_PER_SAMPLE = 2

# ============================================
# MODE SELECTION
# ============================================
LOAD_EXISTING_MODEL = False  # Set to True to generate stats for an existing model
# Path to the model you want to analyze (e.g., in ../models/ or paper_results/)
EXISTING_MODEL_PATH = "../models/static_model_person_split_v1.h5" 


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# DATA LOADING & SPLITTING
# ============================================

def load_data_by_person(dataset_path):
    """Load data tracking person IDs"""
    data_by_person = defaultdict(lambda: {'X': [], 'y': []})
    labels = sorted([d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d))])
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    print(f"Loading data from {dataset_path}...")
    
    label_counts = defaultdict(int)
    
    for label in labels:
        label_path = os.path.join(dataset_path, label)
        if not os.path.exists(label_path): continue
            
        files = [f for f in os.listdir(label_path) if f.endswith('.npy')]
        for file in files:
            person_id = file.split('_')[0]
            file_path = os.path.join(label_path, file)
            features = np.load(file_path)
            
            data_by_person[person_id]['X'].append(features)
            data_by_person[person_id]['y'].append(label_map[label])
            label_counts[label] += 1
            
    # Convert to numpy arrays
    for person_id in data_by_person:
        data_by_person[person_id]['X'] = np.array(data_by_person[person_id]['X'])
        data_by_person[person_id]['y'] = np.array(data_by_person[person_id]['y'])
        
    return data_by_person, labels, label_map, label_counts

def split_by_person(data_by_person):
    """Split data ensuring different people in train/val/test"""
    people = list(data_by_person.keys())
    # Sort then shuffle with fixed seed for reproducibility
    people.sort() 
    random.shuffle(people)
    
    num_people = len(people)
    print(f"\nFound {num_people} participants: {', '.join(people)}")
    
    if num_people < 3:
        print("⚠ WARNING: Fewer than 3 people. Using random split (Not ideal for paper).")
        return split_randomly(data_by_person)
    
    # Split Config: 60% Train, 20% Val, 20% Test (approx)
    num_test = max(1, int(num_people * 0.2))
    num_val = max(1, int(num_people * 0.2))
    num_train = num_people - num_test - num_val
    
    train_people = people[:num_train]
    val_people = people[num_train:num_train + num_val]
    test_people = people[num_train + num_val:]
    
    print(f"\nSPLIT ASSIGNMENT:")
    print(f"  Training ({len(train_people)}):   {train_people}")
    print(f"  Validation ({len(val_people)}): {val_people}")
    print(f"  Testing ({len(test_people)}):    {test_people}")
    
    def combine_data(person_list):
        X, y = [], []
        for p in person_list:
            X.extend(data_by_person[p]['X'])
            y.extend(data_by_person[p]['y'])
        return np.array(X), np.array(y)
    
    X_train, y_train = combine_data(train_people)
    X_val, y_val = combine_data(val_people)
    X_test, y_test = combine_data(test_people)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def split_randomly(data_by_person):
    """Fallback random split"""
    X_all, y_all = [], []
    for person in data_by_person:
        X_all.extend(data_by_person[person]['X'])
        y_all.extend(data_by_person[person]['y'])
    
    X_all, y_all = np.array(X_all), np.array(y_all)
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

def augment_training_data(X_train, y_train):
    if not AUGMENT_TRAINING: return X_train, y_train
    
    print(f"\nAugmenting training data ({AUGMENTATIONS_PER_SAMPLE}x)...")
    X_aug, y_aug = [], []
    
    for i in range(len(X_train)):
        X_aug.append(X_train[i])
        y_aug.append(y_train[i])
        for _ in range(AUGMENTATIONS_PER_SAMPLE):
            X_aug.append(augment_sample(X_train[i]))
            y_aug.append(y_train[i])
            
    return np.array(X_aug), np.array(y_aug)

# ============================================
# MODEL BUILDING
# ============================================

def build_model(num_classes):
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ============================================
# VISUALIZATION & REPORTING
# ============================================

def plot_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy over Epochs', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss over Epochs', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_training_history.png'), dpi=300)
    plt.close()
    print("✓ Saved training history plot")

def plot_confusion_matrix(y_true, y_pred, labels):
    """Generate confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8)) # Fixed size
    
    # Calculate percentage based
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_confusion_matrix.png'), dpi=300)
    plt.close()
    print("✓ Saved confusion matrix")

def save_classification_report(y_true, y_pred, labels):
    """Save detailed classification metrics"""
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Save CSV
    report_df.to_csv(os.path.join(OUTPUT_DIR, '3_classification_report.csv'))
    
    # Also save as pretty text
    report_str = classification_report(y_true, y_pred, target_names=labels)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report_str)
        
    print("✓ Saved classification report")
    return report_df

def plot_per_class_accuracy(report_df):
    """Plot F1-score for each class"""
    # Filter out accuracy/macro avg/weighted avg rows for plotting
    class_rows = report_df.iloc[:-3] if 'accuracy' in report_df.index else report_df
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_rows.index, class_rows['f1-score'], color='skyblue', edgecolor='black')
    
    plt.title('F1-Score per Class (Model Robustness)', fontsize=15)
    plt.xlabel('ASL Gestures')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1.1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_per_class_performance.png'), dpi=300)
    plt.close()
    print("✓ Saved per-class performance plot")

# ============================================
# MAIN
# ============================================

def main():
    start_global = time.time()
    print("\n" + "="*70)
    print("ASL RESEARCH DATA GENERATOR")
    print(f"Saving all results to: {os.path.abspath(OUTPUT_DIR)}")
    print("="*70)
    
    # 1. Load Data
    data_by_person, labels, label_map, label_counts = load_data_by_person(DATASET_PATH)
    if not data_by_person:
        print("ERROR: No data found.")
        return

    # 2. Split Data
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_person(data_by_person)
    
    print(f"\nDATASET DISTRIBUTION:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # 3. Augment
    X_train_aug, y_train_aug = augment_training_data(X_train, y_train)
    
    # 4. Encodings
    num_classes = len(labels)
    y_train_cat = to_categorical(y_train_aug, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # 5. Train or Load Model
    history = None
    
    if LOAD_EXISTING_MODEL and os.path.exists(EXISTING_MODEL_PATH):
        print(f"\nLOADING MODEL FROM: {EXISTING_MODEL_PATH}")
        try:
            model = load_model(EXISTING_MODEL_PATH)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return
    else:
        print("\nTRAINING MODEL...")
        if LOAD_EXISTING_MODEL:
            print(f"⚠ WARNING: Could not find model at {EXISTING_MODEL_PATH}. Training new one instead.")
            
        model = build_model(num_classes)
        
        history = model.fit(
            X_train_aug, y_train_cat,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val_cat),
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5),
                ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
            ],
            verbose=1
        )
    
    # 6. Generate Results
    print("\nGENERATING RESEARCH ASSETS...")
    
    # Training Curves
    if history:
        plot_history(history)
    else:
        print("⚠ Skipping training curves (using pre-trained model)")
    
    # Prediction on Test Set
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, labels)
    
    # Classification Report
    report_df = save_classification_report(y_test, y_pred, labels)
    
    # Per-class Plot
    plot_per_class_accuracy(report_df)
    
    # Summary File
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    
    val_acc_str = "N/A (Pre-trained)"
    train_acc_str = "N/A (Pre-trained)"
    gap_str = "N/A"
    
    if history:
        val_acc_str = f"{max(history.history['val_accuracy'])*100:.2f}%"
        train_acc_str = f"{history.history['accuracy'][-1]*100:.2f}%"
        gap_str = f"{(history.history['accuracy'][-1] - test_acc)*100:.2f}%"
    
    summary = [
        "ASL RECOGNITION MODEL - RESEARCH SUMMARY",
        "========================================",
        f"Date Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model Source: {'Pre-trained (' + EXISTING_MODEL_PATH + ')' if LOAD_EXISTING_MODEL else 'Newly Trained'}",
        f"Dataset Path: {DATASET_PATH}",
        "",
        "DATASET STATISTICS",
        "------------------",
        f"Total Classes: {num_classes}",
        f"Total Samples (Original): {sum(len(d['X']) for d in data_by_person.values())}",
        f"Training Samples (Augmented): {len(X_train_aug)}",
        f"Test Samples: {len(X_test)}",
        "",
        "MODEL PERFORMANCE",
        "-----------------",
        f"Test Accuracy: {test_acc*100:.2f}%",
        f"Test Loss: {test_loss:.4f}",
        f"Best Val Accuracy: {val_acc_str}",
        f"Final Train Accuracy: {train_acc_str}",
        "",
        "OVERFITTING ANALYSIS",
        "--------------------",
        f"Train-Test Gap: {gap_str}",
    ]
    
    with open(os.path.join(OUTPUT_DIR, 'model_performance_summary.txt'), 'w') as f:
        f.write('\n'.join(summary))
        
    print(f"\n✓ All results saved to folder: {OUTPUT_DIR}")
    print(f"Total time: {(time.time() - start_global):.1f}s")
    print("="*70)

if __name__ == "__main__":
    main()
