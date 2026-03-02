"""
DYNAMIC GESTURE MODEL - LEAVE-ONE-PERSON-OUT (LOPO) CROSS-VALIDATION
=====================================================================
Trains N folds (one per participant), evaluates generalization,
and saves the best-performing fold model for deployment.

OUTPUTS (in 'paper_results_dynamic_lopo' folder):
1. 1_lopo_confusion_matrix_aggregated.png
2. 2_lopo_fold_accuracies.png
3. 3_lopo_per_class_performance.png
4. 4_lopo_classification_report.csv
5. lopo_classification_report.txt
6. lopo_per_fold_results.txt
7. lopo_summary.txt
8. dynamic_model_final.h5  ← best fold model saved for deployment

USAGE:
python lopo_dynamic.py
"""

import os
import time
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# ============================================
# REPRODUCIBILITY
# ============================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================
# CONFIGURATION
# ============================================
DATASET_PATH    = "../dataset_dynamic_merged"
OUTPUT_DIR      = "paper_results_dynamic_lopo"
SEQUENCE_LENGTH = 30
NUM_FEATURES    = 68
AUGMENT         = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# DATA AUGMENTATION
# ============================================

def augment_sequence(sequence):
    """Noise + scaling + small rotation applied to every frame"""
    aug = sequence.copy()
    aug += np.random.normal(0, 0.015, aug.shape)
    scale = np.random.uniform(0.90, 1.10)
    aug[:, :63] *= scale
    angle = np.radians(np.random.uniform(-8, 8))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    for f in range(len(aug)):
        for i in range(0, 63, 3):
            x, y = aug[f, i], aug[f, i + 1]
            aug[f, i]     = x * cos_a - y * sin_a
            aug[f, i + 1] = x * sin_a + y * cos_a
    return aug


def augment_data(X, y):
    """2x augmentation: original + 1 augmented copy per sequence"""
    if not AUGMENT:
        return X, y
    X_aug, y_aug = [], []
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
        X_aug.append(augment_sequence(X[i]))
        y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)

# ============================================
# DATA LOADING
# ============================================

def load_dynamic_data(dataset_path):
    """Load .npy sequence files, grouped by person"""
    data_by_person = defaultdict(lambda: {'X': [], 'y': []})

    gestures  = sorted([d for d in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, d))])
    label_map = {g: i for i, g in enumerate(gestures)}

    print("Loading dynamic data...")
    print(f"Gestures: {gestures}")

    for gesture in gestures:
        gesture_path = os.path.join(dataset_path, gesture)
        files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
        for file in files:
            person_id = file.split('_')[0]
            try:
                seq = np.load(os.path.join(gesture_path, file))
                if seq.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                    print(f"  ⚠ Skipping {file}: wrong shape {seq.shape}")
                    continue
                data_by_person[person_id]['X'].append(seq)
                data_by_person[person_id]['y'].append(label_map[gesture])
            except Exception as e:
                print(f"  ✗ Error loading {file}: {e}")

    for pid in data_by_person:
        data_by_person[pid]['X'] = np.array(data_by_person[pid]['X'])
        data_by_person[pid]['y'] = np.array(data_by_person[pid]['y'])
        print(f"  {pid}: {len(data_by_person[pid]['X'])} sequences")

    return data_by_person, gestures, label_map

# ============================================
# MODEL
# ============================================

def build_model(num_classes):
    """Compact Bidirectional LSTM — rebuilt fresh for each fold"""
    tf.random.set_seed(SEED)
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True,
                           kernel_regularizer=l2(0.001)),
                      input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        Dropout(0.5),

        Bidirectional(LSTM(32, return_sequences=False,
                           kernel_regularizer=l2(0.001))),
        Dropout(0.5),

        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================
# LOPO CROSS-VALIDATION
# ============================================

def run_lopo(data_by_person, gestures):
    """Run Leave-One-Person-Out cross-validation across all participants"""
    people      = sorted(data_by_person.keys())
    num_classes = len(gestures)
    num_folds   = len(people)

    print(f"\n{'='*70}")
    print(f"LEAVE-ONE-PERSON-OUT CROSS-VALIDATION")
    print(f"Participants: {people}")
    print(f"Folds: {num_folds}  |  Classes: {num_classes}")
    print(f"{'='*70}\n")

    fold_results   = []
    all_y_true     = []
    all_y_pred     = []
    fold_log_lines = []

    for fold_idx, test_person in enumerate(people):
        fold_num     = fold_idx + 1
        train_people = [p for p in people if p != test_person]

        print(f"\n{'─'*70}")
        print(f"FOLD {fold_num}/{num_folds}")
        print(f"  Test:  [{test_person}]  ← completely unseen")
        print(f"  Train: {train_people}")
        print(f"{'─'*70}")

        # ── Combine all non-test people into training pool ───────────────
        X_pool, y_pool = [], []
        for p in train_people:
            X_pool.extend(data_by_person[p]['X'])
            y_pool.extend(data_by_person[p]['y'])
        X_pool = np.array(X_pool)
        y_pool = np.array(y_pool)

        # ── 80/20 val split within training pool ────────────────────────
        X_train, X_val, y_train, y_val = train_test_split(
            X_pool, y_pool,
            test_size=0.2, random_state=SEED, stratify=y_pool
        )

        # ── Test = entire held-out person's data ────────────────────────
        X_test = data_by_person[test_person]['X']
        y_test = data_by_person[test_person]['y']

        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        # ── Augment training data only ───────────────────────────────────
        X_train_aug, y_train_aug = augment_data(X_train, y_train)
        print(f"  After augmentation: {len(X_train_aug)} training sequences")

        # ── One-hot encode ───────────────────────────────────────────────
        y_train_cat = to_categorical(y_train_aug, num_classes)
        y_val_cat   = to_categorical(y_val,       num_classes)
        y_test_cat  = to_categorical(y_test,      num_classes)

        # ── Train model ──────────────────────────────────────────────────
        fold_model_path = os.path.join(OUTPUT_DIR, f"fold_{fold_num}_best.h5")
        model = build_model(num_classes)

        history = model.fit(
            X_train_aug, y_train_cat,
            epochs=100,
            batch_size=16,
            validation_data=(X_val, y_val_cat),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=15,
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=8, min_lr=1e-6, verbose=0),
                ModelCheckpoint(fold_model_path, monitor='val_accuracy',
                                save_best_only=True, verbose=0)
            ],
            verbose=1
        )

        # ── Evaluate on unseen test person ───────────────────────────────
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        y_pred     = np.argmax(model.predict(X_test, verbose=0), axis=1)
        best_val   = max(history.history['val_accuracy'])
        epochs_run = len(history.history['accuracy'])
        best_epoch = np.argmax(history.history['val_accuracy']) + 1

        print(f"\n  ── Fold {fold_num} Results ──────────────────────────")
        print(f"  Test Accuracy: {test_acc*100:.2f}%")
        print(f"  Best Val Acc:  {best_val*100:.2f}%")
        print(f"  Epochs:        {epochs_run}  (best at epoch {best_epoch})")

        fold_results.append({
            'fold':        fold_num,
            'test_person': test_person,
            'test_acc':    test_acc,
            'best_val':    best_val,
            'epochs':      epochs_run,
            'best_epoch':  best_epoch,
            'model_path':  fold_model_path,
        })

        # Accumulate all predictions for aggregate confusion matrix
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        fold_log_lines += [
            f"\nFOLD {fold_num}: Test person = {test_person}",
            f"  Train people:    {train_people}",
            f"  Train sequences: {len(X_train_aug)} (after aug)",
            f"  Val sequences:   {len(X_val)}",
            f"  Test sequences:  {len(X_test)}",
            f"  Test Accuracy:   {test_acc*100:.2f}%",
            f"  Best Val Acc:    {best_val*100:.2f}%",
            f"  Epochs run:      {epochs_run} (best epoch: {best_epoch})",
        ]

    return fold_results, np.array(all_y_true), np.array(all_y_pred), fold_log_lines

# ============================================
# SAVE BEST MODEL
# ============================================

def save_best_model(fold_results):
    """
    Copy the best-performing fold model as the final deployment model.
    Cleans up all other fold model files.
    """
    best_fold     = max(fold_results, key=lambda r: r['test_acc'])
    best_src      = best_fold['model_path']
    final_path    = os.path.join(OUTPUT_DIR, "dynamic_model_final.h5")

    if os.path.exists(best_src):
        shutil.copy(best_src, final_path)
        print(f"\n✓ Best model saved:")
        print(f"  Fold {best_fold['fold']}  |  "
              f"Test person: {best_fold['test_person']}  |  "
              f"Accuracy: {best_fold['test_acc']*100:.2f}%")
        print(f"  → {os.path.abspath(final_path)}")
    else:
        print(f"\n⚠ Best fold model not found at: {best_src}")
        final_path = None

    # Clean up all fold model files
    for r in fold_results:
        if os.path.exists(r['model_path']):
            os.remove(r['model_path'])

    return final_path

# ============================================
# VISUALISATION
# ============================================

def plot_aggregated_confusion_matrix(y_true, y_pred, gestures):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(8, len(gestures)), max(6, len(gestures) - 1)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=gestures, yticklabels=gestures)
    plt.title('Aggregated Confusion Matrix (All LOPO Folds)', fontsize=15)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,
                '1_lopo_confusion_matrix_aggregated.png'), dpi=300)
    plt.close()
    print("✓ Saved aggregated confusion matrix")


def plot_fold_accuracies(fold_results):
    folds     = [f"Fold {r['fold']}\n({r['test_person'].split()[0]})"
                 for r in fold_results]
    test_accs = [r['test_acc'] * 100 for r in fold_results]
    mean_acc  = np.mean(test_accs)

    plt.figure(figsize=(max(8, len(fold_results) * 2), 6))
    colors = ['gold' if r['test_acc'] == max(
        x['test_acc'] for x in fold_results) else 'steelblue'
              for r in fold_results]
    bars = plt.bar(folds, test_accs, color=colors, edgecolor='black')
    plt.axhline(mean_acc, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_acc:.2f}%')

    for bar, acc in zip(bars, test_accs):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    plt.title('Test Accuracy per LOPO Fold\n(gold = best fold / saved model)',
              fontsize=13)
    plt.xlabel('Fold (Test Person)')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim(0, 115)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_lopo_fold_accuracies.png'), dpi=300)
    plt.close()
    print("✓ Saved per-fold accuracy chart")


def plot_per_class_performance(y_true, y_pred, gestures):
    report_dict = classification_report(
        y_true, y_pred, target_names=gestures, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()

    class_rows = report_df.drop(
        index=[r for r in ['accuracy', 'macro avg', 'weighted avg']
               if r in report_df.index]
    )

    plt.figure(figsize=(max(10, len(class_rows) * 1.3), 6))
    bars = plt.bar(class_rows.index, class_rows['f1-score'],
                   color='steelblue', edgecolor='black')
    plt.title('F1-Score per Gesture Class (Aggregated LOPO)', fontsize=15)
    plt.xlabel('ASL Gesture')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1.15)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 0.02,
                 f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,
                '3_lopo_per_class_performance.png'), dpi=300)
    plt.close()
    print("✓ Saved per-class performance chart")

    return report_df

# ============================================
# REPORTS
# ============================================

def save_reports(y_true, y_pred, gestures, fold_results,
                 report_df, fold_log_lines, final_model_path, t0):

    # CSV
    report_df.to_csv(os.path.join(OUTPUT_DIR,
                     '4_lopo_classification_report.csv'))

    # TXT
    report_str = classification_report(y_true, y_pred, target_names=gestures)
    with open(os.path.join(OUTPUT_DIR,
              'lopo_classification_report.txt'), 'w') as f:
        f.write(report_str)
    print("✓ Saved classification report")

    # Per-fold log
    with open(os.path.join(OUTPUT_DIR,
              'lopo_per_fold_results.txt'), 'w') as f:
        f.write("LOPO CROSS-VALIDATION — PER FOLD RESULTS\n")
        f.write("=" * 55 + "\n")
        f.write('\n'.join(fold_log_lines))
    print("✓ Saved per-fold log")

    # Summary
    test_accs = [r['test_acc'] for r in fold_results]
    mean_acc  = np.mean(test_accs)
    std_acc   = np.std(test_accs)
    min_acc   = np.min(test_accs)
    max_acc   = np.max(test_accs)
    best_fold = max(fold_results, key=lambda r: r['test_acc'])

    class_rows = report_df.drop(
        index=[r for r in ['accuracy', 'macro avg', 'weighted avg']
               if r in report_df.index]
    )

    summary = [
        "DYNAMIC ASL GESTURE RECOGNITION — LOPO SUMMARY",
        "=" * 55,
        f"Date:                  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset:               {DATASET_PATH}",
        f"Sequence Length:       {SEQUENCE_LENGTH} frames",
        f"Features per Frame:    {NUM_FEATURES}",
        f"Gestures ({len(gestures)}):          {', '.join(gestures)}",
        f"Saved Model:           "
        f"{os.path.abspath(final_model_path) if final_model_path else 'Not saved'}",
        f"Best Fold:             Fold {best_fold['fold']} "
        f"({best_fold['test_person']}) — {best_fold['test_acc']*100:.2f}%",
        "",
        "LOPO CROSS-VALIDATION RESULTS",
        "-" * 55,
        f"Mean Test Accuracy:    {mean_acc*100:.2f}%",
        f"Std Dev:               ±{std_acc*100:.2f}%",
        f"Min:                   {min_acc*100:.2f}%",
        f"Max:                   {max_acc*100:.2f}%",
        "",
        "PER-FOLD BREAKDOWN",
        "-" * 55,
    ]

    for r in fold_results:
        marker = "  ← saved" if r['fold'] == best_fold['fold'] else ""
        summary.append(
            f"  Fold {r['fold']}  [{r['test_person']:<38}]  "
            f"Test: {r['test_acc']*100:.2f}%  "
            f"BestVal: {r['best_val']*100:.2f}%  "
            f"Epochs: {r['epochs']} (best: {r['best_epoch']}){marker}"
        )

    summary += [
        "",
        "PER-CLASS F1 SCORES (Aggregated across all folds)",
        "-" * 55,
    ]
    for cls in class_rows.index:
        summary.append(
            f"  {cls:<12}  "
            f"F1: {class_rows.loc[cls, 'f1-score']:.4f}  "
            f"Prec: {class_rows.loc[cls, 'precision']:.4f}  "
            f"Rec: {class_rows.loc[cls, 'recall']:.4f}"
        )

    summary += ["", f"Total Time: {(time.time() - t0):.1f}s"]

    with open(os.path.join(OUTPUT_DIR, 'lopo_summary.txt'), 'w') as f:
        f.write('\n'.join(summary))
    print("✓ Saved LOPO summary")

    return mean_acc, std_acc

# ============================================
# MAIN
# ============================================

def main():
    t0 = time.time()
    print("\n" + "=" * 70)
    print("DYNAMIC GESTURE — LOPO CROSS-VALIDATION")
    print(f"Results → {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)

    # 1. Load data
    data_by_person, gestures, label_map = load_dynamic_data(DATASET_PATH)
    if not data_by_person:
        print("ERROR: No data found. Run collect_dynamic_data.py first.")
        return

    # 2. Run LOPO
    fold_results, all_y_true, all_y_pred, fold_log_lines = run_lopo(
        data_by_person, gestures
    )

    # 3. Save best fold model for deployment
    print("\n" + "=" * 70)
    print("SAVING BEST MODEL")
    print("=" * 70)
    final_model_path = save_best_model(fold_results)

    # 4. Generate plots and reports
    print("\n" + "=" * 70)
    print("GENERATING RESEARCH ASSETS")
    print("=" * 70)
    plot_aggregated_confusion_matrix(all_y_true, all_y_pred, gestures)
    plot_fold_accuracies(fold_results)
    report_df = plot_per_class_performance(all_y_true, all_y_pred, gestures)
    mean_acc, std_acc = save_reports(
        all_y_true, all_y_pred, gestures,
        fold_results, report_df, fold_log_lines, final_model_path, t0
    )

    # 5. Final console summary
    print("\n" + "=" * 70)
    print("LOPO FINAL RESULTS")
    print("=" * 70)
    best_acc = max(r['test_acc'] for r in fold_results)
    for r in fold_results:
        marker = "  ← best (saved)" if r['test_acc'] == best_acc else ""
        print(f"  Fold {r['fold']}  "
              f"[{r['test_person']:<38}]  "
              f"Test: {r['test_acc']*100:.2f}%{marker}")
    print(f"  {'─'*60}")
    print(f"  Mean: {mean_acc*100:.2f}%  ±{std_acc*100:.2f}%")
    if final_model_path:
        print(f"\n✓ Deployment model → {os.path.abspath(final_model_path)}")
    print(f"✓ All outputs     → {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Total time: {(time.time() - t0):.1f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()