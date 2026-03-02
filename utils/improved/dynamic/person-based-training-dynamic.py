"""
DYNAMIC GESTURE MODEL TRAINING (IMPROVED)
==========================================
Train LSTM model for sequential hand movements with fixes for overfitting:
  - Smaller, regularized model
  - Richer augmentation (noise, rotation, scaling, frame dropout, time warp)
  - Proper person-based split (no val/test person overlap)
  - Tighter early stopping

Model Architecture:
  - Input:  (30, 68) — 30 frames × 68 features
  - Bidirectional LSTM layers for temporal pattern recognition
  - Dense layers for classification

USAGE:
  python train_dynamic_model.py
"""

import numpy as np
import os
import random
import time
from collections import defaultdict

import tensorflow as tf
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH    = "../dataset_dynamic_merged"
MODEL_SAVE_PATH = "../models/dynamic_model_v2.h5"
SEQUENCE_LENGTH = 30   # Must match data collection
NUM_FEATURES    = 68

AUGMENT_COPIES  = 3    # Number of augmented copies per original sample
RANDOM_SEED     = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================
# AUGMENTATION HELPERS
# ============================================================

def add_noise(sequence, std=0.015):
    """Add Gaussian noise to all frames."""
    return sequence + np.random.normal(0, std, sequence.shape)


def random_scale(sequence, low=0.90, high=1.10):
    """Uniformly scale spatial features (first 63 dims)."""
    aug = sequence.copy()
    aug[:, :63] *= np.random.uniform(low, high)
    return aug


def random_rotation(sequence, max_deg=8):
    """Apply a small 2-D rotation to all frames."""
    aug = sequence.copy()
    angle = np.radians(np.random.uniform(-max_deg, max_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    for t in range(len(aug)):
        for i in range(0, 63, 3):
            x, y = aug[t, i], aug[t, i + 1]
            aug[t, i]     = x * cos_a - y * sin_a
            aug[t, i + 1] = x * sin_a + y * cos_a
    return aug


def frame_dropout(sequence, drop_rate=0.1):
    """Zero out random frames to simulate occlusion."""
    aug = sequence.copy()
    n_drop = max(1, int(len(aug) * drop_rate))
    drop_idx = np.random.choice(len(aug), n_drop, replace=False)
    aug[drop_idx] = 0.0
    return aug


def time_warp(sequence, factor_range=(0.8, 1.2)):
    """
    Resample the sequence to simulate speed variation.
    Stretches or compresses time by interpolating frames.
    """
    T = len(sequence)
    factor = np.random.uniform(*factor_range)
    src_len = max(2, int(T * factor))

    # Build a warped source at `src_len` frames then resample back to T
    src_t  = np.linspace(0, 1, src_len)
    orig_t = np.linspace(0, 1, T)

    aug = np.zeros_like(sequence)
    for feat in range(sequence.shape[1]):
        f = interp1d(orig_t, sequence[:, feat], kind='linear')
        src_vals = f(np.clip(src_t, 0, 1))
        g = interp1d(src_t, src_vals, kind='linear', fill_value='extrapolate')
        aug[:, feat] = g(orig_t)
    return aug


def augment_sequence(sequence):
    """
    Apply a combination of augmentations to one sequence.
    Order: noise → scale → rotation (always applied),
    then randomly apply time warp and/or frame dropout.
    """
    aug = sequence.copy()
    aug = add_noise(aug)
    aug = random_scale(aug)
    aug = random_rotation(aug)

    if np.random.rand() < 0.5:
        aug = time_warp(aug)

    if np.random.rand() < 0.5:
        drop_rate = np.random.uniform(0.05, 0.20)
        aug = frame_dropout(aug, drop_rate)

    return aug

# ============================================================
# DATA LOADING
# ============================================================

def load_dynamic_data(dataset_path):
    """Load .npy gesture sequences, grouped by person ID."""

    data_by_person = defaultdict(lambda: {'X': [], 'y': []})

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    gestures = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    if not gestures:
        raise ValueError("No gesture folders found in dataset path.")

    label_map = {g: i for i, g in enumerate(gestures)}

    print(f"\nGestures found ({len(gestures)}): {gestures}")
    print("Loading sequences...")

    total_loaded = 0
    total_skipped = 0

    for gesture in gestures:
        gesture_path = os.path.join(dataset_path, gesture)
        files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]

        for fname in files:
            person_id = fname.split('_')[0]
            fpath = os.path.join(gesture_path, fname)
            try:
                seq = np.load(fpath)
                if seq.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                    print(f"  ⚠  Skipping {fname}: shape {seq.shape}")
                    total_skipped += 1
                    continue
                data_by_person[person_id]['X'].append(seq)
                data_by_person[person_id]['y'].append(label_map[gesture])
                total_loaded += 1
            except Exception as e:
                print(f"  ✗  Error loading {fname}: {e}")
                total_skipped += 1

    for pid in data_by_person:
        data_by_person[pid]['X'] = np.array(data_by_person[pid]['X'])
        data_by_person[pid]['y'] = np.array(data_by_person[pid]['y'])
        print(f"  Person '{pid}': {len(data_by_person[pid]['X'])} sequences")

    print(f"\nTotal loaded: {total_loaded}  |  Skipped: {total_skipped}")
    return data_by_person, gestures, label_map

# ============================================================
# DATA SPLITTING  (no person overlap between val and test)
# ============================================================

def split_by_person(data_by_person):
    """
    Strict person-based split.

    ≥4 people  →  ~60% train | ~20% val | ~20% test  (all different people)
    3 people   →  2 train | 1 val | 1 test
                  (val and test come from the same held-out person,
                   split 50/50 by sample — unavoidable with only 3 people)
    <3 people  →  random fallback split

    NOTE: If you only have 3 people, please collect data from at least
    2 more people so that val and test can be fully independent.
    """

    people = list(data_by_person.keys())
    random.shuffle(people)
    n = len(people)

    print(f"\nFound {n} person(s): {people}")

    # ── ≥4 people: fully independent splits ──────────────────────────────
    if n >= 4:
        n_test  = max(1, n // 5)
        n_val   = max(1, n // 5)
        n_train = n - n_test - n_val

        train_people = people[:n_train]
        val_people   = people[n_train:n_train + n_val]
        test_people  = people[n_train + n_val:]

        print(f"\nPerson-based split (≥4-person mode):")
        print(f"  Train : {train_people}")
        print(f"  Val   : {val_people}")
        print(f"  Test  : {test_people}")

        def collect(plist):
            X, y = [], []
            for p in plist:
                X.extend(data_by_person[p]['X'])
                y.extend(data_by_person[p]['y'])
            return np.array(X), np.array(y)

        X_train, y_train = collect(train_people)
        X_val,   y_val   = collect(val_people)
        X_test,  y_test  = collect(test_people)

        return X_train, y_train, X_val, y_val, X_test, y_test

    # ── Exactly 3 people ─────────────────────────────────────────────────
    if n == 3:
        print("\n⚠  Only 3 people — val and test share 1 person (split 50/50).")
        print("   Collect ≥2 more people for fully independent evaluation.\n")

        train_people   = people[:2]
        held_person    = people[2]

        print(f"  Train : {train_people}")
        print(f"  Val/Test (shared): [{held_person}]")

        X_train = np.concatenate([data_by_person[p]['X'] for p in train_people])
        y_train = np.concatenate([data_by_person[p]['y'] for p in train_people])

        X_held  = data_by_person[held_person]['X']
        y_held  = data_by_person[held_person]['y']

        X_val, X_test, y_val, y_test = train_test_split(
            X_held, y_held, test_size=0.5,
            random_state=RANDOM_SEED, stratify=y_held
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    # ── <3 people: random fallback ────────────────────────────────────────
    print("\n⚠  Fewer than 3 people — using random split (results unreliable).")
    return split_randomly(data_by_person)


def split_randomly(data_by_person):
    """Fallback: random 60/20/20 split ignoring person identity."""
    X_all = np.concatenate([data_by_person[p]['X'] for p in data_by_person])
    y_all = np.concatenate([data_by_person[p]['y'] for p in data_by_person])

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_all, y_all, test_size=0.20,
        random_state=RANDOM_SEED, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25,
        random_state=RANDOM_SEED, stratify=y_tmp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

# ============================================================
# AUGMENTATION (training set only)
# ============================================================

def augment_training_sequences(X_train, y_train, copies=AUGMENT_COPIES):
    """
    Expand training set: keep every original and add `copies`
    independently augmented versions of each sequence.
    """
    print(f"\nAugmenting training data ({copies}× copies per sample)...")

    X_aug, y_aug = [X_train], [y_train]

    for _ in range(copies):
        batch = np.array([augment_sequence(x) for x in X_train])
        X_aug.append(batch)
        y_aug.append(y_train)

    X_aug = np.concatenate(X_aug, axis=0)
    y_aug = np.concatenate(y_aug, axis=0)

    # Shuffle
    idx = np.random.permutation(len(X_aug))
    X_aug, y_aug = X_aug[idx], y_aug[idx]

    print(f"  {len(X_train)} → {len(X_aug)} sequences")
    return X_aug, y_aug

# ============================================================
# MODEL  (smaller + stronger regularisation)
# ============================================================

def build_lstm_model(num_classes, sequence_length=SEQUENCE_LENGTH,
                     num_features=NUM_FEATURES):
    """
    Compact Bidirectional LSTM with stronger regularisation.
    Reduced LSTM units (32/16 instead of 64/32) to limit capacity,
    higher dropout and L2 to prevent overfitting.
    """
    reg = l2(0.005)

    model = Sequential([
        # ── LSTM block 1 ──
        Bidirectional(
            LSTM(32, return_sequences=True, kernel_regularizer=reg),
            input_shape=(sequence_length, num_features)
        ),
        Dropout(0.6),

        # ── LSTM block 2 ──
        Bidirectional(
            LSTM(16, return_sequences=False, kernel_regularizer=reg)
        ),
        Dropout(0.6),

        # ── Dense block ──
        Dense(32, activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        Dropout(0.5),

        # ── Output ──
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("DYNAMIC GESTURE MODEL TRAINING  (improved — v2)")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────
    data_by_person, gestures, label_map = load_dynamic_data(DATASET_PATH)

    if not data_by_person:
        print("\n✗  No data found! Run collect_dynamic_data.py first.")
        return

    # ── Split ─────────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_person(data_by_person)

    print(f"\nInitial split:")
    print(f"  Train : {len(X_train)} sequences")
    print(f"  Val   : {len(X_val)} sequences")
    print(f"  Test  : {len(X_test)} sequences")

    # ── Augment training only ─────────────────────────────────────────────
    X_train, y_train = augment_training_sequences(X_train, y_train)

    # ── One-hot encode ────────────────────────────────────────────────────
    num_classes  = len(gestures)
    y_train_cat  = to_categorical(y_train, num_classes)
    y_val_cat    = to_categorical(y_val,   num_classes)
    y_test_cat   = to_categorical(y_test,  num_classes)

    print(f"\nFinal training set : {len(X_train)} sequences")
    print(f"Sequence shape     : {X_train.shape}")
    print(f"Classes ({num_classes})        : {gestures}")

    # ── Build model ───────────────────────────────────────────────────────
    print("\nBuilding LSTM model...")
    model = build_lstm_model(num_classes)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,                   # tighter than before
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    start = time.time()

    history = model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, y_val_cat),
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    elapsed = (time.time() - start) / 60

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    final_train_acc = history.history['accuracy'][-1]
    final_val_acc   = history.history['val_accuracy'][-1]
    best_val_acc    = max(history.history['val_accuracy'])
    best_epoch      = int(np.argmax(history.history['val_accuracy'])) + 1
    total_epochs    = len(history.history['accuracy'])

    print(f"\nTraining Time    : {elapsed:.2f} min")
    print(f"Total Epochs     : {total_epochs}")
    print(f"Best Epoch       : {best_epoch}")
    print("-" * 70)
    print(f"Train Accuracy   : {final_train_acc * 100:.2f}%")
    print(f"Val Accuracy     : {final_val_acc   * 100:.2f}%")
    print(f"Best Val Acc     : {best_val_acc    * 100:.2f}%")
    print(f"Test Accuracy    : {test_acc        * 100:.2f}%")
    print("-" * 70)
    gap = (final_train_acc - test_acc) * 100
    print(f"Train–Test Gap   : {gap:.2f}%  {'⚠ possible overfit' if gap > 10 else '✓ healthy'}")

    # ── Per-class report ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-CLASS REPORT")
    print("=" * 70)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=gestures))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    # Pretty-print with gesture labels
    header = "         " + "  ".join(f"{g[:5]:>5}" for g in gestures)
    print(header)
    for i, row in enumerate(cm):
        label = f"{gestures[i][:8]:<8}"
        print(label + "  " + "  ".join(f"{v:>5}" for v in row))

    print(f"\n✓ Model saved → {MODEL_SAVE_PATH}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n✗ Error: {e}")
        traceback.print_exc()