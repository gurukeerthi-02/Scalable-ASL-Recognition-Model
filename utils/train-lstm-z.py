import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATASET_PATH = "../motion-dataset"
SEQUENCE_LENGTH = 30

X = []
y = []

# ---------------- LOAD LABELS ----------------
labels = sorted([
    f for f in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, f))
])

label_map = {label: idx for idx, label in enumerate(labels)}
print("Label map:", label_map)

# ---------------- SEQUENCE NORMALIZATION ----------------
def normalize_sequence(seq, target_len=SEQUENCE_LENGTH):
    if seq.shape[0] >= target_len:
        return seq[:target_len]
    last = seq[-1:]
    pad = np.repeat(last, target_len - seq.shape[0], axis=0)
    return np.vstack((seq, pad))

# ---------------- LOAD DATA ----------------
for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            sequence = np.load(os.path.join(folder, file))

            # Normalize length
            sequence = normalize_sequence(sequence)

            # Z-score normalization
            sequence = (sequence - np.mean(sequence, axis=0)) / (np.std(sequence, axis=0) + 1e-6)

            # Delta (motion) features
            delta = np.diff(sequence, axis=0)
            delta = np.vstack([np.zeros((1, sequence.shape[1])), delta])

            sequence = np.concatenate([sequence, delta], axis=1)  # (30, 126)

            X.append(sequence)
            y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)  # (samples, 30, 126)
print("y shape:", y.shape)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- CLASS WEIGHTS ----------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# ---------------- MODEL ----------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 126)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(len(labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

import time

print("\n=== STARTING TRAINING ===")

start_time = time.time()
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    verbose=1
)
end_time = time.time()
training_time = (end_time - start_time) / 60

# ---------------- METRICS ----------------
best_val_acc = max(history.history['val_accuracy'])
best_epoch = np.argmax(history.history['val_accuracy']) + 1

print("\n" + "="*40)
print("TRAINING PERFORMANCE SUMMARY (DYNAMIC)")
print("="*40)
print(f"Training Time           : {training_time:.2f} min")
print(f"Training Accuracy       : {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy     : {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Best Val Accuracy       : {best_val_acc * 100:.2f}% (at epoch {best_epoch})")
print(f"Training Loss           : {history.history['loss'][-1]:.4f}")
print("="*40)

# ---------------- SAVE ----------------
os.makedirs("../results", exist_ok=True)
np.save("../results/training_history_dynamic.npy", history.history)
model.save("z_j_motion_lstm.h5")

print("Model saved successfully")
print(f"Labels used: {labels}")
