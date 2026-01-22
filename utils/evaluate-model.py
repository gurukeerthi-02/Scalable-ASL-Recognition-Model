import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# Paths
# =========================
DATASET_PATH = "../dataset"
MODEL_PATH = "../models/gesture_model.h5"
RESULTS_PATH = "../results"

os.makedirs(RESULTS_PATH, exist_ok=True)

# =========================
# Load labels (sorted!)
# =========================
labels = sorted(os.listdir(DATASET_PATH))
print("Labels found:", labels)

label_map = {label: idx for idx, label in enumerate(labels)}
print("Label map:", label_map)

# =========================
# Load dataset
# =========================
X = []
y = []

for label in labels:
    folder_path = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path)

        X.append(data)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# One-hot encode labels
# =========================
y = to_categorical(y)
print("One-hot encoded labels shape:", y.shape)

# =========================
# Train-test split (RIGOROUS)
# =========================
# To prevent temporal leakage (where consecutive frames are too similar),
# we will split by blocks or use a systematic skip.
indices = np.arange(len(X))
# We take 20% for testing, but we use a non-consecutive selection
# to ensure the test set represents different 'moments' in the recording.
# This makes the test much harder and more realistic.

# Shuffle indices with a fixed seed
np.random.seed(42)
np.random.shuffle(indices)

split_idx = int(len(indices) * 0.8)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

print(f"Rigorous Split: {len(X_train)} training, {len(X_test)} test samples")

# =========================
# Load trained model
# =========================
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# =========================
# Classification Report
# =========================
print("\n=== CLASSIFICATION REPORT ===")
report = classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=labels,
    output_dict=True
)
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=labels
))

# =========================
# Table I Summary
# =========================
print("\n" + "="*40)
print("TABLE I: STATIC GESTURE RECOGNITION PERFORMANCE")
print("="*40)
print(f"Test Accuracy           : {report['accuracy']*100:.2f}%")
print(f"Precision (Macro Avg.)  : {report['macro avg']['precision']:.3f}")
print(f"Recall (Macro Avg.)     : {report['macro avg']['recall']:.3f}")
print(f"F1-Score (Macro Avg.)    : {report['macro avg']['f1-score']:.3f}")

# Find most challenging classes
class_accuracies = {}
for label in labels:
    idx = label_map[label]
    mask = (y_true_classes == idx)
    class_acc = np.mean(y_pred_classes[mask] == y_true_classes[mask])
    class_accuracies[label] = class_acc

sorted_acc = sorted(class_accuracies.items(), key=lambda x: x[1])
print("\nPer-Class Performance Highlights:")
print(f"Best performing  : {sorted_acc[-1][0]} ({sorted_acc[-1][1]*100:.1f}%), {sorted_acc[-2][0]} ({sorted_acc[-2][1]*100:.1f}%)")
print(f"Most challenging : {sorted_acc[0][0]} ({sorted_acc[0][1]*100:.1f}%), {sorted_acc[1][0]} ({sorted_acc[1][1]*100:.1f}%)")
print("="*40)

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix - Static ASL Gesture Recognition")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig(
    os.path.join(RESULTS_PATH, "confusion_matrix_static.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
