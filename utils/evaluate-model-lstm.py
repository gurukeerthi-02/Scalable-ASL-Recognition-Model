import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# =========================
# Constants
# =========================
DATASET_PATH = "../motion-dataset"
MODEL_PATH = "z_j_motion_lstm.h5"
SEQUENCE_LENGTH = 30
RESULTS_PATH = "../results"

os.makedirs(RESULTS_PATH, exist_ok=True)

# =========================
# Load labels (same as training)
# =========================
labels = sorted([
    f for f in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, f))
])

label_map = {label: idx for idx, label in enumerate(labels)}
print("Label map:", label_map)

# =========================
# Sequence normalization
# =========================
def normalize_sequence(seq, target_len=SEQUENCE_LENGTH):
    if seq.shape[0] >= target_len:
        return seq[:target_len]
    last = seq[-1:]
    pad = np.repeat(last, target_len - seq.shape[0], axis=0)
    return np.vstack((seq, pad))

# =========================
# Load dataset (MATCH TRAINING)
# =========================
X = []
y = []

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            sequence = np.load(os.path.join(folder, file))

            # Length normalization
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

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Load model
# =========================
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# =========================
# Predictions
# =========================
print("\nRunning evaluation...")
y_pred = model.predict(X_test, batch_size=32, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# =========================
# Debug info
# =========================
print("Unique y_test labels:", np.unique(y_test))
print("Unique y_pred labels:", np.unique(y_pred_classes))

# =========================
# Accuracy
# =========================
accuracy = np.mean(y_pred_classes == y_test) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

# =========================
# Classification Report
# =========================
unique_labels = np.unique(y_test)
target_names = [labels[i] for i in unique_labels]

print("\n=== CLASSIFICATION REPORT ===")
report = classification_report(
    y_test,
    y_pred_classes,
    labels=unique_labels,
    target_names=target_names,
    output_dict=True,
    zero_division=0
)
print(classification_report(
    y_test,
    y_pred_classes,
    labels=unique_labels,
    target_names=target_names,
    zero_division=0
))

# =========================
# Table II Summary
# =========================
print("\n" + "="*40)
print("TABLE II: DYNAMIC GESTURE RECOGNITION PERFORMANCE")
print("="*40)
print(f"Test Accuracy           : {report['accuracy']*100:.2f}%")
print(f"Precision (Macro Avg.)  : {report['macro avg']['precision']:.3f}")
print(f"Recall (Macro Avg.)     : {report['macro avg']['recall']:.3f}")
print(f"F1-Score (Macro Avg.)    : {report['macro avg']['f1-score']:.3f}")

# Find most challenging classes
class_accuracies = {}
for i, label_name in enumerate(target_names):
    idx = unique_labels[i]
    mask = (y_test == idx)
    class_acc = np.mean(y_pred_classes[mask] == y_test[mask])
    class_accuracies[label_name] = class_acc

sorted_acc = sorted(class_accuracies.items(), key=lambda x: x[1])
print("\nPer-Class Performance Highlights:")
if len(sorted_acc) >= 2:
    print(f"Best performing  : {sorted_acc[-1][0]} ({sorted_acc[-1][1]*100:.1f}%), {sorted_acc[-2][0]} ({sorted_acc[-2][1]*100:.1f}%)")
    print(f"Most challenging : {sorted_acc[0][0]} ({sorted_acc[0][1]*100:.1f}%), {sorted_acc[1][0]} ({sorted_acc[1][1]*100:.1f}%)")
else:
    print(f"Performance: {sorted_acc[0][0]} ({sorted_acc[0][1]*100:.1f}%)")
print("="*40)

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(
    y_test,
    y_pred_classes,
    labels=unique_labels
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[labels[i] for i in unique_labels],
    yticklabels=[labels[i] for i in unique_labels]
)

plt.title("Confusion Matrix - Dynamic ASL (LSTM)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig(
    os.path.join(RESULTS_PATH, "confusion_matrix_dynamic.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
