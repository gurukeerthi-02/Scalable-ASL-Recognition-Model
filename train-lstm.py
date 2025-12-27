import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- CONFIG ----------------
DATASET_DIR = "motion-dataset"
SEQUENCE_LENGTH = 30
FEATURES = 63
CLASSES = ["no-gesture", "Z", "J"]

LABEL_MAP = {
    "no-gesture": 0,
    "Z": 1,
    "J": 2
}

# ---------------- LOAD DATA ----------------
X = []
y = []

for class_name in CLASSES:
    class_dir = os.path.join(DATASET_DIR, class_name)
    label = LABEL_MAP[class_name]

    for file in os.listdir(class_dir):
        if file.endswith(".npy"):
            data = np.load(os.path.join(class_dir, file))
            if data.shape == (SEQUENCE_LENGTH, FEATURES):
                X.append(data)
                y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ---------------- ONE-HOT ENCODE LABELS ----------------
y = to_categorical(y, num_classes=3)

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ---------------- BUILD LSTM MODEL ----------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES)),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop]
)

# ---------------- EVALUATE ----------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# ---------------- SAVE MODEL ----------------
model.save("z_j_motion_lstm.h5")
print("Model saved as z_j_motion_lstm.h5")
