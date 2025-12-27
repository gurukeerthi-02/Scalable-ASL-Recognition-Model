import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


DATASET_PATH = "motion-dataset"
SEQUENCE_LENGTH = 20

X = []
y = []

labels = os.listdir(DATASET_PATH)
label_map = {label: idx for idx, label in enumerate(labels)}

print("Label map:", label_map)

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        sequence = np.load(os.path.join(folder, file))
        X.append(sequence)
        y.append(label_map[label])


X = np.array(X)   # (N, 20, 63)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(20, 63)),
    Dropout(0.3),

    LSTM(32),
    Dropout(0.3),

    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test)
)

model.save("z_motion_lstm.h5")
print("Z motion LSTM model saved")
