import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os
import time

# ---------------- CONFIG ----------------
SEQUENCE_LENGTH = 30
DATASET_DIR = "../motion-dataset"

CLASSES = {
    1: "Z",
    2: "J",
    3: "Hello"
}

os.makedirs(DATASET_DIR, exist_ok=True)
for c in CLASSES.values():
    os.makedirs(os.path.join(DATASET_DIR, c), exist_ok=True)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)

current_label = None
sample_count = {
    c: len(os.listdir(os.path.join(DATASET_DIR, c)))
    for c in CLASSES.values()
}

print("Controls:")
print("1 → Z")
print("2 → J")
print("3 → Hello")
print("s → Save sequence")
print("q → Quit")

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(hand_landmarks):
    base = hand_landmarks.landmark[0]  # wrist
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([
            lm.x - base.x,
            lm.y - base.y,
            lm.z - base.z
        ])
    return np.array(features)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        features = extract_features(hand_landmarks)
        motion_buffer.append(features)

    label_text = "None" if current_label is None else CLASSES[current_label]
    cv2.putText(frame, f"Recording: {label_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Frames: {len(motion_buffer)}/{SEQUENCE_LENGTH}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in [ord('1'), ord('2'), ord('3')]:
        current_label = int(chr(key))
        motion_buffer.clear()
        print(f"Switched to class: {CLASSES[current_label]}")

    elif key == ord('s') and current_label is not None:
        if len(motion_buffer) == SEQUENCE_LENGTH:
            class_name = CLASSES[current_label]
            filename = f"{class_name}_{sample_count[class_name]}.npy"
            path = os.path.join(DATASET_DIR, class_name, filename)

            np.save(path, np.array(motion_buffer))
            sample_count[class_name] += 1
            motion_buffer.clear()

            print(f"Saved: {path}")
            time.sleep(0.3)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
