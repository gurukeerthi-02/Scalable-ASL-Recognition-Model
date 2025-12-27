import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time

# ---------------- LOAD MODEL ----------------
model = load_model("z_j_motion_lstm.h5")
labels = ["no-gesture", "Z", "J"]
print("LSTM model loaded")

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

SEQUENCE_LENGTH = 30
FEATURES = 63
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)

# ---------------- CONTROL PARAMS ----------------
CONFIDENCE_THRESHOLD = 0.80
COOLDOWN_TIME = 2.0  # seconds
last_prediction_time = 0
current_display = ""

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(hand_landmarks):
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
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

        cv2.putText(
            frame,
            f"Frames: {len(motion_buffer)}/{SEQUENCE_LENGTH}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # ---------- PREDICTION ----------
        if len(motion_buffer) == SEQUENCE_LENGTH:
            now = time.time()

            if now - last_prediction_time > COOLDOWN_TIME:
                sequence = np.array(motion_buffer).reshape(1, SEQUENCE_LENGTH, FEATURES)
                prediction = model.predict(sequence, verbose=0)[0]

                class_id = np.argmax(prediction)
                confidence = prediction[class_id]
                label = labels[class_id]

                if confidence > CONFIDENCE_THRESHOLD and label != "no-gesture":
                    current_display = f"{label} ({confidence:.2f})"
                    last_prediction_time = now
                    motion_buffer.clear()
                else:
                    current_display = ""

    # ---------------- DISPLAY ----------------
    if current_display:
        cv2.putText(
            frame,
            current_display,
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3
        )

    cv2.imshow("Z / J Gesture Recognition (LSTM)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
