import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# =========================
# CONFIGURATION
# =========================
LABEL = "Z"
SEQUENCE_LENGTH = 20
SAVE_DIR = f"motion-dataset/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

START_THRESHOLD = 0.005   # movement to start recording
STOP_THRESHOLD = 0.002    # movement to stop recording

# =========================
# INITIALISE
# =========================
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

motion_buffer = deque(maxlen=SEQUENCE_LENGTH)
prev_tip = None
motion_started = False
sequence_count = len(os.listdir(SAVE_DIR))

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(hand_landmarks):
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    # =========================
    # DRAW Z GUIDE
    # =========================
    cv2.rectangle(frame, (100, 100), (w - 100, h - 100), (255, 255, 255), 2)
    cv2.line(frame, (120, 140), (w - 120, 140), (0, 255, 255), 2)
    cv2.line(frame, (w - 120, 140), (120, h - 140), (0, 255, 255), 2)
    cv2.line(frame, (120, h - 140), (w - 120, h - 140), (0, 255, 255), 2)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Track index fingertip (landmark 8)
        tip = hand_landmarks.landmark[8]
        tip_pos = np.array([tip.x, tip.y])

        if prev_tip is not None:
            dist = np.linalg.norm(tip_pos - prev_tip)

            # Detect motion start
            if dist > START_THRESHOLD:
                motion_started = True

            # Record motion
            if motion_started:
                features = extract_features(hand_landmarks)
                motion_buffer.append(features)

            # Detect motion stop
            if motion_started and dist < STOP_THRESHOLD and len(motion_buffer) == SEQUENCE_LENGTH:
                sequence = np.array(motion_buffer)
                np.save(f"{SAVE_DIR}/{sequence_count}.npy", sequence)
                print(f"Saved Z gesture #{sequence_count}")

                sequence_count += 1
                motion_buffer.clear()
                motion_started = False

        prev_tip = tip_pos

    # =========================
    # DISPLAY INFO
    # =========================
    cv2.putText(
        frame,
        f"Z Samples: {sequence_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Frames: {len(motion_buffer)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collect Z Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
