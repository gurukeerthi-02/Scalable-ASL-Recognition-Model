import cv2 as cv
import mediapipe as mp
import numpy as np
from collections import deque


capture = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
)

mp_draw = mp.solutions.drawing_utils



SEQUENCE_LENGTH = 20 # collect 20 frames for rolling buffer creation
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)

def extract_features(hand_landmarks):
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x,lm.y,lm.z])
    return np.array(features)


while True:

    import os

    LABEL = "Z"
    SAVE_DIR = f"motion-dataset/{LABEL}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    sequence_count = 0


    ret, frame = capture.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        features = extract_features(hand_landmarks)

        motion_buffer.append(features)

        cv.putText(
            frame,
            f"Frames captured: {len(motion_buffer)}",
            (30, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv.imshow("Hand Landmark Stream", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


        if len(motion_buffer) == SEQUENCE_LENGTH:
            sequence = np.array(motion_buffer)
            np.save(f"{SAVE_DIR}/{sequence_count}.npy", sequence)
            print(f"Saved sequence {sequence_count}")
            sequence_count += 1
            motion_buffer.clear()

