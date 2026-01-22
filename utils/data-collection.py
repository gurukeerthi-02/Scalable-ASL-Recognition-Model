import cv2
import mediapipe as mp
import numpy as np
import os

LABEL = "Y"     # change this when collecting other gestures
SAMPLES = 1000

SAVE_DIR = f"../dataset/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        features = []

        def dist(a, b):
            return np.linalg.norm(
                np.array([a.x, a.y]) - np.array([b.x, b.y])
            )

        lm = hand.landmark
        wrist = lm[0]
        features = []

        # ---- NORMALISED LANDMARKS (63) ----
        for point in lm:
            features.extend([
                point.x - wrist.x,
                point.y - wrist.y,
                point.z - wrist.z
            ])

        # ---- FINGER CURL DISTANCES (4) ----
        index_curl  = dist(lm[8],  wrist)
        middle_curl = dist(lm[12], wrist)
        ring_curl   = dist(lm[16], wrist)
        pinky_curl  = dist(lm[20], wrist)

        features.extend([
            index_curl,
            middle_curl,
            ring_curl,
            pinky_curl
        ])

        # ---- THUMB–INDEX DISTANCE (1) ----
        thumb_index_dist = dist(lm[4], lm[8])
        features.append(thumb_index_dist)

        np.save(f"{SAVE_DIR}/{count}.npy", np.array(features))
        count += 1
        print(f"Saved {count}/{SAMPLES}")

    cv2.imshow("Collecting Data", frame)

    if count >= SAMPLES:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
