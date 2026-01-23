import cv2 as cv
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
import time

model = load_model("../models/gesture_model.h5")
print("Model loaded successfully !")

dataset_path = "../dataset"
labels = sorted(os.listdir(dataset_path))
print("Labels: ",labels)

print("Starting generalization testing...")
predictions = []


capture = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)

mp_draw = mp.solutions.drawing_utils


while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv.flip(frame,1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        def dist(a, b):
            return np.linalg.norm(
                np.array([a.x, a.y]) - np.array([b.x, b.y])
            )

        lm = hand_landmarks.landmark
        wrist = lm[0]
        features = []

        # ---- NORMALISED LANDMARKS ----
        for point in lm:
            features.extend([
                point.x - wrist.x,
                point.y - wrist.y,
                point.z - wrist.z
            ])

        # ---- FINGER CURLS ----
        features.extend([
            dist(lm[8],  wrist),
            dist(lm[12], wrist),
            dist(lm[16], wrist),
            dist(lm[20], wrist)
        ])

        # ---- THUMB–INDEX ----
        features.append(dist(lm[4], lm[8]))
        
        features = np.array(features).reshape(1, -1)

        for i in range(10):
            prediciton = model.predict(features, verbose=0)
            predicted_index = np.argmax(prediciton)
            predicted_label = labels[predicted_index]
            predictions.append(predicted_label)
            time.sleep(1)

        accuracy = predictions.count('A') / 10
        print(f"Accuracy for 'A': {accuracy * 100}%")


        confidence = prediciton[0][predicted_index]

        cv.putText(
            frame,
            f"{predicted_label} ({confidence:.2f})",
            (30,50),
            cv.FONT_HERSHEY_COMPLEX,
            1,
            (0,255,0),
            2
        )

    cv.imshow("Sign Language Recognition ", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
