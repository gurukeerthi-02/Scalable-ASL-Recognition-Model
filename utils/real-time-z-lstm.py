import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

model = load_model("z_motion_lstm.h5")
labels = ["Not Z","Z"]  # currently only Z
print("LSTM model loaded")

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

SEQUENCE_LENGTH = 20
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)


def extract_features(hand_landmarks):
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        # Display buffer status
        cv2.putText(
            frame,
            f"Frames: {len(motion_buffer)}/{SEQUENCE_LENGTH}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Predict only when buffer is full
        if len(motion_buffer) == SEQUENCE_LENGTH:
            sequence = np.array(motion_buffer)
            sequence = sequence.reshape(1, SEQUENCE_LENGTH, 63)

            prediction = model.predict(sequence, verbose=0)
            print("Prediction shape:", prediction.shape)
            print("Prediction values:", prediction)
            confidence = np.max(prediction)
            predicted_label = labels[np.argmax(prediction)]

            if confidence > 0.8:
                cv2.putText(
                    frame,
                    f"{predicted_label} ({confidence:.2f})",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

    cv2.imshow("Z Gesture Recognition (LSTM)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





cap.release()
cv2.destroyAllWindows()
