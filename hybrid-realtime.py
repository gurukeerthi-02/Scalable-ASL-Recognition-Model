# main code

import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from tensorflow.keras.models import load_model
import time

# =======================
# LOAD MODELS
# =======================
static_model = load_model("gesture_model.h5")
dynamic_model = load_model("z_j_motion_lstm.h5")

static_labels = sorted(os.listdir("dataset"))
dynamic_labels = ["No Gesture", "Z", "J"]

# =======================
# MEDIAPIPE
# =======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =======================
# CAMERA
# =======================
cap = cv2.VideoCapture(0)

# =======================
# MODES (State Machine)
# =======================
MODE_IDLE = 0           # No hand detected or waiting
MODE_STATIC = 1         # Detecting static gestures
MODE_DYNAMIC_COLLECT = 2  # Collecting motion sequence
MODE_HOLD_RESULT = 3    # Displaying result
mode = MODE_IDLE

# =======================
# TUNED PARAMETERS
# =======================
SEQUENCE_LENGTH = 30

# Motion thresholds with hysteresis
STATIC_MAX_MOTION = 0.005      # Must be below this to qualify as static
DYNAMIC_MIN_MOTION = 0.012     # Must exceed this to enter dynamic mode
MOTION_SMOOTHING_WINDOW = 5    # Frames to average motion over

# Confidence thresholds
CONF_STATIC = 0.80             # Lowered slightly for better detection
CONF_DYNAMIC = 0.75            # FIXED: Was 0.0, now reasonable threshold

# Timing parameters
STATIC_STABLE_FRAMES = 8       # Frames hand must be stable before static detection
DYNAMIC_HOLD_TIME = 2.0        # Seconds to display dynamic result
RESULT_DISPLAY_TIME = 1.5      # Seconds to display static result
COOLDOWN_TIME = 0.3            # Seconds between gestures

# =======================
# STATE VARIABLES
# =======================
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)
motion_history = deque(maxlen=MOTION_SMOOTHING_WINDOW)
prev_tip = None

display_text = ""
display_conf = 0.0
result_start_time = 0
last_gesture_time = 0

stable_frame_count = 0  # Counts frames hand has been stable

# =======================
# FEATURE EXTRACTION
# =======================
def extract_static_features(hand):
    """Extract 68-dimensional feature vector for static gestures"""
    def dist(a, b):
        return np.linalg.norm(
            np.array([a.x, a.y]) - np.array([b.x, b.y])
        )

    lm = hand.landmark
    wrist = lm[0]
    features = []

    # Relative 3D positions (21 landmarks × 3 = 63 features)
    for p in lm:
        features.extend([
            p.x - wrist.x,
            p.y - wrist.y,
            p.z - wrist.z
        ])

    # Finger tip distances from wrist (4 features)
    features.extend([
        dist(lm[8], wrist),   # Index
        dist(lm[12], wrist),  # Middle
        dist(lm[16], wrist),  # Ring
        dist(lm[20], wrist)   # Pinky
    ])

    # Thumb-to-index distance (1 feature)
    features.append(dist(lm[4], lm[8]))

    return np.array(features).reshape(1, -1)


def extract_dynamic_features(hand):
    """Extract 63-dimensional feature vector for dynamic gestures"""
    features = []
    for lm in hand.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features)


def get_smoothed_motion():
    """Calculate smoothed motion value to reduce jitter"""
    if len(motion_history) == 0:
        return 0.0
    return np.mean(motion_history)

# =======================
# MAIN LOOP
# =======================
print("Starting Hybrid ASL Recognition System...")
print("Press 'q' to quit, 'r' to reset state")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    now = time.time()
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Check cooldown period (must be before hand detection block)
    in_cooldown = (now - last_gesture_time) < COOLDOWN_TIME

    # =======================
    # HAND DETECTION
    # =======================
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Calculate instantaneous motion
        tip = hand.landmark[8]  # Index finger tip
        tip_pos = np.array([tip.x, tip.y])

        instant_motion = 0.0
        if prev_tip is not None:
            instant_motion = np.linalg.norm(tip_pos - prev_tip)
        
        prev_tip = tip_pos
        motion_history.append(instant_motion)
        motion = get_smoothed_motion()

        # =======================
        # STATE MACHINE LOGIC
        # =======================
        
        # IDLE → STATIC (hand detected and stable)
        if mode == MODE_IDLE and not in_cooldown:
            if motion < STATIC_MAX_MOTION:
                stable_frame_count += 1
                if stable_frame_count >= STATIC_STABLE_FRAMES:
                    mode = MODE_STATIC
                    stable_frame_count = 0
            else:
                stable_frame_count = 0

        # STATIC → DYNAMIC (motion detected)
        elif mode == MODE_STATIC:
            if motion > DYNAMIC_MIN_MOTION:
                mode = MODE_DYNAMIC_COLLECT
                motion_buffer.clear()
                display_text = ""  # Clear old static result
                stable_frame_count = 0
                
            # Detect static gesture
            elif motion < STATIC_MAX_MOTION:
                stable_frame_count += 1
                
                # Only predict after hand is stable
                if stable_frame_count >= STATIC_STABLE_FRAMES:
                    static_feat = extract_static_features(hand)
                    pred = static_model.predict(static_feat, verbose=0)[0]
                    idx = np.argmax(pred)
                    conf = pred[idx]

                    if conf > CONF_STATIC:
                        display_text = static_labels[idx]
                        display_conf = conf
                        mode = MODE_HOLD_RESULT
                        result_start_time = now
                        last_gesture_time = now
                        stable_frame_count = 0
            else:
                stable_frame_count = max(0, stable_frame_count - 2)

        # DYNAMIC COLLECT → Process sequence
        elif mode == MODE_DYNAMIC_COLLECT:
            motion_buffer.append(extract_dynamic_features(hand))

            # When buffer is full, make prediction
            if len(motion_buffer) == SEQUENCE_LENGTH:
                seq = np.array(motion_buffer).reshape(1, SEQUENCE_LENGTH, 63)
                pred = dynamic_model.predict(seq, verbose=0)[0]
                idx = np.argmax(pred)
                conf = pred[idx]

                # Only accept confident non-"No Gesture" predictions
                if conf > CONF_DYNAMIC and dynamic_labels[idx] != "No Gesture":
                    display_text = dynamic_labels[idx]
                    display_conf = conf
                    mode = MODE_HOLD_RESULT
                    result_start_time = now
                    last_gesture_time = now
                else:
                    # Prediction not confident enough, return to idle
                    mode = MODE_IDLE
                    display_text = ""
                
                motion_buffer.clear()
                stable_frame_count = 0

        # HOLD RESULT → Return to IDLE
        elif mode == MODE_HOLD_RESULT:
            hold_duration = DYNAMIC_HOLD_TIME if "Z" in display_text or "J" in display_text else RESULT_DISPLAY_TIME
            
            if now - result_start_time >= hold_duration:
                mode = MODE_IDLE
                display_text = ""
                display_conf = 0.0
                stable_frame_count = 0

    else:
        # No hand detected
        prev_tip = None
        motion_history.clear()
        if mode != MODE_HOLD_RESULT:
            mode = MODE_IDLE
            stable_frame_count = 0

    # =======================
    # UI OVERLAY
    # =======================
    mode_text = {
        MODE_IDLE: "IDLE",
        MODE_STATIC: "STATIC DETECTION",
        MODE_DYNAMIC_COLLECT: f"DYNAMIC ({len(motion_buffer)}/{SEQUENCE_LENGTH})",
        MODE_HOLD_RESULT: "SHOWING RESULT"
    }

    # Status bar background
    cv2.rectangle(frame, (0, 0), (640, 120), (0, 0, 0), -1)
    
    cv2.putText(frame, f"Mode: {mode_text[mode]}", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    motion_val = get_smoothed_motion()
    cv2.putText(frame, f"Motion: {motion_val:.5f}", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(frame, f"Stable: {stable_frame_count}/{STATIC_STABLE_FRAMES}", 
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Cooldown indicator
    if in_cooldown:
        cv2.putText(frame, "COOLDOWN", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Display recognized gesture
    if display_text:
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        
        # Background for better visibility
        cv2.rectangle(frame, 
                      (text_x - 10, 140), 
                      (text_x + text_size[0] + 10, 200), 
                      (0, 0, 0), -1)
        
        cv2.putText(frame,
                    f"{display_text}",
                    (text_x, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3)
        
        cv2.putText(frame,
                    f"{display_conf*100:.1f}%",
                    (text_x, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    cv2.imshow("Hybrid ASL Recognition - OPTIMIZED", frame)

    # =======================
    # KEYBOARD CONTROLS
    # =======================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset system state
        mode = MODE_IDLE
        display_text = ""
        display_conf = 0.0
        motion_buffer.clear()
        motion_history.clear()
        stable_frame_count = 0
        print("System reset")

cap.release()
cv2.destroyAllWindows()
print("System terminated")