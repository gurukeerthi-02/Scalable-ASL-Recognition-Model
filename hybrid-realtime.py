# main code

import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from tensorflow.keras.models import load_model
import time
import pyttsx3
import threading

# =======================
# LOAD MODELS
# =======================
static_model = load_model("./models/gesture_model.h5")
dynamic_model = load_model("./models/z_j_motion_lstm.h5")

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
STATIC_MAX_MOTION = 0.004      # Must be below this to qualify as static
DYNAMIC_MIN_MOTION = 0.010      # Increased to prevent false dynamic triggers
MOTION_SMOOTHING_WINDOW = 5    # Frames to average motion over

# Confidence thresholds
CONF_STATIC = 0.85             # Increased to reduce flickering on similar gestures
CONF_DYNAMIC = 0.75            # FIXED: Was 0.0, now reasonable threshold

# Timing parameters
STATIC_STABLE_FRAMES = 10      # Increased to ensure hand is truly static
DYNAMIC_HOLD_TIME = 2.0        # Seconds to display dynamic result
RESULT_DISPLAY_TIME = 1.5      # Seconds to display static result
COOLDOWN_TIME = 0.3            # Seconds between gestures

# =======================
# STATE VARIABLES
# =======================
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)
feature_history = deque(maxlen=10)   # Capture context before motion trigger
static_pred_buffer = deque(maxlen=10) # Increased buffer to smooth static predictions
motion_history = deque(maxlen=MOTION_SMOOTHING_WINDOW)
prev_tip = None

display_text = ""
display_conf = 0.0
result_start_time = 0
last_gesture_time = 0

current_sentence = ""
stable_frame_count = 0  # Counts frames hand has been stable
selected_voice_index = 0 # Toggle for voice selection

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

def speak_text(text, voice_idx=0):
    """Speak text in a separate thread to avoid blocking"""
    def _speak():
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if voices:
                # Cycle through available voices
                engine.setProperty('voice', voices[voice_idx % len(voices)].id)
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    if text.strip():
        threading.Thread(target=_speak, daemon=True).start()

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
        
        # Always track dynamic features for history (to catch start of motion)
        curr_dyn_features = extract_dynamic_features(hand)
        feature_history.append(curr_dyn_features)

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
                    static_pred_buffer.clear()
            else:
                stable_frame_count = 0

        # STATIC → DYNAMIC (motion detected)
        elif mode == MODE_STATIC:
            if motion > DYNAMIC_MIN_MOTION:
                mode = MODE_DYNAMIC_COLLECT
                motion_buffer.clear()
                motion_buffer.extend(feature_history) # Pre-fill with history
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
                        # Add to buffer for smoothing
                        static_pred_buffer.append(idx)
                        
                        # Only confirm if buffer is full and consistent (8/10 agreement)
                        if len(static_pred_buffer) == static_pred_buffer.maxlen:
                            most_common = max(set(static_pred_buffer), key=static_pred_buffer.count)
                            if most_common != -1 and static_pred_buffer.count(most_common) >= 8:
                                display_text = static_labels[most_common]
                                display_conf = pred[most_common]
                                mode = MODE_HOLD_RESULT
                                current_sentence += display_text
                                result_start_time = now
                                last_gesture_time = now
                                stable_frame_count = 0
                                static_pred_buffer.clear()
                    else:
                        static_pred_buffer.append(-1) # Penalize low confidence
            else:
                stable_frame_count = max(0, stable_frame_count - 2)
                static_pred_buffer.clear()

        # DYNAMIC COLLECT → Process sequence
        elif mode == MODE_DYNAMIC_COLLECT:
            motion_buffer.append(curr_dyn_features)

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
                    current_sentence += display_text
                    result_start_time = now
                    last_gesture_time = now
                else:
                    # allow another attempt instead of dropping immediately
                    motion_buffer.clear()
                
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

    # Create semi-transparent overlay
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1) # Bottom panel for sentence

    if display_text:
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        cv2.rectangle(overlay, (text_x - 10, 140), (text_x + text_size[0] + 10, 200), (0, 0, 0), -1)

    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
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

    # Display constructed sentence
    cv2.putText(frame, f"Sentence: {current_sentence}|", 
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display Voice Toggle
    voice_label = "Voice: MALE" if selected_voice_index % 2 == 0 else "Voice: FEMALE"
    cv2.putText(frame, f"[V] {voice_label}", (w - 250, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Display recognized gesture
    if display_text:
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        
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
    elif key == 32: # Space bar
        current_sentence += " "
    elif key == 8:  # Backspace
        current_sentence = current_sentence[:-1]
    elif key == ord('c'): # Clear
        current_sentence = ""
    elif key == 13: # Enter
        speak_text(current_sentence, selected_voice_index)
    elif key == ord('v'): # Voice toggle
        selected_voice_index += 1
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