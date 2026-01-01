# not main code
# still not ready to use in prod

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
# UI CONFIGURATION
# =======================
# Detection box (center of frame)
BOX_WIDTH = 400
BOX_HEIGHT = 400

def draw_detection_box(frame):
    """Draw dotted detection box in center of frame"""
    h, w = frame.shape[:2]
    
    # Calculate box corners (centered)
    box_x1 = (w - BOX_WIDTH) // 2
    box_y1 = (h - BOX_HEIGHT) // 2
    box_x2 = box_x1 + BOX_WIDTH
    box_y2 = box_y1 + BOX_HEIGHT
    
    # Dotted line parameters
    color = (0, 255, 255)  # Cyan
    thickness = 2
    dot_length = 15
    gap_length = 10
    
    # Draw top and bottom lines
    for x in range(box_x1, box_x2, dot_length + gap_length):
        cv2.line(frame, (x, box_y1), (min(x + dot_length, box_x2), box_y1), color, thickness)
        cv2.line(frame, (x, box_y2), (min(x + dot_length, box_x2), box_y2), color, thickness)
    
    # Draw left and right lines
    for y in range(box_y1, box_y2, dot_length + gap_length):
        cv2.line(frame, (box_x1, y), (box_x1, min(y + dot_length, box_y2)), color, thickness)
        cv2.line(frame, (box_x2, y), (box_x2, min(y + dot_length, box_y2)), color, thickness)
    
    # Corner markers for emphasis
    corner_size = 30
    cv2.line(frame, (box_x1, box_y1), (box_x1 + corner_size, box_y1), color, thickness + 1)
    cv2.line(frame, (box_x1, box_y1), (box_x1, box_y1 + corner_size), color, thickness + 1)
    
    cv2.line(frame, (box_x2, box_y1), (box_x2 - corner_size, box_y1), color, thickness + 1)
    cv2.line(frame, (box_x2, box_y1), (box_x2, box_y1 + corner_size), color, thickness + 1)
    
    cv2.line(frame, (box_x1, box_y2), (box_x1 + corner_size, box_y2), color, thickness + 1)
    cv2.line(frame, (box_x1, box_y2), (box_x1, box_y2 - corner_size), color, thickness + 1)
    
    cv2.line(frame, (box_x2, box_y2), (box_x2 - corner_size, box_y2), color, thickness + 1)
    cv2.line(frame, (box_x2, box_y2), (box_x2, box_y2 - corner_size), color, thickness + 1)
    
    return box_x1, box_y1, box_x2, box_y2


def is_hand_in_box(hand, box_coords, frame_width, frame_height):
    """Check if hand is within detection box"""
    box_x1, box_y1, box_x2, box_y2 = box_coords
    
    # Get hand bounding box
    x_coords = [lm.x * frame_width for lm in hand.landmark]
    y_coords = [lm.y * frame_height for lm in hand.landmark]
    
    hand_x1, hand_x2 = min(x_coords), max(x_coords)
    hand_y1, hand_y2 = min(y_coords), max(y_coords)
    
    # Check if majority of hand is in box (80% overlap)
    hand_center_x = (hand_x1 + hand_x2) / 2
    hand_center_y = (hand_y1 + hand_y2) / 2
    
    return (box_x1 < hand_center_x < box_x2 and box_y1 < hand_center_y < box_y2)


def draw_overlay_panel(frame, mode, motion, stable_count, buffer_len, in_cooldown):
    """Draw semi-transparent overlay panels for metrics"""
    h, w = frame.shape[:2]
    
    # Create overlay for metrics (top-left)
    overlay = frame.copy()
    
    # Top panel background (semi-transparent)
    cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    mode_text = {
        MODE_IDLE: "IDLE - Place hand in box",
        MODE_STATIC: "DETECTING STATIC",
        MODE_DYNAMIC_COLLECT: f"RECORDING MOTION",
        MODE_HOLD_RESULT: "RESULT"
    }
    
    # Mode indicator with color coding
    mode_color = {
        MODE_IDLE: (100, 100, 100),
        MODE_STATIC: (255, 165, 0),
        MODE_DYNAMIC_COLLECT: (255, 0, 255),
        MODE_HOLD_RESULT: (0, 255, 0)
    }
    
    cv2.putText(frame, mode_text[mode], 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color[mode], 2)
    
    # Metrics
    cv2.putText(frame, f"Motion: {motion:.5f}", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    if mode == MODE_DYNAMIC_COLLECT:
        # Progress bar for motion buffer
        progress = buffer_len / SEQUENCE_LENGTH
        bar_width = 200
        cv2.rectangle(frame, (20, 70), (20 + bar_width, 85), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 70), (20 + int(bar_width * progress), 85), (255, 0, 255), -1)
        cv2.putText(frame, f"{buffer_len}/{SEQUENCE_LENGTH} frames", 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    elif mode == MODE_STATIC:
        # Stability indicator
        progress = stable_count / STATIC_STABLE_FRAMES
        bar_width = 200
        cv2.rectangle(frame, (20, 70), (20 + bar_width, 85), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 70), (20 + int(bar_width * progress), 85), (255, 165, 0), -1)
        cv2.putText(frame, f"Stability: {stable_count}/{STATIC_STABLE_FRAMES}", 
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Cooldown indicator
    if in_cooldown:
        cv2.putText(frame, "COOLDOWN", 
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    
    # Instructions panel (bottom-left)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - 80), (350, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, "Q: Quit  |  R: Reset", 
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Keep hand in box for detection", 
                (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def draw_gesture_result(frame, text, confidence):
    """Draw large centered gesture result"""
    h, w = frame.shape[:2]
    
    # Create semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h//2 - 80), (w, h//2 + 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Main gesture text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2.5, 4)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h // 2
    
    cv2.putText(frame, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 4)
    
    # Confidence percentage
    conf_text = f"{confidence * 100:.1f}% confidence"
    conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    conf_x = (w - conf_size[0]) // 2
    
    cv2.putText(frame, conf_text, (conf_x, text_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

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
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw detection box first (so it's behind hand landmarks)
    box_coords = draw_detection_box(frame)

    # Check cooldown period (must be before hand detection block)
    in_cooldown = (now - last_gesture_time) < COOLDOWN_TIME

    # Initialize hand_in_box flag
    hand_in_box = False

    # =======================
    # HAND DETECTION
    # =======================
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        
        # Check if hand is in detection box
        hand_in_box = is_hand_in_box(hand, box_coords, w, h)
        
        # Draw hand landmarks with different colors based on position
        landmark_color = (0, 255, 0) if hand_in_box else (128, 128, 128)
        connection_color = (0, 200, 0) if hand_in_box else (100, 100, 100)
        
        # Custom drawing with color coding
        mp_draw.draw_landmarks(
            frame, 
            hand, 
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=landmark_color, thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=connection_color, thickness=2)
        )
        

        # Only process if hand is in box
        if hand_in_box:
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
            # Hand detected but not in box - show warning
            if mode not in [MODE_HOLD_RESULT]:
                cv2.putText(frame, "Move hand into box!", 
                           (w//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 0, 255), 2)

    else:
        # No hand detected
        prev_tip = None
        motion_history.clear()
        if mode != MODE_HOLD_RESULT:
            mode = MODE_IDLE
            stable_frame_count = 0

    # =======================
    # UI RENDERING
    # =======================
    
    # Draw overlay panels with metrics
    draw_overlay_panel(frame, mode, get_smoothed_motion(), 
                      stable_frame_count, len(motion_buffer), in_cooldown)
    
    # Draw gesture result if available
    if display_text:
        draw_gesture_result(frame, display_text, display_conf)

    cv2.imshow("ASL Recognition System", frame)

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