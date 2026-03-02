import cv2
import time
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# ==========================================
# CONFIGURATION
# ==========================================
# Adjust paths as needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "utils/models/static_model_person_split_v7.h5")
DATASET_PATH = os.path.join(BASE_DIR, "utils/dataset_merged")

CAP_WIDTH = 640
CAP_HEIGHT = 480
USE_WEBCAM = True

# Target Gestures to Test
TARGET_GESTURES = ['A', 'B', 'C', 'L', 'V']

# Duration
TEST_DURATION = 5.0
PREPARE_DURATION = 3.0

# ==========================================
# SETUP
# ==========================================
def setup_local_inference():
    print("="*60)
    print("INITIALIZING LOCAL MODEL")
    print("="*60)
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        print("Please check the path.")
        sys.exit(1)
        
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # 2. Load Labels
    labels = []
    if os.path.exists(DATASET_PATH):
        labels = sorted(os.listdir(DATASET_PATH))
    else:
        # Fallback if dataset folder missing
        print("[WARN] Dataset folder not found, using default labels.")
        labels = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])
    
    print(f"Loaded {len(labels)} labels.")

    # 3. MediaPipe
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    return model, labels, hands_detector, mp.solutions.drawing_utils

def extract_features(hand_landmarks):
    """
    Exact replica of server.py feature extraction (68-dim).
    """
    def dist(a, b):
        return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))

    lm = hand_landmarks.landmark
    wrist = lm[0]
    features = []

    # 1. Relative 3D positions (21 * 3 = 63 dims)
    for p in lm:
        features.extend([
            p.x - wrist.x,
            p.y - wrist.y,
            p.z - wrist.z
        ])

    # 2. Finger tip distances from wrist (4 dims)
    features.extend([
        dist(lm[8], wrist),   # Index
        dist(lm[12], wrist),  # Middle
        dist(lm[16], wrist),  # Ring
        dist(lm[20], wrist)   # Pinky
    ])

    # 3. Thumb-to-index distance (1 dim)
    features.append(dist(lm[4], lm[8]))

    return np.array(features).reshape(1, -1)

# ==========================================
# MAIN
# ==========================================
def main():
    print("\n")
    print("="*60)
    print("ASL LIGHTING CONDITION ACCURACY TEST (LOCAL)")
    print("="*60)
    
    # 0. User Input
    print("Select Lighting Condition to Label this Test:")
    print("1. Standard (LED, ~500 lux)")
    print("2. Overexposed (>2000 lux)")
    print("3. Low-light (<50 lux)")
    print("4. Custom")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
    except EOFError:
        return

    conditions = {
        '1': "Standard (LED, ~500 lux)",
        '2': "Overexposed (>2000 lux)",
        '3': "Low-light (<50 lux)",
        '4': "Custom"
    }
    condition_name = conditions.get(choice, "Unknown")
    
    # 1. Setup
    model, labels, hands_detector, drawer = setup_local_inference()

    # 2. Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print(f"\nCondition Selected: {condition_name}")
    print("Instructions:")
    print(f"- You will be asked to perform: {', '.join(TARGET_GESTURES)}")
    print("- Prepare your environment (adjust lights) NOW.")
    print("- Press ENTER when ready to start.")
    input()
    
    cv2.namedWindow("Local Accuracy Test", cv2.WINDOW_NORMAL)
    
    # Metrics
    total_frames_all = 0
    total_detected_all = 0
    total_correct_all = 0
    
    try:
        for target in TARGET_GESTURES:
            target_metrics = {"total": 0, "detected": 0, "correct": 0}
            
            # --- PHASE 1: PREPARE ---
            start_prep = time.time()
            while time.time() - start_prep < PREPARE_DURATION:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                
                # Visuals
                remaining = int(PREPARE_DURATION - (time.time() - start_prep)) + 1
                cv2.rectangle(frame, (0,0), (640, 480), (40, 40, 40), -1)
                cv2.putText(frame, f"NEXT: {target}", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                cv2.putText(frame, f"Starts in {remaining}...", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow("Local Accuracy Test", frame)
                cv2.waitKey(1)
            
            # --- PHASE 2: RECORD & INFERENCE ---
            print(f"Testing Gesture: {target}...")
            start_test = time.time()
            
            while time.time() - start_test < TEST_DURATION:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process
                results = hands_detector.process(rgb)
                
                target_metrics["total"] += 1
                detected_text = ""
                box_color = (0, 0, 255) # Red default
                
                if results.multi_hand_landmarks:
                    target_metrics["detected"] += 1
                    
                    hand_landmarks = results.multi_hand_landmarks[0]
                    drawer.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    
                    # Inference
                    features = extract_features(hand_landmarks)
                    pred = model.predict(features, verbose=0)[0]
                    idx = np.argmax(pred)
                    conf = pred[idx]
                    
                    if conf > 0.6: # Confidence threshold
                        detected_text = labels[idx]
                        
                        if detected_text == target:
                            target_metrics["correct"] += 1
                            box_color = (0, 255, 0) # Green for match
                    
                    # Debug text
                    cv2.putText(frame, f"Det: {detected_text} ({conf:.2f})", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                # Overlay
                rem = int(TEST_DURATION - (time.time() - start_test)) + 1
                cv2.putText(frame, f"TARGET: {target}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
                cv2.putText(frame, f"{rem}s", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                
                cv2.imshow("Local Accuracy Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Accumulate
            total_frames_all += target_metrics["total"]
            total_detected_all += target_metrics["detected"]
            total_correct_all += target_metrics["correct"]
            
            print(f"  -> Total: {target_metrics['total']}, Det: {target_metrics['detected']}, Corr: {target_metrics['correct']}")

    except KeyboardInterrupt:
        print("\nTest stopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands_detector.close()
    
    # 3. Report
    print("\n\n")
    print("="*60)
    print("TABLE VIII: SGN ACCURACY ACROSS LIGHTING CONDITIONS")
    print("="*60)
    print(f"{'Condition':<25} {'Acc. (%)':<10} {'Detection rate':<15}")
    print("-" * 60)
    
    detection_rate = 0.0
    if total_frames_all > 0:
        detection_rate = (total_detected_all / total_frames_all) * 100
        
    accuracy = 0.0
    if total_detected_all > 0:
        accuracy = (total_correct_all / total_detected_all) * 100
    
    print(f"{condition_name:<25} {accuracy:<10.1f} {detection_rate:.1f}%")
    print("-" * 60)
    print(f"Total Frames: {total_frames_all}, Detected: {total_detected_all}, Correct: {total_correct_all}")
    print("="*60)

if __name__ == "__main__":
    main()
