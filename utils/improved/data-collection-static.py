import cv2
import mediapipe as mp
import numpy as np
import os
import time

LABEL = "Y"     # change this when collecting other gestures
SAMPLES = 500

SAVE_DIR = f"../improved_dataset/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================================
# AUGMENTATION FUNCTIONS
# ============================================

def add_noise(features, noise_level=0.02):
    """Add random Gaussian noise to features"""
    noise = np.random.normal(0, noise_level, features.shape)
    return features + noise

def scale_hand(features, scale_range=(0.9, 1.1)):
    """Simulate different hand sizes/distances"""
    scale = np.random.uniform(*scale_range)
    # Scale only positional features (first 63), not distances
    features_copy = features.copy()
    features_copy[:63] *= scale
    return features_copy

def rotate_hand(features, angle_range=(-15, 15)):
    """Rotate hand landmarks slightly"""
    angle = np.radians(np.random.uniform(*angle_range))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    features_copy = features.copy()
    # Rotate x,y coordinates (every 3rd element starting from 0 and 1)
    for i in range(0, 63, 3):
        x, y = features_copy[i], features_copy[i+1]
        features_copy[i] = x * cos_a - y * sin_a
        features_copy[i+1] = x * sin_a + y * cos_a
    
    return features_copy

def mirror_hand(features):
    """Flip hand horizontally (for symmetric gestures)"""
    features_copy = features.copy()
    # Flip x-coordinates (every 3rd element starting from 0)
    for i in range(0, 63, 3):
        features_copy[i] *= -1
    return features_copy

# ============================================
# COLLECTION WITH VARIATION PROMPTS
# ============================================

count = 0
last_save_time = time.time()
save_interval = 0.1  # Save every 100ms to get varied poses

# Variation prompts
prompts = [
    "Normal pose",
    "Move hand closer to camera",
    "Move hand farther away",
    "Tilt hand slightly left",
    "Tilt hand slightly right",
    "Change hand position",
    "Different angle"
]
current_prompt_idx = 0
prompt_change_interval = SAMPLES // len(prompts)

print("\n=== ENHANCED DATA COLLECTION ===")
print("Tips for better diversity:")
print("- Change your hand position frequently")
print("- Move closer/farther from camera")
print("- Tilt your hand at different angles")
print("- Vary lighting (turn slightly)")
print("=====================================\n")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    # Show current prompt
    if count % prompt_change_interval == 0 and count > 0:
        current_prompt_idx = min(current_prompt_idx + 1, len(prompts) - 1)
    
    cv2.putText(frame, f"Prompt: {prompts[current_prompt_idx]}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Collected: {count}/{SAMPLES}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

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
        features.extend([
            dist(lm[8],  wrist),
            dist(lm[12], wrist),
            dist(lm[16], wrist),
            dist(lm[20], wrist)
        ])

        # ---- THUMB–INDEX DISTANCE (1) ----
        features.append(dist(lm[4], lm[8]))

        features = np.array(features)

        # Save with time-based throttling for natural variation
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            np.save(f"{SAVE_DIR}/{count}.npy", features)
            count += 1
            last_save_time = current_time
            print(f"Saved {count}/{SAMPLES}")

    cv2.imshow("Collecting Data", frame)

    if count >= SAMPLES:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✓ Data collection complete!")
print(f"✓ Saved {count} samples to {SAVE_DIR}")