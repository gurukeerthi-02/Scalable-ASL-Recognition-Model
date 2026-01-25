"""
PORTABLE ASL DATA COLLECTOR
============================
Simple standalone script for friends to collect ASL gesture data
No complex setup needed - just run and follow prompts!

USAGE:
1. Install requirements: pip install opencv-python mediapipe numpy
2. Run: python collect_data_portable.py
3. Follow on-screen instructions
4. Send the generated ZIP file back to you
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import zipfile
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

# All 24 ASL letters (excluding J and Z - dynamic gestures)
ALL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

SAMPLES_PER_LABEL = 200  # Reduced from 1000 for friends
SAVE_INTERVAL = 0.15     # Save every 150ms for natural variation

# ============================================
# SETUP
# ============================================

def setup_collection():
    """Get person ID and create directories"""
    print("\n" + "="*60)
    print("     ASL GESTURE DATA COLLECTION")
    print("="*60)
    
    person_id = input("\nEnter your name (e.g., john, sarah): ").strip().lower()
    if not person_id:
        person_id = f"person_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    base_dir = f"asl_data_{person_id}"
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"\n✓ Data will be saved to: {base_dir}/")
    return person_id, base_dir

def create_info_file(base_dir, person_id):
    """Create info file with metadata"""
    info_path = os.path.join(base_dir, "collection_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"Collector: {person_id}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Labels: {', '.join(ALL_LABELS)}\n")
        f.write(f"Samples per label: {SAMPLES_PER_LABEL}\n")
        f.write(f"Total samples: {len(ALL_LABELS) * SAMPLES_PER_LABEL}\n")

# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_features(hand_landmarks):
    """Extract 68 features from hand landmarks"""
    def dist(a, b):
        return np.linalg.norm(
            np.array([a.x, a.y]) - np.array([b.x, b.y])
        )
    
    lm = hand_landmarks.landmark
    wrist = lm[0]
    features = []
    
    # Normalized landmarks (63 features)
    for point in lm:
        features.extend([
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z
        ])
    
    # Finger curl distances (4 features)
    features.extend([
        dist(lm[8],  wrist),  # Index
        dist(lm[12], wrist),  # Middle
        dist(lm[16], wrist),  # Ring
        dist(lm[20], wrist)   # Pinky
    ])
    
    # Thumb-index distance (1 feature)
    features.append(dist(lm[4], lm[8]))
    
    return np.array(features)

# ============================================
# SHOW ASL REFERENCE
# ============================================

def show_instructions(label, count, total):
    """Display collection instructions"""
    instructions = {
        'A': 'Fist with thumb to the side',
        'B': 'Flat hand, fingers together, thumb across palm',
        'C': 'Curved hand forming C shape',
        'D': 'Index finger up, thumb and middle finger touching',
        'E': 'All fingers curled down touching thumb',
        'F': 'Index and thumb touching in circle, other fingers up',
        'G': 'Index and thumb pointing horizontally',
        'H': 'Index and middle fingers together pointing horizontally',
        'I': 'Pinky up, other fingers down',
        'K': 'Index up, middle finger out, thumb between them',
        'L': 'Index up, thumb out at 90 degrees',
        'M': 'Thumb under first three fingers',
        'N': 'Thumb under first two fingers',
        'O': 'All fingers curved touching thumb in O shape',
        'P': 'Like K but pointing down',
        'Q': 'Like G but pointing down',
        'R': 'Index and middle crossed',
        'S': 'Fist with thumb over fingers',
        'T': 'Thumb between index and middle fingers',
        'U': 'Index and middle fingers together pointing up',
        'V': 'Index and middle fingers apart pointing up',
        'W': 'Index, middle, and ring fingers up and apart',
        'X': 'Index finger bent in hook shape',
        'Y': 'Thumb and pinky out, other fingers down'
    }
    
    print("\n" + "="*60)
    print(f"  COLLECTING: Letter '{label}' ({count}/{total})")
    print("="*60)
    print(f"  Description: {instructions.get(label, 'See ASL reference')}")
    print("="*60)
    print("\n  TIPS:")
    print("  • Hold the gesture steady")
    print("  • Move your hand around slightly (don't keep it frozen)")
    print("  • Vary the distance from camera")
    print("  • Try slightly different angles")
    print("  • Press 's' to skip this letter")
    print("  • Press 'q' to quit completely\n")

# ============================================
# MAIN COLLECTION FUNCTION
# ============================================

def collect_gesture(label, save_dir, cap, hands, mp_draw, mp_hands):
    """Collect samples for a single gesture"""
    label_dir = os.path.join(save_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    
    show_instructions(label, 0, SAMPLES_PER_LABEL)
    
    count = 0
    last_save_time = time.time()
    
    # Countdown before starting
    print("  Starting in: ", end='', flush=True)
    for i in range(3, 0, -1):
        print(f"{i}... ", end='', flush=True)
        time.sleep(1)
    print("GO!\n")
    
    while count < SAMPLES_PER_LABEL:
        ret, frame = cap.read()
        if not ret:
            print("  ✗ Camera error!")
            return False
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        # Progress bar
        progress = count / SAMPLES_PER_LABEL
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Display info on frame
        cv2.putText(frame, f"Letter: {label}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, f"Progress: {count}/{SAMPLES_PER_LABEL}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"[{bar}] {progress*100:.1f}%", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            
            # Draw landmarks
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Extract and save features
            current_time = time.time()
            if current_time - last_save_time >= SAVE_INTERVAL:
                features = extract_features(hand)
                np.save(os.path.join(label_dir, f"{count}.npy"), features)
                count += 1
                last_save_time = current_time
                
                # Console progress
                print(f"\r  Progress: [{bar}] {count}/{SAMPLES_PER_LABEL} ({progress*100:.1f}%)", 
                      end='', flush=True)
        else:
            cv2.putText(frame, "NO HAND DETECTED!", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('ASL Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False  # Quit
        elif key == ord('s'):
            print("\n  ⊳ Skipping this letter...")
            return True  # Skip
    
    print(f"\n  ✓ Completed letter '{label}'!\n")
    return True

# ============================================
# PACKAGE DATA
# ============================================

def package_data(base_dir, person_id):
    """Create ZIP file for easy sharing"""
    zip_filename = f"asl_data_{person_id}_{datetime.now().strftime('%Y%m%d')}.zip"
    
    print(f"\n📦 Packaging data into {zip_filename}...")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(base_dir))
                zipf.write(file_path, arcname)
    
    file_size = os.path.getsize(zip_filename) / (1024 * 1024)  # MB
    print(f"✓ Package created: {zip_filename} ({file_size:.2f} MB)")
    print(f"\n📧 Send this file to the project owner!")
    
    return zip_filename

# ============================================
# MAIN
# ============================================

def main():
    """Main collection workflow"""
    
    # Setup
    person_id, base_dir = setup_collection()
    create_info_file(base_dir, person_id)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Initialize camera
    print("\n📷 Initializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ ERROR: Could not open camera!")
        return
    
    print("✓ Camera ready!")
    
    # Collect each gesture
    print(f"\n🎯 Will collect {len(ALL_LABELS)} letters × {SAMPLES_PER_LABEL} samples each")
    input("\nPress ENTER when ready to start...")
    
    completed = []
    skipped = []
    
    for i, label in enumerate(ALL_LABELS, 1):
        result = collect_gesture(label, base_dir, cap, hands, mp_draw, mp_hands)
        
        if result:
            completed.append(label)
        else:
            print(f"\n⊳ Collection stopped at letter '{label}'")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # Summary
    print("\n" + "="*60)
    print("  COLLECTION SUMMARY")
    print("="*60)
    print(f"  Completed: {len(completed)}/{len(ALL_LABELS)} letters")
    print(f"  Letters: {', '.join(completed)}")
    print(f"  Total samples: {len(completed) * SAMPLES_PER_LABEL}")
    print("="*60)
    
    if len(completed) > 0:
        package_data(base_dir, person_id)
        print("\n✓ All done! Thank you for contributing! 🎉")
    else:
        print("\n✗ No data collected.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Collection interrupted by user")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()