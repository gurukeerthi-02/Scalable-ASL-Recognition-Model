"""
PORTABLE DYNAMIC ASL DATA COLLECTOR
====================================
Complete standalone script for collecting dynamic gesture data
Friends can run this on their laptops and send you the ZIP file

FEATURES:
- Collects all dynamic gestures in one session
- Built-in gesture guides
- Quality checks
- Auto-generates ZIP file
- Progress tracking

USAGE:
python collect_dynamic_portable.py
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

# Dynamic gestures to collect
# Basic set (for quick testing)
# GESTURES = ['J', 'Z', 'HELLO', 'BYE']

# Extended set (recommended)
GESTURES = ['J', 'Z', 'HELLO', 'BYE', 'YES', 'NO', 'PLEASE', 'THANKYOU', 
            'REPEAT', 'IMFINE', 'DONE', 'HOWAREYOU', 'NICETOMEETYOU']

SAMPLES_PER_GESTURE = 100
SEQUENCE_LENGTH = 30  # 30 frames per sequence (~1 second)

# ============================================
# GESTURE GUIDES
# ============================================

GESTURE_GUIDES = {
    'J': {
        'desc': 'Draw letter J in the air with pinky finger',
        'steps': [
            'Extend pinky finger (I hand shape)',
            'Move straight down',
            'Curve left at bottom (hook)',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
        START
          |
          |  (straight down)
          |
          └─  (curve left)
        ''',
        'tips': 'Keep other fingers closed, smooth J shape'
    },
    'Z': {
        'desc': 'Draw letter Z in the air with index finger',
        'steps': [
            'Extend index finger',
            'Draw horizontal line (left→right)',
            'Draw diagonal (down-left)',
            'Draw horizontal line (left→right)',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
        ─────►
         ╲
          ╲
           ─────►
        ''',
        'tips': 'Sharp angles, clear zigzag pattern'
    },
    'HELLO': {
        'desc': 'Wave hand side to side',
        'steps': [
            'Open hand, palm facing forward',
            'Move right, then left',
            'Repeat 2-3 times',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
          ◄──────►
         (wave side to side)
        ''',
        'tips': 'Smooth waving, keep fingers together'
    },
    'BYE': {
        'desc': 'Wave hand up and down',
        'steps': [
            'Open hand, palm facing forward',
            'Move up, then down',
            'Repeat 2-3 times',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
            ▲
            │
            │  (wave up and down)
            ▼
        ''',
        'tips': 'Vertical waving motion'
    },
    'THANKYOU': {
        'desc': 'Touch chin, move hand forward',
        'steps': [
            'Flat hand near chin',
            'Fingertips touching chin',
            'Move hand forward/down',
            'End with palm up',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
        chin → → → forward
        ''',
        'tips': 'Graceful forward motion'
    },
    'PLEASE': {
        'desc': 'Circle hand on chest',
        'steps': [
            'Flat hand on chest',
            'Make circular motion',
            'Clockwise circle',
            '2-3 circles',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
            ╭─╮
            │ │  (circle on chest)
            ╰─╯
        ''',
        'tips': 'Smooth circle, stay on chest'
    },
    'YES': {
        'desc': 'Fist nods up and down (like nodding)',
        'steps': [
            'Make fist (S hand shape)',
            'Move fist up',
            'Move fist down',
            'Repeat 2-3 times',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
            ▲
          ╔═╗ (fist nodding)
          ║ ║
            ▼
        ''',
        'tips': 'Like nodding your head with fist'
    },
    'NO': {
        'desc': 'Index and middle fingers snap together',
        'steps': [
            'Extend index and middle fingers',
            'Bring fingers together (snap)',
            'Separate fingers',
            'Repeat 2-3 times',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
          ✌ → ✊ → ✌
         (snap together)
        ''',
        'tips': 'Quick snapping motion, like scissors'
    },
    'REPEAT': {
        'desc': 'Bent hand bounces back to starting position',
        'steps': [
            'Bent hand, fingertips touch',
            'Move forward',
            'Bounce back to chest',
            'Repeat motion',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
        chest ⟲ forward ⟲ chest
         (bouncing motion)
        ''',
        'tips': 'Circular bouncing, like "again"'
    },
    'IMFINE': {
        'desc': 'Flat hand taps chest, moves forward with thumb up',
        'steps': [
            'Flat hand on chest',
            'Tap chest lightly',
            'Move hand forward',
            'End with thumb up',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
        chest → tap → forward (👍)
        ''',
        'tips': 'Two-part: chest tap, then forward with thumb'
    },
    'DONE': {
        'desc': 'Hands flip from palms up to palms down',
        'steps': [
            'Both hands up, palms facing up',
            'Flip hands over',
            'End with palms facing down',
            'Quick flipping motion',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
          🙌 (palms up)
           ↓ (flip)
          🙌 (palms down)
        ''',
        'tips': 'Quick flip, both hands together'
    },
    'HOWAREYOU': {
        'desc': 'Bent hands rotate forward from chest',
        'steps': [
            'Bent hands at chest level',
            'Fingertips pointing at each other',
            'Rotate hands forward and up',
            'Open hands while rotating',
            'Complete in 1-2 seconds'
        ],
        'demo': '''
        chest ⟲⟲ forward+up
         (rotating outward)
        ''',
        'tips': 'Smooth rotation, hands open at end'
    },
    'NICETOMEETYOU': {
        'desc': 'Index finger moves from mouth forward',
        'steps': [
            'Index finger touches lips/chin',
            'Move hand forward',
            'Arc downward',
            'End palm up toward person',
            'Complete in 2 seconds'
        ],
        'demo': '''
        lips → → → forward ↘
         (graceful arc)
        ''',
        'tips': 'Start at face, smooth arc forward'
    }
}

# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_features(hand_landmarks):
    """Extract 68 features from hand landmarks"""
    
    def dist(a, b):
        return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))
    
    lm = hand_landmarks.landmark
    wrist = lm[0]
    features = []
    
    # Normalized landmarks (63)
    for point in lm:
        features.extend([
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z
        ])
    
    # Finger curl distances (4)
    features.extend([
        dist(lm[8],  wrist),
        dist(lm[12], wrist),
        dist(lm[16], wrist),
        dist(lm[20], wrist)
    ])
    
    # Thumb-index distance (1)
    features.append(dist(lm[4], lm[8]))
    
    return np.array(features)

# ============================================
# UI FUNCTIONS
# ============================================

def show_gesture_guide(gesture):
    """Display gesture guide in console"""
    guide = GESTURE_GUIDES.get(gesture, {})
    
    print("\n" + "="*70)
    print(f"GESTURE: {gesture}")
    print("="*70)
    print(f"\n{guide.get('desc', '')}\n")
    
    print("Steps:")
    for i, step in enumerate(guide.get('steps', []), 1):
        print(f"  {i}. {step}")
    
    print(f"\nVisualization:")
    print(guide.get('demo', ''))
    
    print(f"\n💡 Tip: {guide.get('tips', '')}")
    print("="*70)

def draw_ui(frame, gesture, countdown, recording, buffer_len, sample_count, total_samples):
    """Draw UI overlay"""
    
    height, width = frame.shape[:2]
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Gesture name
    cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Sample count
    cv2.putText(frame, f"Samples: {sample_count}/{total_samples}", (20, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Status
    y_offset = 120
    if countdown > 0:
        cv2.putText(frame, f"GET READY: {countdown}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
    elif recording:
        progress = buffer_len / SEQUENCE_LENGTH
        cv2.putText(frame, f"RECORDING: {buffer_len}/{SEQUENCE_LENGTH}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Progress bar
        y_offset += 40
        bar_width = int(600 * progress)
        cv2.rectangle(frame, (20, y_offset), (620, y_offset + 25), (255, 255, 255), 2)
        cv2.rectangle(frame, (20, y_offset), (20 + bar_width, y_offset + 25), (0, 255, 0), -1)
    else:
        cv2.putText(frame, "Press SPACE to record", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Instructions
    y_offset = height - 60
    cv2.putText(frame, "SPACE - Record | S - Skip gesture | Q - Quit", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

# ============================================
# COLLECT SINGLE GESTURE
# ============================================

def collect_gesture(gesture, person_name, base_dir, cap, hands, mp_draw, mp_hands):
    """Collect samples for one gesture"""
    
    save_dir = os.path.join(base_dir, gesture)
    os.makedirs(save_dir, exist_ok=True)
    
    show_gesture_guide(gesture)
    input(f"\nPress ENTER to start collecting {gesture}...")
    
    count = 0
    recording = False
    sequence_buffer = []
    countdown = 0
    countdown_start = 0
    
    print(f"\nCollecting {SAMPLES_PER_GESTURE} samples for '{gesture}'...")
    print("Press SPACE to start each recording\n")
    
    while count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            return count
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        # Handle countdown
        if countdown > 0:
            elapsed = time.time() - countdown_start
            countdown = max(0, 3 - int(elapsed))
            
            if countdown == 0:
                recording = True
                sequence_buffer = []
        
        # Draw landmarks
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Record if active
            if recording:
                features = extract_features(hand)
                sequence_buffer.append(features)
                
                # Check if complete
                if len(sequence_buffer) >= SEQUENCE_LENGTH:
                    # Save
                    sequence_array = np.array(sequence_buffer)
                    filename = f"{person_name}_{count:04d}.npy"
                    np.save(os.path.join(save_dir, filename), sequence_array)
                    
                    count += 1
                    recording = False
                    sequence_buffer = []
                    
                    print(f"  ✓ Saved {count}/{SAMPLES_PER_GESTURE}")
        
        # Draw UI
        draw_ui(frame, gesture, countdown, recording, len(sequence_buffer), 
                count, SAMPLES_PER_GESTURE)
        
        cv2.imshow("Dynamic Data Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return count
        elif key == ord('s'):
            print(f"\n⊳ Skipping {gesture}")
            return count
        elif key == ord(' ') and not recording and countdown == 0:
            countdown = 3
            countdown_start = time.time()
    
    print(f"\n✓ Completed {gesture}!")
    return count

# ============================================
# MAIN COLLECTION
# ============================================

def main():
    """Main collection workflow"""
    
    print("\n" + "="*70)
    print("DYNAMIC ASL DATA COLLECTION - PORTABLE VERSION")
    print("="*70)
    
    # Get person name
    person_name = input("\nEnter your name: ").strip().lower()
    if not person_name:
        person_name = f"person_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\nCollector: {person_name}")
    print(f"Gestures to collect: {', '.join(GESTURES)}")
    print(f"Samples per gesture: {SAMPLES_PER_GESTURE}")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames")
    print(f"Total samples: {len(GESTURES) * SAMPLES_PER_GESTURE}")
    
    estimated_time = len(GESTURES) * SAMPLES_PER_GESTURE * 3 / 60  # ~3 sec per sample
    print(f"\nEstimated time: {estimated_time:.0f} minutes")
    
    # Create base directory
    base_dir = f"asl_dynamic_{person_name}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Save collection info
    info_path = os.path.join(base_dir, "collection_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Collector: {person_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Gestures: {', '.join(GESTURES)}\n")
        f.write(f"Samples per gesture: {SAMPLES_PER_GESTURE}\n")
        f.write(f"Sequence length: {SEQUENCE_LENGTH}\n")
        f.write(f"Feature dimensions: (30, 68)\n")
    
    input("\nPress ENTER to begin...")
    
    # Initialize camera and MediaPipe
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Could not open camera!")
        return
    
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    print("✓ Camera ready!")
    
    # Collect each gesture
    results = {}
    
    for i, gesture in enumerate(GESTURES, 1):
        print(f"\n{'='*70}")
        print(f"GESTURE {i}/{len(GESTURES)}")
        print(f"{'='*70}")
        
        count = collect_gesture(gesture, person_name, base_dir, cap, hands, mp_draw, mp_hands)
        results[gesture] = count
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # Summary
    print("\n" + "="*70)
    print("COLLECTION SUMMARY")
    print("="*70)
    
    total_collected = sum(results.values())
    total_expected = len(GESTURES) * SAMPLES_PER_GESTURE
    
    for gesture, count in results.items():
        status = "✓" if count == SAMPLES_PER_GESTURE else "⚠"
        print(f"{status} {gesture}: {count}/{SAMPLES_PER_GESTURE} samples")
    
    print("-" * 70)
    print(f"Total: {total_collected}/{total_expected} samples")
    print("="*70)
    
    # Create ZIP file
    print("\n📦 Creating ZIP file...")
    
    zip_filename = f"asl_dynamic_{person_name}_{datetime.now().strftime('%Y%m%d')}.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(base_dir))
                zipf.write(file_path, arcname)
    
    file_size = os.path.getsize(zip_filename) / (1024 * 1024)
    
    print(f"✓ Package created: {zip_filename} ({file_size:.2f} MB)")
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"\n📧 Send '{zip_filename}' to the project owner")
    print("\nThank you for contributing! 🎉")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Collection interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()