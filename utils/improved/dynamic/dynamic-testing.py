"""
DYNAMIC GESTURE TESTING
========================
Real-time testing of dynamic gestures

USAGE:
python test_dynamic_model.py
"""

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = "../models/dynamic_model_final.h5"
SEQUENCE_LENGTH = 30  # Must match training

# Dynamic gesture labels
GESTURES = ['BYE', 'HELLO', 'J', 'NO', 'YES', 'Z']  # Update based on what you collected

CONFIDENCE_THRESHOLD = 0.70

# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_features(hand_landmarks):
    """Extract 68 features"""
    
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
# SEQUENCE BUFFER
# ============================================

class SequenceBuffer:
    """Buffer to collect sequences of frames"""
    
    def __init__(self, max_length=30):
        self.max_length = max_length
        self.buffer = deque(maxlen=max_length)
        self.recording = False
    
    def add_frame(self, features):
        """Add frame to buffer"""
        self.buffer.append(features)
    
    def get_sequence(self):
        """Get current sequence"""
        if len(self.buffer) < self.max_length:
            # Pad with zeros if not enough frames
            padding = np.zeros((self.max_length - len(self.buffer), 68))
            sequence = np.vstack([padding, list(self.buffer)])
        else:
            sequence = np.array(list(self.buffer))
        
        return sequence
    
    def is_ready(self):
        """Check if buffer has enough frames"""
        return len(self.buffer) >= self.max_length
    
    def reset(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

# ============================================
# MAIN TESTING
# ============================================

def main():
    """Main testing loop"""
    
    print("\n" + "="*70)
    print("DYNAMIC GESTURE RECOGNITION - LIVE TESTING")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Initialize MediaPipe
    print("Initializing hand detection...")
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    print("✓ Hand detection ready!")
    
    # Initialize camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("✗ Error: Could not open camera!")
        return
    
    print("✓ Camera ready!")
    
    # Initialize sequence buffer
    sequence_buffer = SequenceBuffer(max_length=SEQUENCE_LENGTH)
    
    print("\n" + "="*70)
    print("INSTRUCTIONS:")
    print("  • Perform gesture in front of camera")
    print("  • Hold for ~1 second")
    print("  • Press R to reset buffer")
    print("  • Press Q to quit")
    print("="*70 + "\n")
    
    prediction_text = "Collecting frames..."
    confidence = 0.0
    prediction_color = (255, 255, 255)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Failed to grab frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        result = hands.process(rgb)
        
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Extract features and add to buffer
            features = extract_features(hand)
            sequence_buffer.add_frame(features)
            
            # Predict if buffer is ready
            if sequence_buffer.is_ready():
                sequence = sequence_buffer.get_sequence()
                sequence = sequence.reshape(1, SEQUENCE_LENGTH, 68)
                
                # Predict
                predictions = model.predict(sequence, verbose=0)[0]
                predicted_idx = np.argmax(predictions)
                confidence = predictions[predicted_idx]
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction_text = f"{GESTURES[predicted_idx]}"
                    prediction_color = (0, 255, 0)
                else:
                    prediction_text = f"{GESTURES[predicted_idx]}?"
                    prediction_color = (0, 255, 255)
        else:
            # No hand detected - clear buffer
            sequence_buffer.reset()
            prediction_text = "No hand detected"
            prediction_color = (0, 0, 255)
            confidence = 0.0
        
        # Draw UI
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Prediction
        cv2.putText(frame, f"Gesture: {prediction_text}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, prediction_color, 3)
        
        # Confidence bar
        if confidence > 0:
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            bar_width = int(500 * confidence)
            cv2.rectangle(frame, (20, 110), (520, 140), (255, 255, 255), 2)
            cv2.rectangle(frame, (20, 110), (20 + bar_width, 140), prediction_color, -1)
        
        # Buffer status
        buffer_status = f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH} frames"
        buffer_color = (0, 255, 0) if sequence_buffer.is_ready() else (255, 255, 255)
        cv2.putText(frame, buffer_status, (20, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, buffer_color, 2)
        
        # Controls
        height = frame.shape[0]
        cv2.putText(frame, "R - Reset | Q - Quit", (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Dynamic Gesture Recognition', frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            sequence_buffer.reset()
            print("Buffer reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print("\n✓ Testing complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Testing interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()