"""
COMPREHENSIVE ASL MODEL TESTING SCRIPT
=======================================
Real-time gesture recognition with performance metrics and visualization

FEATURES:
- Live webcam prediction
- Confidence scores
- Prediction smoothing
- Performance statistics
- Recording mode for evaluation
- Visual feedback

USAGE:
python test_model_comprehensive.py
"""

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time
import json
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = "../models/static_model_person_split.h5"  # Your improved model
CONFIDENCE_THRESHOLD = 0.60  # Only show predictions above 60% confidence
SMOOTHING_WINDOW = 5  # Average last 5 predictions for stability

# All ASL labels
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (0, 165, 255)

# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_features(hand_landmarks):
    """Extract 68 features from hand landmarks (same as training)"""
    
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
# PREDICTION SMOOTHING
# ============================================

class PredictionSmoother:
    """Smooth predictions using moving average"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
    
    def add(self, prediction):
        """Add new prediction to buffer"""
        self.predictions.append(prediction)
    
    def get_smooth_prediction(self):
        """Get averaged prediction"""
        if len(self.predictions) == 0:
            return None, 0.0
        
        # Average the prediction probabilities
        avg_probs = np.mean([p for p in self.predictions], axis=0)
        predicted_class = np.argmax(avg_probs)
        confidence = avg_probs[predicted_class]
        
        return predicted_class, confidence
    
    def reset(self):
        """Clear buffer"""
        self.predictions.clear()

# ============================================
# PERFORMANCE TRACKER
# ============================================

class PerformanceTracker:
    """Track performance statistics"""
    
    def __init__(self):
        self.predictions = []
        self.confidences = []
        self.processing_times = []
        self.start_time = time.time()
    
    def add_prediction(self, label, confidence, processing_time):
        """Record a prediction"""
        self.predictions.append(label)
        self.confidences.append(confidence)
        self.processing_times.append(processing_time)
    
    def get_stats(self):
        """Get performance statistics"""
        if len(self.predictions) == 0:
            return None
        
        return {
            'total_predictions': len(self.predictions),
            'avg_confidence': np.mean(self.confidences),
            'max_confidence': np.max(self.confidences),
            'min_confidence': np.min(self.confidences),
            'avg_processing_time': np.mean(self.processing_times) * 1000,  # ms
            'fps': len(self.predictions) / (time.time() - self.start_time),
            'unique_predictions': len(set(self.predictions)),
            'most_common': max(set(self.predictions), key=self.predictions.count)
        }
    
    def save_to_file(self, filename="test_results.json"):
        """Save statistics to file"""
        stats = self.get_stats()
        if stats:
            stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(filename, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\n✓ Stats saved to {filename}")

# ============================================
# UI DRAWING FUNCTIONS
# ============================================

def draw_info_panel(frame, prediction, confidence, fps, stats=None):
    """Draw information panel on frame"""
    
    height, width = frame.shape[:2]
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Main prediction
    if prediction is not None:
        color = COLOR_GREEN if confidence >= CONFIDENCE_THRESHOLD else COLOR_YELLOW
        cv2.putText(frame, f"Prediction: {LABELS[prediction]}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Confidence bar
        bar_width = int(300 * confidence)
        cv2.rectangle(frame, (20, 60), (320, 80), COLOR_WHITE, 2)
        cv2.rectangle(frame, (20, 60), (20 + bar_width, 80), color, -1)
        cv2.putText(frame, f"{confidence*100:.1f}%", (330, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    else:
        cv2.putText(frame, "Prediction: ---", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_RED, 3)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 110),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    
    # Session stats (if available)
    if stats:
        cv2.putText(frame, f"Total: {stats['total_predictions']}", (150, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
        cv2.putText(frame, f"Avg Conf: {stats['avg_confidence']*100:.1f}%", (280, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

def draw_controls(frame):
    """Draw control instructions"""
    
    height, width = frame.shape[:2]
    
    controls = [
        "Controls:",
        "  [SPACE] - Pause/Resume",
        "  [R] - Reset stats",
        "  [S] - Save stats",
        "  [C] - Clear smoothing",
        "  [Q] - Quit"
    ]
    
    y_offset = height - 150
    for i, text in enumerate(controls):
        cv2.putText(frame, text, (10, y_offset + i*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

def draw_confidence_meter(frame, all_probabilities, labels):
    """Draw top-3 predictions with confidence"""
    
    height, width = frame.shape[:2]
    x_start = width - 250
    y_start = 200
    
    # Get top 3 predictions
    top_indices = np.argsort(all_probabilities)[-3:][::-1]
    
    cv2.putText(frame, "Top Predictions:", (x_start, y_start - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    
    for i, idx in enumerate(top_indices):
        label = labels[idx]
        conf = all_probabilities[idx]
        
        y = y_start + i * 40
        
        # Label
        cv2.putText(frame, f"{label}:", (x_start, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        
        # Bar
        bar_width = int(150 * conf)
        color = COLOR_GREEN if i == 0 else COLOR_BLUE
        cv2.rectangle(frame, (x_start + 30, y - 10), (x_start + 180, y + 10),
                     COLOR_WHITE, 1)
        cv2.rectangle(frame, (x_start + 30, y - 10), (x_start + 30 + bar_width, y + 10),
                     color, -1)
        
        # Percentage
        cv2.putText(frame, f"{conf*100:.0f}%", (x_start + 185, y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

# ============================================
# MAIN TESTING FUNCTION
# ============================================

def main():
    """Main testing loop"""
    
    print("\n" + "="*70)
    print("ASL GESTURE RECOGNITION - LIVE TESTING")
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
    
    # Initialize components
    smoother = PredictionSmoother(window_size=SMOOTHING_WINDOW)
    tracker = PerformanceTracker()
    
    # State variables
    paused = False
    fps = 0
    last_time = time.time()
    
    print("\n" + "="*70)
    print("STARTING LIVE TESTING - Show ASL gestures to camera")
    print("="*70 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Failed to grab frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
        last_time = current_time
        
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        if not paused:
            result = hands.process(rgb)
            
            prediction_label = None
            confidence = 0.0
            all_probs = None
            
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                
                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract features and predict
                start_pred = time.time()
                features = extract_features(hand)
                features = features.reshape(1, -1)
                
                predictions = model.predict(features, verbose=0)[0]
                pred_time = time.time() - start_pred
                
                # Add to smoother
                smoother.add(predictions)
                
                # Get smoothed prediction
                smooth_class, smooth_conf = smoother.get_smooth_prediction()
                
                if smooth_class is not None and smooth_conf >= CONFIDENCE_THRESHOLD:
                    prediction_label = smooth_class
                    confidence = smooth_conf
                    all_probs = predictions
                    
                    # Track performance
                    tracker.add_prediction(LABELS[prediction_label], confidence, pred_time)
                
                # Draw top-3 predictions
                if all_probs is not None:
                    draw_confidence_meter(frame, all_probs, LABELS)
        
        # Get stats
        stats = tracker.get_stats()
        
        # Draw UI
        draw_info_panel(frame, prediction_label, confidence, fps, stats)
        draw_controls(frame)
        
        # Pause indicator
        if paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 100, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_RED, 4)
        
        # Show frame
        cv2.imshow('ASL Recognition Testing', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit
            break
        elif key == ord(' '):
            # Pause/Resume
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord('r'):
            # Reset stats
            tracker = PerformanceTracker()
            smoother.reset()
            print("Stats reset")
        elif key == ord('s'):
            # Save stats
            tracker.save_to_file()
        elif key == ord('c'):
            # Clear smoothing buffer
            smoother.reset()
            print("Smoothing cleared")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # Final statistics
    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70)
    
    stats = tracker.get_stats()
    if stats:
        print(f"Total Predictions:      {stats['total_predictions']}")
        print(f"Average Confidence:     {stats['avg_confidence']*100:.2f}%")
        print(f"Max Confidence:         {stats['max_confidence']*100:.2f}%")
        print(f"Min Confidence:         {stats['min_confidence']*100:.2f}%")
        print(f"Average Processing:     {stats['avg_processing_time']:.2f} ms")
        print(f"Average FPS:            {stats['fps']:.1f}")
        print(f"Unique Predictions:     {stats['unique_predictions']}")
        print(f"Most Common:            {stats['most_common']}")
        
        # Save final stats
        tracker.save_to_file()
    else:
        print("No predictions made")
    
    print("="*70)
    print("\n✓ Testing complete!")

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Testing interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()