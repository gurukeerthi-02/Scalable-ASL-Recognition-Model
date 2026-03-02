"""
LETTER RECOGNITION DIAGNOSIS TOOL
==================================
Identifies which letters your model struggles with in real-time

USAGE:
python diagnose_letters.py
"""

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter, defaultdict
import time

MODEL_PATH = "../models/static_model_person_split_v3.h5"
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def extract_features(hand_landmarks):
    """Extract 68 features"""
    def dist(a, b):
        return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))
    
    lm = hand_landmarks.landmark
    wrist = lm[0]
    features = []
    
    for point in lm:
        features.extend([point.x - wrist.x, point.y - wrist.y, point.z - wrist.z])
    
    features.extend([dist(lm[8], wrist), dist(lm[12], wrist),
                    dist(lm[16], wrist), dist(lm[20], wrist)])
    features.append(dist(lm[4], lm[8]))
    
    return np.array(features)

def test_letter_recognition(target_letter, model, hands, cap, mp_draw, mp_hands):
    """Test recognition of a specific letter"""
    
    print(f"\n{'='*70}")
    print(f"Testing Letter: {target_letter}")
    print(f"{'='*70}")
    print("Show the gesture for this letter...")
    print("Collecting data for 5 seconds...")
    
    predictions = []
    confidences = []
    start_time = time.time()
    test_duration = 5  # seconds
    
    while time.time() - start_time < test_duration:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        # Progress
        elapsed = time.time() - start_time
        progress = elapsed / test_duration
        
        # UI
        cv2.putText(frame, f"Testing: {target_letter}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, f"Time: {test_duration - elapsed:.1f}s", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Progress bar
        bar_width = int(600 * progress)
        cv2.rectangle(frame, (20, 120), (620, 150), (255, 255, 255), 2)
        cv2.rectangle(frame, (20, 120), (20 + bar_width, 150), (0, 255, 0), -1)
        
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            features = extract_features(hand).reshape(1, -1)
            prediction = model.predict(features, verbose=0)[0]
            
            predicted_idx = np.argmax(prediction)
            confidence = prediction[predicted_idx]
            predicted_label = LABELS[predicted_idx]
            
            predictions.append(predicted_label)
            confidences.append(confidence)
            
            # Show what it sees
            color = (0, 255, 0) if predicted_label == target_letter else (0, 0, 255)
            cv2.putText(frame, f"Seeing: {predicted_label} ({confidence*100:.0f}%)",
                       (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Diagnosis', frame)
        cv2.waitKey(1)
    
    # Analyze results
    if not predictions:
        return {
            'target': target_letter,
            'success': False,
            'message': 'No hand detected',
            'details': {}
        }
    
    prediction_counts = Counter(predictions)
    most_common = prediction_counts.most_common(3)
    
    target_count = prediction_counts.get(target_letter, 0)
    target_percentage = (target_count / len(predictions)) * 100
    avg_confidence = np.mean(confidences)
    
    # Determine success
    is_recognized = target_percentage > 50  # Recognized if seen >50% of time
    
    result = {
        'target': target_letter,
        'success': is_recognized,
        'target_percentage': target_percentage,
        'total_frames': len(predictions),
        'avg_confidence': avg_confidence,
        'most_common': most_common,
        'all_predictions': prediction_counts
    }
    
    # Print summary
    print(f"\nResults for '{target_letter}':")
    print(f"  Total frames: {len(predictions)}")
    print(f"  Correctly recognized: {target_count}/{len(predictions)} ({target_percentage:.1f}%)")
    print(f"  Average confidence: {avg_confidence*100:.1f}%")
    print(f"\n  Top predictions:")
    for pred, count in most_common:
        pct = (count / len(predictions)) * 100
        status = "✓" if pred == target_letter else "✗"
        print(f"    {status} {pred}: {count} times ({pct:.1f}%)")
    
    if is_recognized:
        print(f"\n  ✓ SUCCESS: '{target_letter}' is recognized!")
    else:
        confused_with = most_common[0][0] if most_common else "unknown"
        print(f"\n  ✗ PROBLEM: '{target_letter}' confused with '{confused_with}'")
    
    return result

def main():
    print("\n" + "="*70)
    print("ASL LETTER RECOGNITION DIAGNOSIS")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model = load_model(MODEL_PATH)
    print("✓ Model loaded!")
    
    # Initialize
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Camera ready!")
    
    # Test mode selection
    print("\n" + "="*70)
    print("SELECT TEST MODE:")
    print("="*70)
    print("1. Test all 24 letters (takes ~2-3 minutes)")
    print("2. Test specific letters only")
    print("3. Quick test (problematic letters only)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    letters_to_test = []
    
    if choice == "1":
        letters_to_test = LABELS
        print("\n✓ Will test all 24 letters")
    elif choice == "2":
        custom = input("\nEnter letters to test (e.g., M N S E T): ").strip().upper().split()
        letters_to_test = [l for l in custom if l in LABELS]
        print(f"\n✓ Will test: {', '.join(letters_to_test)}")
    else:
        # Common problematic letters
        letters_to_test = ['M', 'N', 'S', 'E', 'T', 'U']
        print(f"\n✓ Will test common problem letters: {', '.join(letters_to_test)}")
    
    if not letters_to_test:
        print("✗ No valid letters to test!")
        return
    
    input("\nPress ENTER to start testing...")
    
    # Test each letter
    results = []
    for i, letter in enumerate(letters_to_test, 1):
        print(f"\n\n{'#'*70}")
        print(f"Letter {i}/{len(letters_to_test)}")
        print(f"{'#'*70}")
        
        result = test_letter_recognition(letter, model, hands, cap, mp_draw, mp_hands)
        results.append(result)
        
        time.sleep(1)  # Brief pause between letters
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    
    print("\n\n" + "="*70)
    print("DIAGNOSIS COMPLETE - SUMMARY")
    print("="*70)
    
    working = [r for r in results if r['success']]
    failing = [r for r in results if not r['success']]
    
    print(f"\nWorking Letters: {len(working)}/{len(results)}")
    print(f"Problematic Letters: {len(failing)}/{len(results)}")
    
    if working:
        print(f"\n✓ WORKING WELL:")
        for r in working:
            print(f"  {r['target']}: {r['target_percentage']:.0f}% recognition")
    
    if failing:
        print(f"\n✗ PROBLEMATIC:")
        for r in failing:
            confused = r['most_common'][0][0] if r['most_common'] else 'none'
            print(f"  {r['target']}: {r['target_percentage']:.0f}% recognition (confused with '{confused}')")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if len(failing) == 0:
        print("\n✓ All tested letters work perfectly! Great job!")
    elif len(failing) <= 3:
        print(f"\n⚠ {len(failing)} problematic letter(s):")
        print("\nQuick fixes:")
        print("  1. Re-collect data for these specific letters")
        print("  2. Check ASL reference - ensure correct hand shape")
        print("  3. Practice holding gesture more steadily")
    else:
        print(f"\n✗ {len(failing)} problematic letters - systematic issue:")
        print("\nLikely causes:")
        print("  1. Training data quality issues")
        print("  2. Hand orientation inconsistency")
        print("  3. Lighting/background mismatch")
        print("\nSuggested actions:")
        print("  1. Re-collect ALL data with consistent lighting")
        print("  2. Use ASL reference chart while collecting")
        print("  3. Have same person verify gesture correctness")
        print("  4. Increase samples per letter to 500")
    
    # Save report
    import json
    report_file = "diagnosis_report.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Detailed report saved to: {report_file}")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()