import eventlet
eventlet.monkey_patch()

from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
from collections import deque
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

# =======================
# GLOBAL CONFIG
# =======================
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading models...")
STATIC_MODEL = load_model(os.path.join(BASE_DIR, "models", "gesture_model.h5"))
DYNAMIC_MODEL = load_model(os.path.join(BASE_DIR, "models", "z_j_motion_lstm.h5"))
print("Models loaded successfully!")

STATIC_LABELS = sorted(os.listdir(os.path.join(BASE_DIR, "dataset")))
DYNAMIC_LABELS = ["Hello", "J", "Z"]  # Matches training order: Hello=0, J=1, Z=2

# =======================
# MODES
# =======================
MODE_IDLE = 0
MODE_STATIC = 1
MODE_DYNAMIC_COLLECT = 2
MODE_HOLD_RESULT = 3

# =======================
# ASL SESSION
# =======================
class ASLSession:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.drawer = mp.solutions.drawing_utils

        # State machine
        self.mode = MODE_IDLE
        
        # Buffers
        self.motion_buffer = deque(maxlen=30)
        self.motion_history = deque(maxlen=5)
        self.stable_frame_count = 0
        
        # Tracking
        self.prev_tip = None
        self.display_text = ""
        self.display_conf = 0.0
        self.last_gesture_time = 0
        self.result_start_time = 0

        # Thresholds (aligned with desktop version)
        self.STATIC_MAX_MOTION = 0.005
        self.DYNAMIC_MIN_MOTION = 0.012
        self.STATIC_STABLE_FRAMES = 8
        
        self.CONF_STATIC = 0.80
        self.CONF_DYNAMIC = 0.75
        
        self.COOLDOWN_TIME = 0.3
        self.STATIC_HOLD_TIME = 1.5
        self.DYNAMIC_HOLD_TIME = 2.0

        self.frame_counter = 0
        
        print(f"[SESSION] New ASL session created")

    # ---------- STATIC FEATURES (68 dims) ----------
    def extract_static_features(self, hand):
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

    # ---------- DYNAMIC FEATURES (63 dims) ----------
    def extract_dynamic_features(self, hand):
        """Extract 63-dimensional feature vector for dynamic gestures"""
        features = []
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features)

    # ---------- MOTION SMOOTHING ----------
    def get_smoothed_motion(self):
        """Calculate smoothed motion value to reduce jitter"""
        if len(self.motion_history) == 0:
            return 0.0
        return np.mean(self.motion_history)

    # ---------- PROCESS FRAME ----------
    def process_frame(self, frame_data):
        self.frame_counter += 1
        now = time.time()

        # Decode base64 frame
        encoded = frame_data.split(",")[1]
        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), np.uint8),
            cv2.IMREAD_COLOR
        )
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        t0 = time.time()
        result = self.hands.process(rgb)
        lm_time = (time.time() - t0) * 1000

        # Check cooldown
        in_cooldown = (now - self.last_gesture_time) < self.COOLDOWN_TIME

        # =======================
        # HAND DETECTION
        # =======================
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(
                frame, 
                hand, 
                mp.solutions.hands.HAND_CONNECTIONS
            )

            # Calculate motion
            tip = hand.landmark[8]
            tip_pos = np.array([tip.x, tip.y])

            instant_motion = 0.0
            if self.prev_tip is not None:
                instant_motion = np.linalg.norm(tip_pos - self.prev_tip)
            
            self.prev_tip = tip_pos
            self.motion_history.append(instant_motion)
            motion = self.get_smoothed_motion()

            # =======================
            # STATE MACHINE LOGIC
            # =======================
            
            # HOLD RESULT → Return to IDLE
            if self.mode == MODE_HOLD_RESULT:
                is_dynamic = "Z" in self.display_text or "J" in self.display_text
                hold_duration = self.DYNAMIC_HOLD_TIME if is_dynamic else self.STATIC_HOLD_TIME
                
                if now - self.result_start_time >= hold_duration:
                    self.mode = MODE_IDLE
                    self.display_text = ""
                    self.display_conf = 0.0
                    self.stable_frame_count = 0

            # IDLE → STATIC (hand stable)
            elif self.mode == MODE_IDLE and not in_cooldown:
                if motion < self.STATIC_MAX_MOTION:
                    self.stable_frame_count += 1
                    if self.stable_frame_count >= self.STATIC_STABLE_FRAMES:
                        self.mode = MODE_STATIC
                        self.stable_frame_count = 0
                else:
                    self.stable_frame_count = 0

            # STATIC → DYNAMIC or detect static
            elif self.mode == MODE_STATIC:
                if motion > self.DYNAMIC_MIN_MOTION:
                    self.mode = MODE_DYNAMIC_COLLECT
                    self.motion_buffer.clear()
                    self.display_text = ""
                    self.stable_frame_count = 0
                    
                elif motion < self.STATIC_MAX_MOTION:
                    self.stable_frame_count += 1
                    
                    if self.stable_frame_count >= self.STATIC_STABLE_FRAMES:
                        static_feat = self.extract_static_features(hand)
                        pred = STATIC_MODEL.predict(static_feat, verbose=0)[0]
                        idx = np.argmax(pred)
                        conf = pred[idx]

                        if conf > self.CONF_STATIC:
                            self.display_text = STATIC_LABELS[idx]
                            self.display_conf = float(conf)
                            self.mode = MODE_HOLD_RESULT
                            self.result_start_time = now
                            self.last_gesture_time = now
                            self.stable_frame_count = 0
                            
                            print(f"[STATIC] Detected: {self.display_text} ({conf:.2f})")
                else:
                    self.stable_frame_count = max(0, self.stable_frame_count - 2)

            # DYNAMIC COLLECT → Process sequence
            elif self.mode == MODE_DYNAMIC_COLLECT:
                self.motion_buffer.append(self.extract_dynamic_features(hand))

                if len(self.motion_buffer) == 30:
                    seq = np.array(self.motion_buffer)
                    # 1. Z-score (Match training)
                    seq = (seq - np.mean(seq, axis=0)) / (np.std(seq, axis=0) + 1e-6)
                    # 2. Delta (Match training)
                    delta = np.diff(seq, axis=0)
                    delta = np.vstack([np.zeros((1, 63)), delta])
                    # 3. Concatenate (Match training)
                    seq = np.concatenate([seq, delta], axis=1).reshape(1, 30, 126)

                    pred = DYNAMIC_MODEL.predict(seq, verbose=0)[0]
                    idx = np.argmax(pred)
                    conf = pred[idx]

                    if conf > self.CONF_DYNAMIC:
                        self.display_text = DYNAMIC_LABELS[idx]
                        self.display_conf = float(conf)
                        self.mode = MODE_HOLD_RESULT
                        self.result_start_time = now
                        self.last_gesture_time = now
                        print(f"[DYNAMIC] Detected: {self.display_text} ({conf:.2f})")
                    else:
                        self.mode = MODE_IDLE
                        self.display_text = ""
                    
                    self.motion_buffer.clear()
                    self.stable_frame_count = 0

        else:
            # No hand detected
            self.prev_tip = None
            self.motion_history.clear()
            
            # Only reset if not showing result
            if self.mode != MODE_HOLD_RESULT:
                self.mode = MODE_IDLE
                self.stable_frame_count = 0

        # Performance logging
        if self.frame_counter % 30 == 0:
            mode_names = {
                MODE_IDLE: "IDLE",
                MODE_STATIC: "STATIC",
                MODE_DYNAMIC_COLLECT: "DYNAMIC",
                MODE_HOLD_RESULT: "RESULT"
            }
            print(f"[PERF] Landmark: {lm_time:.1f}ms | Mode: {mode_names[self.mode]} | Buffer: {len(self.motion_buffer)}/30")

        # Encode frame
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        encoded_frame = base64.b64encode(buf).decode()
        
        return encoded_frame, {
            "text": self.display_text,
            "confidence": round(self.display_conf, 3),
            "mode": self.mode,
            "motion": round(self.get_smoothed_motion(), 5),
            "buffer_size": len(self.motion_buffer),
            "stable_count": self.stable_frame_count
        }

# =======================
# SOCKET.IO
# =======================
sessions = {}

@socketio.on("connect")
def connect(auth=None):
    print(f"[SOCKET] Client connected: {request.sid}")
    sessions[request.sid] = ASLSession()
    emit("connection_status", {"status": "connected", "message": "ASL Recognition Ready"})

@socketio.on("disconnect")
def disconnect():
    print(f"[SOCKET] Client disconnected: {request.sid}")
    session = sessions.pop(request.sid, None)
    if session:
        session.hands.close()

@socketio.on("process_frame")
def handle_frame(data):
    try:
        if request.sid not in sessions:
            emit("error", {"message": "Session not found"})
            return
            
        frame, result = sessions[request.sid].process_frame(data["frame"])
        emit("frame_result", {
            "frame": f"data:image/jpeg;base64,{frame}",
            "result": result
        })
    except Exception as e:
        print(f"[ERROR] Frame processing failed: {str(e)}")
        emit("error", {"message": str(e)})

@socketio.on("reset_session")
def reset_session():
    """Allow client to reset session state"""
    if request.sid in sessions:
        sessions[request.sid] = ASLSession()
        emit("connection_status", {"status": "reset", "message": "Session reset"})
        print(f"[SOCKET] Session reset: {request.sid}")

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    print("=" * 60)
    print("ASL RECOGNITION SERVER")
    print("=" * 60)
    print(f"Static Labels: {STATIC_LABELS}")
    print(f"Dynamic Labels: {DYNAMIC_LABELS}")
    print("Server starting on http://0.0.0.0:5000")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)