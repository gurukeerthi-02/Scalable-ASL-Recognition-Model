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
import tensorflow as tf
# Optimize for low-latency inference
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
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
STATIC_MODEL = load_model(os.path.join(BASE_DIR, "utils/models/static_model_person_split_v7.h5"))
DYNAMIC_MODEL = load_model(os.path.join(BASE_DIR, "utils/models/dynamic_model_final.h5"))
print("Models loaded successfully!")

# Optimized Inference Functions (XLA Compiled)
@tf.function(jit_compile=True)
def run_static_inference(model, feat):
    return model(feat, training=False)

@tf.function(jit_compile=True)
def run_dynamic_inference(model, seq):
    return model(seq, training=False)

print("Warm-up (XLA)...")
dummy_static = np.zeros((1, 68), dtype=np.float32)
dummy_dynamic = np.zeros((1, 30, 68), dtype=np.float32)
# Multi-pass warm-up to trigger JIT
for _ in range(3):
    run_static_inference(STATIC_MODEL, dummy_static)
    run_dynamic_inference(DYNAMIC_MODEL, dummy_dynamic)
print("Warm-up complete!")

STATIC_LABELS = sorted(os.listdir(os.path.join(BASE_DIR, "utils/dataset_merged")))
DYNAMIC_LABELS = ["BYE", "HELLO", "J", "NO", "YES", "Z"]

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
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,  # 0 is faster, 1 is more accurate
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.drawer = mp.solutions.drawing_utils

        # State machine
        self.mode = MODE_IDLE
        
        # Buffers
        self.motion_buffer = deque(maxlen=30)
        self.motion_buffer_np = np.zeros((30, 68), dtype=np.float32) # Pre-allocated for speed
        self.motion_history = deque(maxlen=3) # Faster response than 5
        self.collect_count = 0
        self.stable_frame_count = 0
        
        # Tracking
        self.prev_tip = None
        self.display_text = ""
        self.display_conf = 0.0
        self.last_gesture_time = 0
        self.result_start_time = 0

        # Thresholds (Optimized for responsiveness)
        self.STATIC_MAX_MOTION = 0.006
        self.DYNAMIC_MIN_MOTION = 0.012
        self.STATIC_STABLE_FRAMES = 5  # Reduced from 8 for faster detection
        
        self.CONF_STATIC = 0.70
        self.CONF_DYNAMIC = 0.70
        
        self.COOLDOWN_TIME = 0.2
        self.STATIC_HOLD_TIME = 0.8    # Reduced from 1.5 for faster flow
        self.DYNAMIC_HOLD_TIME = 1.2   # Reduced from 2.0

        self.frame_counter = 0
        
        print(f"[SESSION] New ASL session created")

    # ---------- FEATURE EXTRACTION (68 dims) ----------
    def extract_features(self, hand):
        """Extract 68-dimensional feature vector (shared by static and dynamic)"""
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

        return np.array(features, dtype=np.float32)


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

        # TIMING METRICS
        times = {
            "decode": 0,
            "mediapipe": 0,
            "velocity": 0,
            "inference": 0,
            "encode": 0
        }

        t_start = time.time()

        # Decode base64 frame
        encoded = frame_data.split(",")[1]
        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), np.uint8),
            cv2.IMREAD_COLOR
        )
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        times["decode"] = (time.time() - t_start) * 1000

        # Process with MediaPipe
        t_mp_start = time.time()
        result = self.hands.process(rgb)
        times["mediapipe"] = (time.time() - t_mp_start) * 1000
        lm_time = times["mediapipe"]

        # Check cooldown
        in_cooldown = (now - self.last_gesture_time) < self.COOLDOWN_TIME

        # =======================
        # HAND DETECTION & LOGIC
        # =======================
        t_logic_start = time.time()
        
        if not result.multi_hand_landmarks:
            # No hand detected: early exit to save CPU
            self.prev_tip = None
            self.motion_history.clear()
            
            # Reset state if we aren't displaying a result
            if self.mode != MODE_HOLD_RESULT:
                self.mode = MODE_IDLE
                self.stable_frame_count = 0
                self.collect_count = 0
                self.motion_buffer_np.fill(0)
                
            return {
                "text": self.display_text, # Keep showing result if in HOLD_RESULT mode
                "confidence": round(self.display_conf, 3),
                "mode": self.mode,
                "motion": 0.0,
                "server_timings": times,
                "hand_detected": False
            }

        # HAND DETECTED: Proceed with logic
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

            # Result Logic / State Machine
            
            # HOLD RESULT → Return to IDLE
            if self.mode == MODE_HOLD_RESULT:
                is_dynamic = self.display_text in DYNAMIC_LABELS
                hold_duration = self.DYNAMIC_HOLD_TIME if is_dynamic else self.STATIC_HOLD_TIME
                
                if now - self.result_start_time >= hold_duration:
                    self.mode = MODE_IDLE
                    self.display_text = ""
                    self.display_conf = 0.0
                    self.stable_frame_count = 0

            # IDLE → STATIC or DYNAMIC
            elif self.mode == MODE_IDLE and not in_cooldown:
                if motion < self.STATIC_MAX_MOTION:
                    self.stable_frame_count += 1
                    if self.stable_frame_count >= self.STATIC_STABLE_FRAMES:
                        # Stability reached: Attempt static recognition immediately
                        t_inf_start = time.time()
                        feat = self.extract_features(hand).reshape(1, -1).astype(np.float32)
                        pred = run_static_inference(STATIC_MODEL, feat).numpy()[0]
                        times["inference"] = (time.time() - t_inf_start) * 1000
                        
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
                            # Not enough confidence, but we are still still, so enter STATIC mode
                            self.mode = MODE_STATIC
                            self.stable_frame_count = 0
                            print("[MODE] Switched to STATIC")
                elif motion > self.DYNAMIC_MIN_MOTION:
                    self.mode = MODE_DYNAMIC_COLLECT
                    self.collect_count = 0
                    self.motion_buffer_np.fill(0)
                    self.display_text = ""
                    self.stable_frame_count = 0
                else:
                    self.stable_frame_count = 0

            # STATIC → DYNAMIC or recognize again if stable
            elif self.mode == MODE_STATIC:
                if motion > self.DYNAMIC_MIN_MOTION:
                    self.mode = MODE_DYNAMIC_COLLECT
                    self.collect_count = 0
                    self.motion_buffer_np.fill(0)
                    self.display_text = ""
                    self.stable_frame_count = 0
                elif motion > self.STATIC_MAX_MOTION * 1.5:
                    # Too much motion for static, go back to IDLE
                    self.mode = MODE_IDLE
                    self.stable_frame_count = 0
                elif motion < self.STATIC_MAX_MOTION:
                    self.stable_frame_count += 1
                    # Periodic re-check every 5 frames if we stay in static
                    if self.stable_frame_count >= self.STATIC_STABLE_FRAMES:
                        t_inf_start = time.time()
                        feat = self.extract_features(hand).reshape(1, -1).astype(np.float32)
                        pred = run_static_inference(STATIC_MODEL, feat).numpy()[0]
                        times["inference"] = (time.time() - t_inf_start) * 1000
                        
                        idx = np.argmax(pred)
                        conf = pred[idx]

                        if conf > self.CONF_STATIC:
                            self.display_text = STATIC_LABELS[idx]
                            self.display_conf = float(conf)
                            self.mode = MODE_HOLD_RESULT
                            self.result_start_time = now
                            self.last_gesture_time = now
                        
                        self.stable_frame_count = 0

            # DYNAMIC COLLECT → Process sequence
            elif self.mode == MODE_DYNAMIC_COLLECT:
                # Abort if motion stops for too long (e.g. user gave up or hand went out of frame)
                if motion < self.STATIC_MAX_MOTION and self.collect_count > 10:
                    print("[DYNAMIC] Aborted: Motion stopped during collection")
                    self.mode = MODE_IDLE
                    self.collect_count = 0
                    self.motion_buffer_np.fill(0)
                else:
                    feat = self.extract_features(hand)
                    self.motion_buffer_np[self.collect_count] = feat
                    self.collect_count += 1

                if self.collect_count == 30:
                    t_inf_start = time.time()
                    if 'DYNAMIC_MODEL' in globals():
                        seq = self.motion_buffer_np.reshape(1, 30, 68)
                        pred = run_dynamic_inference(DYNAMIC_MODEL, seq).numpy()[0]
                        idx = np.argmax(pred)
                        conf = pred[idx]
                    else:
                        idx = 0
                        conf = 0.0

                    times["inference"] = (time.time() - t_inf_start) * 1000

                    if conf > self.CONF_DYNAMIC:
                        self.display_text = DYNAMIC_LABELS[idx]
                        self.display_conf = float(conf)
                        self.mode = MODE_HOLD_RESULT
                        self.result_start_time = now
                        self.last_gesture_time = now
                        print(f"[DYNAMIC] Detected: {self.display_text} ({conf:.2f})")
                    else:
                        print(f"[DYNAMIC] Low confidence: {DYNAMIC_LABELS[idx]} ({conf:.2f})")
                        self.mode = MODE_IDLE
                        self.display_text = ""
                    
                    self.collect_count = 0
                    self.motion_buffer_np.fill(0)
                    self.stable_frame_count = 0

        else:
            # No hand detected
            self.prev_tip = None
            self.motion_history.clear()
            
            # Only reset if not showing result
            if self.mode != MODE_HOLD_RESULT:
                self.mode = MODE_IDLE
                self.stable_frame_count = 0
                self.collect_count = 0
                self.motion_buffer_np.fill(0)
        
        times["velocity"] = (time.time() - t_logic_start) * 1000 - times["inference"]

        # Performance logging
        if self.frame_counter % 30 == 0:
            mode_names = {
                MODE_IDLE: "IDLE",
                MODE_STATIC: "STATIC",
                MODE_DYNAMIC_COLLECT: "DYNAMIC",
                MODE_HOLD_RESULT: "RESULT"
            }

        # Optimization: We NO LONGER encode and send the frame back.
        # This saves ~15-30ms of CPU and 100KB+ per frame of network bandwidth.
        # self.drawer.draw_landmarks is still called for internal logic if needed, 
        # but the frame isn't returned.
        
        return {
            "text": self.display_text,
            "confidence": round(self.display_conf, 3),
            "mode": self.mode,
            "motion": round(self.get_smoothed_motion(), 5),
            "buffer_size": self.collect_count,
            "stable_count": self.stable_frame_count,
            "server_timings": times,
            "hand_detected": bool(result.multi_hand_landmarks)
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
            
        result = sessions[request.sid].process_frame(data["frame"])
        # We only emit the metadata result, not the 'frame'
        emit("frame_result", {
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