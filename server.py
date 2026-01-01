import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import os
from collections import deque
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

# =======================
# GLOBAL CONFIG & MODELS
# =======================
# template_folder='.' allows index.html to be in the root directory
app = Flask(__name__, template_folder='.')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Load models once to save memory
print("Loading models...")
try:
    STATIC_MODEL = load_model("./models/gesture_model.h5")
    DYNAMIC_MODEL = load_model("./models/z_j_motion_lstm.h5")
    print("Models loaded.")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Ensure 'gesture_model.h5' and 'z_j_motion_lstm.h5' are in the 'models' folder.")

# Load labels
try:
    STATIC_LABELS = sorted(os.listdir("dataset"))
except:
    STATIC_LABELS = [] # Fallback if dataset folder missing
    
DYNAMIC_LABELS = ["No Gesture", "Z", "J"]

# =======================
# ASL RECOGNIZER CLASS
# =======================
class ASLSession:
    def __init__(self):
        # Mediapipe instance per session
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # State Variables
        self.mode = 0 # MODE_IDLE
        self.motion_buffer = deque(maxlen=30)
        self.feature_history = deque(maxlen=10)
        self.static_pred_buffer = deque(maxlen=10)
        self.motion_history = deque(maxlen=5)
        self.prev_tip = None
        
        self.display_text = ""
        self.display_conf = 0.0
        self.result_start_time = 0
        self.last_gesture_time = 0
        self.current_sentence = ""
        self.stable_frame_count = 0
        
        # Constants (Matched to hybrid-realtime.py)
        self.STATIC_MAX_MOTION = 0.004
        self.DYNAMIC_MIN_MOTION = 0.010
        self.CONF_STATIC = 0.85
        self.CONF_DYNAMIC = 0.75
        self.STATIC_STABLE_FRAMES = 10

    def extract_static_features(self, hand):
        def dist(a, b):
            return np.linalg.norm(np.array([a.x, a.y]) - np.array([b.x, b.y]))
        lm = hand.landmark
        wrist = lm[0]
        features = []
        # Relative 3D positions
        for p in lm:
            features.extend([p.x - wrist.x, p.y - wrist.y, p.z - wrist.z])
        # Finger tip distances
        features.extend([dist(lm[8], wrist), dist(lm[12], wrist), 
                         dist(lm[16], wrist), dist(lm[20], wrist)])
        # Thumb-index distance
        features.append(dist(lm[4], lm[8]))
        return np.array(features).reshape(1, -1)

    def extract_dynamic_features(self, hand):
        features = []
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features)

    def process_frame(self, frame_data):
        # Decode Base64 image
        try:
            encoded_data = frame_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except:
            return None, None
        
        # Flip for consistency
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = self.hands.process(rgb)
        now = time.time()
        
        # Cooldown check
        in_cooldown = (now - self.last_gesture_time) < 0.3
        speak_trigger = None

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

            # Motion Calculation
            tip = hand.landmark[8]
            tip_pos = np.array([tip.x, tip.y])
            instant_motion = 0.0
            if self.prev_tip is not None:
                instant_motion = np.linalg.norm(tip_pos - self.prev_tip)
            self.prev_tip = tip_pos
            self.motion_history.append(instant_motion)
            motion = np.mean(self.motion_history) if self.motion_history else 0.0

            curr_dyn_features = self.extract_dynamic_features(hand)
            self.feature_history.append(curr_dyn_features)

            # --- STATE MACHINE ---
            MODE_IDLE = 0
            MODE_STATIC = 1
            MODE_DYNAMIC_COLLECT = 2
            MODE_HOLD_RESULT = 3

            if self.mode == MODE_IDLE and not in_cooldown:
                if motion < self.STATIC_MAX_MOTION:
                    self.stable_frame_count += 1
                    if self.stable_frame_count >= self.STATIC_STABLE_FRAMES:
                        self.mode = MODE_STATIC
                        self.stable_frame_count = 0
                        self.static_pred_buffer.clear()
                else:
                    self.stable_frame_count = 0

            elif self.mode == MODE_STATIC:
                if motion > self.DYNAMIC_MIN_MOTION:
                    self.mode = MODE_DYNAMIC_COLLECT
                    self.motion_buffer.clear()
                    self.motion_buffer.extend(self.feature_history)
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
                            self.static_pred_buffer.append(idx)
                            if len(self.static_pred_buffer) == self.static_pred_buffer.maxlen:
                                most_common = max(set(self.static_pred_buffer), key=self.static_pred_buffer.count)
                                if most_common != -1 and self.static_pred_buffer.count(most_common) >= 8:
                                    self.display_text = STATIC_LABELS[most_common]
                                    self.display_conf = pred[most_common]
                                    self.mode = MODE_HOLD_RESULT
                                    self.current_sentence += self.display_text
                                    self.result_start_time = now
                                    self.last_gesture_time = now
                                    self.stable_frame_count = 0
                                    self.static_pred_buffer.clear()
                        else:
                            self.static_pred_buffer.append(-1)
                else:
                    self.stable_frame_count = max(0, self.stable_frame_count - 2)
                    self.static_pred_buffer.clear()

            elif self.mode == MODE_DYNAMIC_COLLECT:
                self.motion_buffer.append(curr_dyn_features)
                if len(self.motion_buffer) == 30:
                    seq = np.array(self.motion_buffer).reshape(1, 30, 63)
                    pred = DYNAMIC_MODEL.predict(seq, verbose=0)[0]
                    idx = np.argmax(pred)
                    conf = pred[idx]

                    if conf > self.CONF_DYNAMIC and DYNAMIC_LABELS[idx] != "No Gesture":
                        self.display_text = DYNAMIC_LABELS[idx]
                        self.display_conf = conf
                        self.mode = MODE_HOLD_RESULT
                        self.current_sentence += self.display_text
                        self.result_start_time = now
                        self.last_gesture_time = now
                    else:
                        self.motion_buffer.clear()
                    
                    self.motion_buffer.clear()
                    self.stable_frame_count = 0

            elif self.mode == MODE_HOLD_RESULT:
                hold_time = 2.0 if "Z" in self.display_text or "J" in self.display_text else 1.5
                if now - self.result_start_time >= hold_time:
                    self.mode = MODE_IDLE
                    self.display_text = ""
                    self.display_conf = 0.0
                    self.stable_frame_count = 0

        else:
            self.prev_tip = None
            self.motion_history.clear()
            if self.mode != 3: # MODE_HOLD_RESULT
                self.mode = 0

        # --- DRAWING OVERLAY ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        
        if self.display_text:
            text_size = cv2.getTextSize(self.display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (w - text_size[0]) // 2
            cv2.rectangle(overlay, (text_x - 10, 140), (text_x + text_size[0] + 10, 200), (0, 0, 0), -1)

        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Status Text
        mode_str = ["IDLE", "STATIC", "DYNAMIC", "RESULT"][self.mode]
        cv2.putText(frame, f"Mode: {mode_str}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Sentence: {self.current_sentence}|", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if self.display_text:
            text_size = cv2.getTextSize(self.display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, self.display_text, (text_x, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Encode back to jpg
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        return jpg_as_text, speak_trigger

    def handle_command(self, cmd):
        if cmd == 'space':
            self.current_sentence += " "
        elif cmd == 'backspace':
            self.current_sentence = self.current_sentence[:-1]
        elif cmd == 'clear':
            self.current_sentence = ""
        elif cmd == 'enter':
            return self.current_sentence
        return None

# =======================
# SERVER LOGIC
# =======================
sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    sessions[request.sid] = ASLSession()

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    if request.sid in sessions:
        del sessions[request.sid]

@socketio.on('process_frame')
def handle_frame(data):
    if request.sid in sessions:
        session = sessions[request.sid]
        processed_image, speak_text = session.process_frame(data['image'])
        if processed_image:
            emit('response_frame', {'image': processed_image})

@socketio.on('command')
def handle_command(data):
    if request.sid in sessions:
        session = sessions[request.sid]
        text_to_speak = session.handle_command(data['cmd'])
        if text_to_speak:
            emit('speak', {'text': text_to_speak})

if __name__ == '__main__':
    # Run on 0.0.0.0 to make it accessible on the network
    print("Server starting on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000)