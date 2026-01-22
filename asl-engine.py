import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
import pyttsx3
from collections import deque
from tensorflow.keras.models import load_model


class HybridASLEngine:
    def __init__(self):
        # =======================
        # LOAD MODELS
        # =======================
        self.static_model = load_model("./models/gesture_model.h5")
        self.dynamic_model = load_model("./models/z_j_motion_lstm.h5")

        self.static_labels = sorted(os.listdir("dataset"))
        self.dynamic_labels = ["No Gesture", "Z", "J"]

        # =======================
        # MEDIAPIPE
        # =======================
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # =======================
        # MODES
        # =======================
        self.MODE_IDLE = 0
        self.MODE_STATIC = 1
        self.MODE_DYNAMIC_COLLECT = 2
        self.MODE_HOLD_RESULT = 3
        self.mode = self.MODE_IDLE

        # =======================
        # PARAMETERS
        # =======================
        self.SEQUENCE_LENGTH = 30

        self.STATIC_MAX_MOTION = 0.004
        self.DYNAMIC_MIN_MOTION = 0.010
        self.MOTION_SMOOTHING_WINDOW = 5

        self.CONF_STATIC = 0.85
        self.CONF_DYNAMIC = 0.75

        self.STATIC_STABLE_FRAMES = 10
        self.DYNAMIC_HOLD_TIME = 2.0
        self.RESULT_DISPLAY_TIME = 1.5
        self.COOLDOWN_TIME = 0.3

        # =======================
        # STATE
        # =======================
        self.motion_buffer = deque(maxlen=self.SEQUENCE_LENGTH)
        self.feature_history = deque(maxlen=10)
        self.static_pred_buffer = deque(maxlen=10)
        self.motion_history = deque(maxlen=self.MOTION_SMOOTHING_WINDOW)

        self.prev_tip = None
        self.display_text = ""
        self.display_conf = 0.0
        self.result_start_time = 0
        self.last_gesture_time = 0
        self.current_sentence = ""
        self.stable_frame_count = 0
        self.selected_voice_index = 0

    # =======================
    # FEATURE EXTRACTION
    # =======================
    def extract_static_features(self, hand):
        def dist(a, b):
            return np.linalg.norm(
                np.array([a.x, a.y]) - np.array([b.x, b.y])
            )

        lm = hand.landmark
        wrist = lm[0]
        features = []

        for p in lm:
            features.extend([
                p.x - wrist.x,
                p.y - wrist.y,
                p.z - wrist.z
            ])

        features.extend([
            dist(lm[8], wrist),
            dist(lm[12], wrist),
            dist(lm[16], wrist),
            dist(lm[20], wrist)
        ])

        features.append(dist(lm[4], lm[8]))

        return np.array(features).reshape(1, -1)

    def extract_dynamic_features(self, hand):
        features = []
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features)

    def get_smoothed_motion(self):
        if len(self.motion_history) == 0:
            return 0.0
        return np.mean(self.motion_history)

    # =======================
    # TEXT TO SPEECH
    # =======================
    def speak_text(self, text):
        def _speak():
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                if voices:
                    engine.setProperty(
                        'voice',
                        voices[self.selected_voice_index % len(voices)].id
                    )
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print("TTS Error:", e)

        if text.strip():
            threading.Thread(target=_speak, daemon=True).start()

    # =======================
    # MAIN FRAME PROCESSOR
    # =======================
    def process_frame(self, frame):
        now = time.time()

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        in_cooldown = (now - self.last_gesture_time) < self.COOLDOWN_TIME

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            tip = hand.landmark[8]
            tip_pos = np.array([tip.x, tip.y])

            instant_motion = 0.0
            if self.prev_tip is not None:
                instant_motion = np.linalg.norm(tip_pos - self.prev_tip)

            self.prev_tip = tip_pos
            self.motion_history.append(instant_motion)
            motion = self.get_smoothed_motion()

            curr_dyn_features = self.extract_dynamic_features(hand)
            self.feature_history.append(curr_dyn_features)

            # -----------------------
            # STATE MACHINE
            # -----------------------
            if self.mode == self.MODE_IDLE and not in_cooldown:
                if motion < self.STATIC_MAX_MOTION:
                    self.stable_frame_count += 1
                    if self.stable_frame_count >= self.STATIC_STABLE_FRAMES:
                        self.mode = self.MODE_STATIC
                        self.stable_frame_count = 0
                        self.static_pred_buffer.clear()
                else:
                    self.stable_frame_count = 0

            elif self.mode == self.MODE_STATIC:
                if motion > self.DYNAMIC_MIN_MOTION:
                    self.mode = self.MODE_DYNAMIC_COLLECT
                    self.motion_buffer.clear()
                    self.motion_buffer.extend(self.feature_history)
                    self.display_text = ""
                    self.stable_frame_count = 0

                elif motion < self.STATIC_MAX_MOTION:
                    self.stable_frame_count += 1
                    if self.stable_frame_count >= self.STATIC_STABLE_FRAMES:
                        static_feat = self.extract_static_features(hand)
                        pred = self.static_model.predict(static_feat, verbose=0)[0]
                        idx = np.argmax(pred)
                        conf = pred[idx]

                        if conf > self.CONF_STATIC:
                            self.static_pred_buffer.append(idx)
                            if len(self.static_pred_buffer) == self.static_pred_buffer.maxlen:
                                most_common = max(
                                    set(self.static_pred_buffer),
                                    key=self.static_pred_buffer.count
                                )
                                if self.static_pred_buffer.count(most_common) >= 8:
                                    self.display_text = self.static_labels[most_common]
                                    self.display_conf = conf
                                    self.current_sentence += self.display_text
                                    self.result_start_time = now
                                    self.last_gesture_time = now
                                    self.mode = self.MODE_HOLD_RESULT
                                    self.static_pred_buffer.clear()
                                    self.stable_frame_count = 0
                        else:
                            self.static_pred_buffer.append(-1)
                else:
                    self.stable_frame_count = max(0, self.stable_frame_count - 2)
                    self.static_pred_buffer.clear()

            elif self.mode == self.MODE_DYNAMIC_COLLECT:
                self.motion_buffer.append(curr_dyn_features)

                if len(self.motion_buffer) == self.SEQUENCE_LENGTH:
                    seq = np.array(self.motion_buffer).reshape(1, self.SEQUENCE_LENGTH, 63)
                    pred = self.dynamic_model.predict(seq, verbose=0)[0]
                    idx = np.argmax(pred)
                    conf = pred[idx]

                    if conf > self.CONF_DYNAMIC and self.dynamic_labels[idx] != "No Gesture":
                        self.display_text = self.dynamic_labels[idx]
                        self.display_conf = conf
                        self.current_sentence += self.display_text
                        self.result_start_time = now
                        self.last_gesture_time = now
                        self.mode = self.MODE_HOLD_RESULT

                    self.motion_buffer.clear()
                    self.stable_frame_count = 0

            elif self.mode == self.MODE_HOLD_RESULT:
                hold_time = (
                    self.DYNAMIC_HOLD_TIME
                    if self.display_text in ["Z", "J"]
                    else self.RESULT_DISPLAY_TIME
                )
                if now - self.result_start_time >= hold_time:
                    self.mode = self.MODE_IDLE
                    self.display_text = ""
                    self.display_conf = 0.0
                    self.stable_frame_count = 0

        else:
            self.prev_tip = None
            self.motion_history.clear()
            if self.mode != self.MODE_HOLD_RESULT:
                self.mode = self.MODE_IDLE
                self.stable_frame_count = 0

        return {
            "letter": self.display_text,
            "confidence": round(self.display_conf, 3),
            "sentence": self.current_sentence,
            "mode": self.mode
        }
