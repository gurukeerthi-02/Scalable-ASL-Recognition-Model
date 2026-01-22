# Software Requirements Specification (SRS) - Scalable ASL Recognition

## 1. Introduction
### 1.1 Purpose
This document provides a comprehensive description of the Scalable ASL Recognition system. It outlines the current state of development, functional and non-functional requirements, and the technical architecture of the project.

### 1.2 Project Overview
The project aims to facilitate communication for the Deaf and Hard-of-Hearing community (DHH) by providing real-time American Sign Language (ASL) recognition. It consists of a high-performance recognition backend and a professional-grade video conferencing platform (**ASL Meet**) built with Next.js.

---

## 2. Overall Description
### 2.1 System Architecture
The system follows a client-server architecture:
- **Client**: A web interface that captures video from the user's webcam and transmits frames to the server.
- **Server**: A Flask-based backend that processes frames using Mediapipe and Keras models, returning recognition results in real-time via WebSockets.

### 2.2 Features (Implemented to Date)
1. **Real-time Hand Tracking**: Uses Mediapipe to extract 21 3D hand landmarks.
2. **Hybrid Recognition Engine**:
   - **Static Mode (MLP)**: Detects letters like A, B, C, etc., when hand motion is minimal.
   - **Dynamic Mode (LSTM)**: Detects motion-based gestures like 'J', 'Z', and 'Hello' by analyzing sequences of 30 frames.
3. **Internal State Machine**: Automatically switches between static and dynamic recognition based on the calculated motion of the index finger tip.
4. **WebSocket Integration**: Low-latency communication for real-time video overlay and result display.
5. **Session Management**: Independent state tracking for multiple concurrent users via SocketIO sessions.
6. **Text-to-Speech (TTS)**: Integration of `pyttsx3` for vocalizing recognized gestures/sentences (available in the standalone engine).

### 2.3 User Classes and Characteristics
- **Signers**: Users who use ASL to communicate.
- **Developers**: Users who contribute to the dataset or refine the ML models.

---

## 3. Technical Stack
- **Languages**: Python 3.10.11, JavaScript.
- **Libraries/Frameworks**:
  - `Flask` & `Flask-SocketIO`: Web server and real-time networking.
  - `Mediapipe`: Hand landmark extraction.
  - `TensorFlow/Keras`: Deep learning model execution.
  - `OpenCV`: Image processing.
  - `NumPy`: Numerical calculations (feature engineering).
  - `pyttsx3`: Cross-platform Text-to-Speech library.
  - `Next.js`: React framework for the conferencing frontend.
  - `Supabase`: Database and authentication for the conferencing platform.
  - `Tailwind CSS`: Utility-first CSS framework for UI design.

---

## 4. Work Completed to Date
### 4.1 Data Collection Utilities (v1.0)
- `collect-motion-data.py`: Tool for capturing 30-frame temporal sequences for LSTM training.
- `data-collection.py`: Tool for capturing static landmarks for MLP training.

### 4.2 Model Development
- **Static Model**: Trained on 68 features (relative landmark distances + raw deltas).
- **Dynamic Model**: Trained on 63 features (raw X,Y,Z for 21 landmarks) across a 30-frame window.
- **Current Dynamic Labels**: `Hello`, `J`, `Z`, `no-gesture`.

### 4.3 Backend Integration
- `server.py`: SocketIO-based implementation for web-based video conferencing.
- `asl-engine.py`: Standalone `HybridASLEngine` class with integrated TTS for local/embedded use.
- Developed motion detection logic to gate static vs. dynamic recognition.
- Integrated `gesture_model.h5` and `z_j_motion_lstm.h5`.

### 4.4 Frontend Integration
- **ASL Meet**: A comprehensive Next.js application providing:
  - Secure room-based video conferencing.
  - Real-time ASL-to-text translation overlay.
  - User authentication and room management via Supabase.
- **Legacy Interface**: Simple `index.html` for rapid backend testing.

---

## 5. Functional Requirements
### 5.1 Recognition Requirements
- **FR1**: The system shall detect hand landmarks with at least 70% confidence.
- **FR2**: The system shall switch to DYNAMIC mode if index finger motion exceeds a threshold (0.010).
- **FR3**: The system shall output the predicted text and confidence score for each recognized gesture.
- **FR4**: The system shall accumulate recognized gestures into a continuous sentence.

### 5.2 Interface Requirements
- **IR1**: The web interface MUST display the live webcam feed with landmark overlays.
- **IR2**: The interface MUST provide a "Clear Sentence" button to reset the accumulated text.

---

## 6. Non-Functional Requirements
- **Performance**: Frame processing latency should be under 100ms for a smooth user experience.
- **Scalability**: The server should handle multiple concurrent WebSocket connections using `eventlet`.
- **Accuracy**: Recognition confidence for standard gestures should exceed 85%.

---

## 7. Next Steps / Future Work
- **Expansion of Vocabulary**: Add more dynamic words (e.g., "HELP", "THANK YOU", "PLEASE").
- **Improved UI/UX**: Enhance the `asl-meet` interface with better styling and notification systems.
- **Environment Robustness**: Improve recognition accuracy in low-light or cluttered backgrounds.
- **Mobile Support**: Optimize the model for delivery on mobile devices via TFLite.
