# Scalable ASL Recognition System

This project provides a comprehensive solution for real-time American Sign Language (ASL) recognition integrated into a peer-to-peer video calling application. The system leverages Mediapipe for high-fidelity hand landmark extraction and employs a hybrid deep learning architecture to recognize both static and dynamic gestures with minimal latency.

## Core Features

- **Hybrid Recognition Model**: Combined Multi-Layer Perceptron (MLP) for static signs and Long Short-Term Memory (LSTM) for dynamic sequences.
- **Low-Latency Backend**: Optimized Python server using XLA (Accelerated Linear Algebra) compiled inference and Socket.IO for real-time metadata exchange.
- **WebRTC Video Conferencing**: Secure, peer-to-peer video calling supporting up to four simultaneous participants.
- **Text-to-Speech Integration**: Automated conversion of recognized signs into audible speech for remote participants.
- **State Machine Logic**: Robust gesture detection system that filters motion noise and handles transitions between idle, collection, and recognition states.
- **Performance Monitoring**: Integrated end-to-end latency tracking across capture, network, landmarking, and inference stages.
- **Portable Data Collection**: Standalone utilities for capturing specialized datasets from multiple users to improve model generalization.

## Project Structure

- **root**: Contains the ASL recognition backend, model assets, and data collection utilities.
- **asl-meet**: Next.js frontend application for WebRTC video calls and ASL visualization.
- **utils**: Collection of training scripts, performance measurement tools, and dataset management utilities.

## Setup Instructions

### 1. Backend Recognition Server

The backend requires Python 3.10+ and several deep learning libraries.

1. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your trained model files are located in the `utils/models/` directory:
   - Static Model: `utils/models/static_model_person_split_v7.h5`
   - Dynamic Model: `utils/models/dynamic_model_final.h5`

### 2. Video Call Frontend

The frontend is built with Next.js 13+ and requires Node.js 18+.

1. Navigate to the frontend directory:
   ```bash
   cd asl-meet
   ```

2. Install the Node dependencies:
   ```bash
   npm install
   ```

3. Configure environment variables:
   Copy `.env.local.example` to `.env.local` and provide your Supabase credentials:
   ```bash
   cp .env.local.example .env.local
   ```
   *Required variables*: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`.

## Data Collection

The project includes portable scripts to facilitate the collection of new ASL data for model refinement.

### Static Data Collection
Captures 68-dimensional hand landmark vectors for 24 static letters.
```bash
python utils/improved/collect-data-portable.py
```
*   Follow the prompts to enter your name.
*   The script will guide you through each letter, saving 200 samples per gesture.
*   Data is automatically packaged into a ZIP file upon completion.

### Dynamic Data Collection
Captures temporal sequences (30 frames) for dynamic gestures like 'HELLO', 'BYE', 'YES', and 'NO'.
```bash
python utils/improved/dynamic/collect-data-portable.py
```
*   Follow the on-screen guide for motion patterns.
*   Hold SPACE to record a gesture sequence.
*   Sequences are saved as `.npy` files and zipped for sharing.

## Running the Project

To run the complete system, you must start three parallel services:

### Step 1: Start the ASL Recognition Backend
This service performs the MediaPipe landmarking and neural network inference.
```bash
python server.py
```
Default URL: `http://localhost:5000`

### Step 2: Start the WebRTC Signaling Server
This service facilitates peer connection negotiation between callers.
```bash
cd asl-meet
npm run socket-server
```
Default URL: `http://localhost:3001`

### Step 3: Start the Next.js Application
The user interface for creating and joining video call rooms.
```bash
cd asl-meet
npm run dev
```
Default URL: `http://localhost:3000`

## Technical Specifications

- **Feature Extraction**: 21 landmarks (x, y, z) + 4 fingertip-to-wrist distances + 1 thumb-to-index distance = 68 total features.
- **Inference Optimization**: Models are loaded with TensorFlow XLA JIT compilation to reduce per-frame inference time to <2ms.
- **Congestion Control**: Frame transmission is throttled to 10 FPS with an adaptive buffer management system to prevent network latency spikes.
- **Model Training**: Systems employ a person-based split strategy to ensure the models generalize across different users and physical backgrounds.

## License

MIT License
