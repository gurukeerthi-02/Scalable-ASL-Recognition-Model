# Deployment Guide: Scalable ASL Recognition

This guide outlines the steps to deploy the complete system, consisting of the ASL Recognition Backend, the Next.js Frontend, and the Socket.IO Signaling Server.

## Architecture Overview

1.  **AI Backend** (`/server.py`): Processes video frames and returns gesture metadata.
2.  **Signaling Server** (`/asl-meet/server/socket-server.js`): Handles WebRTC peer negotiation.
3.  **Frontend** (`/asl-meet/`): User interface for video calls and real-time visualization.

---

## 1. Deploying the AI Backend (Python/Flask)

The backend is resource-intensive due to TensorFlow and MediaPipe. We recommend **Railway**, **Render**, or **DigitalOcean App Platform**.

### Option: Containerized Deployment (Recommended)
The root directory contains a `Dockerfile`. You can use this to deploy to any cloud provider that supports Docker.

1.  **Platform Setup**:
    - Connect your GitHub repository to the platform.
    - Set the **Root Directory** to `/`.
    - The platform should automatically detect the `Dockerfile`.

2.  **Resources**:
    - Minimum **1GB RAM** (2GB recommended) for stable TensorFlow performance.
    - Set the deployment port to `5000`.

---

## 2. Deploying the Signaling Server (Node.js)

The signaling server manages WebRTC connections. It is a lightweight Node.js script.

1.  **Platform**: **Railway**, **Render**, or **Heroku**.
2.  **Configuration**:
    - **Root Directory**: `asl-meet/`
    - **Build Command**: `npm install`
    - **Start Command**: `node server/socket-server.js`
    - **Environment Variables**:
      - `PORT`: `3001` (or the platform's default port).

---

## 3. Deploying the Frontend (Next.js)

The frontend is best deployed to **Vercel** or **Netlify**.

1.  **Platform**: Connect your GitHub repository.
2.  **Build Settings**:
    - **Framework Preset**: `Next.js`
    - **Root Directory**: `asl-meet/`
    - **Build Command**: `npm run build`
    - **Install Command**: `npm install`
3.  **Environment Variables**:
    You MUST configure these on the deployment platform dashboard:
    - `NEXT_PUBLIC_SUPABASE_URL`: Your Supabase Project URL.
    - `NEXT_PUBLIC_SUPABASE_ANON_KEY`: Your Supabase API Key.
    - `NEXT_PUBLIC_SOCKET_URL`: URL of your deployed *Signaling Server* (e.g., `https://your-signaling.up.railway.app`).
    - `NEXT_PUBLIC_ASL_BACKEND_URL`: URL of your deployed *AI Backend* (e.g., `https://your-backend.up.railway.app`).

---

## 4. Production Requirements

### HTTPS Required
- All components **must** run over HTTPS.
- WebRTC `getUserMedia` and browser Text-to-Speech (TTS) will **fail** on insecure `http` connections (except localhost).

### Supabase Setup
- Ensure **Row Level Security (RLS)** is enabled on your `rooms`, `participants`, and `call_logs` tables.

### Performance Checklist
- **Cold Starts**: If using a serverless-style platform (like Render's free tier), the AI Backend may take 30+ seconds to "wake up" and load the models.
- **WebSocket (WSS)**: Ensure the frontend is connecting via `wss://` (secure WebSocket) for production.
