import socketio
import cv2
import base64
import time
import numpy as np
import threading
from collections import deque
import sys

# ==========================================
# CONFIGURATION
# ==========================================
SERVER_URL = "http://localhost:5000"
CAP_WIDTH = 640
CAP_HEIGHT = 480
USE_WEBCAM = True

# ==========================================
# STATE
# ==========================================
sio = socketio.Client()
packet_timestamps = deque()
running = True
latest_result = {}
lock = threading.Lock()

# METRICS STORAGE
metrics = {
    "capture_encode": [],
    "network": [],
    "mediapipe": [],
    "velocity": [],
    "inference": [],
    "response_process": [],
    "total_rtt": []
}

# ==========================================
# SOCKET EVENTS
# ==========================================
@sio.event
def connect():
    print(f"[CLIENT] Connected to server at {SERVER_URL}")

@sio.event
def disconnect():
    print("[CLIENT] Disconnected from server")

@sio.on('frame_result')
def on_frame_result(data):
    global latest_result
    
    now = time.time()
    latency_ms = 0
    
    with lock:
        if packet_timestamps:
            # Sync
            item = packet_timestamps.popleft()
            t_start, t_cap_enc = item
            
            # Total RTT
            latency_ms = (now - t_start) * 1000
            metrics["total_rtt"].append(latency_ms)
            metrics["capture_encode"].append(t_cap_enc)
            
            # Server Timings
            result = data.get('result', {})
            server_times = result.get('server_timings', {})
            
            t_mp = server_times.get('mediapipe', 0)
            t_vel = server_times.get('velocity', 0)
            t_inf = server_times.get('inference', 0)
            t_srv_enc = server_times.get('encode', 0)
            t_srv_dec = server_times.get('decode', 0)
            
            metrics["mediapipe"].append(t_mp)
            metrics["velocity"].append(t_vel)
            # Only count inference if it happened (non-zero) or count 0 is fine for avg?
            # Usually we only care about avg when it runs, but for "pipeline stage" usually implies per-frame avg.
            # If inference doesn't run every frame, the average will be low.
            # The prompt implies an "End-to-End Latency Breakdown" potentially for a *gesture recognition event*.
            # However, collecting per-frame data is safer.
            metrics["inference"].append(t_inf)
            
            metrics["response_process"].append(t_srv_enc) # JSON/Broadcast is dominated by encode
            
            # Network = Total - (ClientTime + ServerTime)
            # ServerTime = decode + mediapipe + velocity + inference + encode
            t_server_total = t_srv_dec + t_mp + t_vel + t_inf + t_srv_enc
            
            # This 'network' includes transport time both ways + Flask overhead
            t_net = latency_ms - (t_cap_enc + t_server_total)
            metrics["network"].append(max(0, t_net))

            # Update display
            latest_result = result
            latest_result['network_latency'] = latency_ms

@sio.on('error')
def on_server_error(data):
    print(f"[SERVER ERROR] {data}")
    with lock:
        if packet_timestamps:
            packet_timestamps.popleft()

# ==========================================
# REPORTING
# ==========================================
def print_report():
    print("\n\n")
    print("="*60)
    print("TABLE VII: END-TO-END LATENCY BREAKDOWN (Collected Data)")
    print("="*60)
    print(f"{'Pipeline stage':<35} {'Avg (ms)':<10} {'Max (ms)':<10}")
    print("-" * 60)
    
    def get_stats(key):
        data = metrics[key]
        if not data: return 0.0, 0.0
        # Filter out outliers or startup noise? defaulting to raw
        return np.mean(data), np.max(data)

    stats_map = [
        ("Frame capture + JPEG encode", "capture_encode"),
        ("Network (client -> Flask)", "network"),
        ("MediaPipe landmark extraction", "mediapipe"),
        ("HMS velocity evaluation", "velocity"),
        ("SGN inference (DNN, 68-dim)", "inference"),
        ("JSON response + broadcast", "response_process"),
        ("Total end-to-end", "total_rtt")
    ]
    
    total_avg_sum = 0
    
    for label, key in stats_map:
        avg, mx = get_stats(key)
        if label != "Total end-to-end":
            total_avg_sum += avg
        else:
            # Check if sum matches total
            pass
            
        print(f"{label:<35} {avg:<10.1f} {mx:<10.1f}")
        
    print("-" * 60)
    print(f"Frames Analyzed: {len(metrics['total_rtt'])}")
    print("="*60)

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    print("="*50)
    print("ASL LATENCY BREAKDOWN TOOL")
    print("="*50)
    
    try:
        sio.connect(SERVER_URL)
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return

    cap = None
    if USE_WEBCAM:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        if not cap.isOpened():
             print("[ERROR] Webcam not found.")
             return

    print("[CLIENT] Streaming... Press 'q' to stop and generate report.")

    while running:
        # Measure Capture + Encode
        t0 = time.time()
        
        if USE_WEBCAM:
            ret, frame = cap.read()
            if not ret: break
        else:
            frame = np.zeros((CAP_HEIGHT, CAP_WIDTH, 3), dtype=np.uint8)
            time.sleep(0.033)

        frame = cv2.flip(frame, 1)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_frame = base64.b64encode(buffer).decode('utf-8')
        frame_data = f"data:image/jpeg;base64,{b64_frame}"
        
        t_capture_encode = (time.time() - t0) * 1000
        
        # Send
        with lock:
            packet_timestamps.append((t0, t_capture_encode)) # Store t0 and processing time
        
        sio.emit('process_frame', {'frame': frame_data})

        # Visualization (HUD)
        display_frame = frame.copy()
        
        with lock:
            text = str(latest_result.get('text', ''))
            mode_map = {0:"IDLE", 1:"STATIC", 2:"DYNAMIC", 3:"RESULT"}
            mode = mode_map.get(latest_result.get('mode', 0), "Unknown")
            lat = latest_result.get('network_latency', 0)
            
            cv2.rectangle(display_frame, (0,0), (300, 120), (0,0,0), -1)
            cv2.putText(display_frame, f"Gesture: {text}", (10, 30), 1, 1.5, (0,255,0), 2)
            cv2.putText(display_frame, f"Mode: {mode}", (10, 60), 1, 1.2, (200,200,200), 1)
            cv2.putText(display_frame, f"Latency: {lat:.1f}ms", (10, 90), 1, 1.2, (0,255,255), 1)

        cv2.imshow('Latency Test', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap: cap.release()
    cv2.destroyAllWindows()
    sio.disconnect()
    
    print_report()

if __name__ == "__main__":
    main()
