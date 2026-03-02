import { ASLRecognitionResponse } from '@/types';
import { io, Socket } from 'socket.io-client';

export class ASLClient {
  private backendUrl: string;
  private socket: Socket | null = null;
  private onRecognitionCallback?: (result: any) => void;

  // Latency Tracking
  private packetTimestamps: number[] = [];
  private metrics = {
    capture_encode: [] as number[],
    network: [] as number[],
    mediapipe: [] as number[],
    velocity: [] as number[],
    inference: [] as number[],
    response_process: [] as number[],
    total_rtt: [] as number[]
  };
  private frameCount = 0;

  // Performance optimizations
  private readonly ENABLE_LOGGING = process.env.NODE_ENV === 'development';
  private readonly LOG_INTERVAL = 60; // Log every 60 frames instead of 30
  private pendingFrame = false;
  private latestResult: any = null;

  constructor(backendUrl: string) {
    this.backendUrl = backendUrl;
  }

  connect(onRecognition: (result: any) => void): void {
    this.onRecognitionCallback = onRecognition;

    this.socket = io(this.backendUrl, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 5
    });

    this.socket.on('connect', () => {
      if (this.ENABLE_LOGGING) {
        console.log('ASL SocketIO connected');
      }
    });

    this.socket.on('frame_result', (data: any) => {
      this.handleFrameResult(data);
    });

    this.socket.on('connect_error', (error: any) => {
      console.error('ASL SocketIO error:', error);
    });

    this.socket.on('disconnect', () => {
      if (this.ENABLE_LOGGING) {
        console.log('ASL SocketIO disconnected');
      }
    });
  }

  private handleFrameResult(data: any): void {
    const now = performance.now();

    // Calculate Latency only if we have a timestamp
    if (this.packetTimestamps.length === 0) return;

    const t_start = this.packetTimestamps.shift()!;
    const latency_ms = now - t_start;

    // Extract server timings
    const result = data.result || {};
    const server_times = result.server_timings || {};

    const t_mp = server_times.mediapipe || 0;
    const t_vel = server_times.velocity || 0;
    const t_inf = server_times.inference || 0;
    const t_srv_enc = server_times.encode || 0;
    const t_srv_dec = server_times.decode || 0;

    // Store Metrics (use array pooling to prevent memory bloat)
    const MAX_SAMPLES = 100; // Keep only last 100 samples

    this.updateMetric(this.metrics.total_rtt, latency_ms, MAX_SAMPLES);
    this.updateMetric(this.metrics.mediapipe, t_mp, MAX_SAMPLES);
    this.updateMetric(this.metrics.velocity, t_vel, MAX_SAMPLES);
    this.updateMetric(this.metrics.inference, t_inf, MAX_SAMPLES);
    this.updateMetric(this.metrics.response_process, t_srv_enc, MAX_SAMPLES);

    const t_server_total = t_srv_dec + t_mp + t_vel + t_inf + t_srv_enc;
    const t_client_network = Math.max(0, latency_ms - t_server_total);
    this.updateMetric(this.metrics.network, t_client_network, MAX_SAMPLES);

    this.frameCount++;

    // Log Table less frequently
    if (this.ENABLE_LOGGING && this.frameCount % this.LOG_INTERVAL === 0) {
      this.printLatencyTable();
    }

    // Use requestAnimationFrame to batch UI updates
    if (this.onRecognitionCallback && result) {
      this.latestResult = {
        text: result.text || '',
        confidence: result.confidence || 0,
        mode: result.mode,
        motion: result.motion,
        buffer_size: result.buffer_size,
        stable_count: result.stable_count,
        latency: latency_ms
      };

      if (!this.pendingFrame) {
        this.pendingFrame = true;
        requestAnimationFrame(() => {
          if (this.onRecognitionCallback && this.latestResult) {
            this.onRecognitionCallback(this.latestResult);
          }
          this.pendingFrame = false;
        });
      }
    }
  }

  private updateMetric(array: number[], value: number, maxSize: number): void {
    array.push(value);
    if (array.length > maxSize) {
      array.shift(); // Remove oldest sample
    }
  }

  sendFrame(frameData: string): void {
    if (this.socket?.connected) {
      // CONGESTION CONTROL:
      // If we have more than 2 frames in flight, skip this frame.
      // This prevents 'buffer bloat' where the network queue piles up,
      // creating the 4-second latency spikes seen in logs.
      if (this.packetTimestamps.length > 2) {
        return;
      }

      this.packetTimestamps.push(performance.now());
      this.socket.emit('process_frame', { frame: frameData });
    }
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    // Clear metrics
    this.packetTimestamps = [];
    Object.keys(this.metrics).forEach(key => {
      (this.metrics as any)[key] = [];
    });
  }

  isConnected(): boolean {
    return this.socket !== null && this.socket.connected;
  }

  private printLatencyTable(): void {
    const getStats = (arr: number[]) => {
      if (arr.length === 0) return { avg: '0.0', max: '0.0', min: '0.0' };
      const sum = arr.reduce((a, b) => a + b, 0);
      const avg = (sum / arr.length).toFixed(1);
      const max = Math.max(...arr).toFixed(1);
      const min = Math.min(...arr).toFixed(1);
      return { avg, max, min };
    };

    const stats = {
      mediaPipe: getStats(this.metrics.mediapipe),
      velocity: getStats(this.metrics.velocity),
      inference: getStats(this.metrics.inference),
      response: getStats(this.metrics.response_process),
      net_client: getStats(this.metrics.network),
      total: getStats(this.metrics.total_rtt)
    };

    // Use a single console.table instead of console.group
    console.log('\n📊 TABLE VII: END-TO-END LATENCY BREAKDOWN (Live Data)\n');
    console.table({
      'Client Capture + Network': {
        'Avg (ms)': stats.net_client.avg,
        'Min (ms)': stats.net_client.min,
        'Max (ms)': stats.net_client.max
      },
      'MediaPipe Landmark': {
        'Avg (ms)': stats.mediaPipe.avg,
        'Min (ms)': stats.mediaPipe.min,
        'Max (ms)': stats.mediaPipe.max
      },
      'Velocity Check': {
        'Avg (ms)': stats.velocity.avg,
        'Min (ms)': stats.velocity.min,
        'Max (ms)': stats.velocity.max
      },
      'SGN Inference': {
        'Avg (ms)': stats.inference.avg,
        'Min (ms)': stats.inference.min,
        'Max (ms)': stats.inference.max
      },
      'Response Encode': {
        'Avg (ms)': stats.response.avg,
        'Min (ms)': stats.response.min,
        'Max (ms)': stats.response.max
      },
      'TOTAL END-TO-END': {
        'Avg (ms)': stats.total.avg,
        'Min (ms)': stats.total.min,
        'Max (ms)': stats.total.max
      }
    });
    console.log(`📈 Frames processed: ${this.frameCount}\n`);
  }

  // Public method to manually trigger latency report
  public getLatencyReport() {
    this.printLatencyTable();
  }
}

export async function sendFrameHTTP(
  backendUrl: string,
  frameData: string
): Promise<ASLRecognitionResponse> {
  const response = await fetch(`${backendUrl}/recognize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ frame: frameData }),
  });

  if (!response.ok) {
    throw new Error('ASL recognition request failed');
  }

  return response.json();
}