export const ICE_SERVERS: RTCConfiguration = {
  iceServers: [
    {
      urls: 'stun:stun.l.google.com:19302',
    },
    {
      urls: 'stun:stun1.l.google.com:19302',
    },
  ],
};

export function createPeerConnection(): RTCPeerConnection {
  return new RTCPeerConnection(ICE_SERVERS);
}

export async function getUserMedia(
  audioEnabled: boolean = true,
  videoEnabled: boolean = true
): Promise<MediaStream> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: audioEnabled,
      video: videoEnabled
        ? {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        }
        : false,
    });
    return stream;
  } catch (error) {
    console.error('Error accessing media devices:', error);
    throw error;
  }
}

export function captureVideoFrame(
  videoElement: HTMLVideoElement
): string | null {
  if (!videoElement || videoElement.readyState !== videoElement.HAVE_ENOUGH_DATA) {
    return null;
  }

  const canvas = document.createElement('canvas');
  // Aggressive Optimization: Resize to 320px width.
  // MediaPipe only needs small resolution for landmark detection.
  // This reduces data size by another 4x compared to 640px.
  const maxWidth = 320;
  const scale = Math.min(1.0, maxWidth / videoElement.videoWidth);

  canvas.width = videoElement.videoWidth * scale;
  canvas.height = videoElement.videoHeight * scale;

  const context = canvas.getContext('2d', { alpha: false });
  if (!context) return null;

  // Faster drawing with no context smoothing
  context.imageSmoothingEnabled = false;
  context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  // Quality 0.6 is the "sweet spot" for speed vs accuracy for MediaPipe.
  return canvas.toDataURL('image/jpeg', 0.6);
}

export function toggleAudioTrack(stream: MediaStream, enabled: boolean): void {
  stream.getAudioTracks().forEach((track) => {
    track.enabled = enabled;
  });
}

export function toggleVideoTrack(stream: MediaStream, enabled: boolean): void {
  stream.getVideoTracks().forEach((track) => {
    track.enabled = enabled;
  });
}

export function stopMediaStream(stream: MediaStream): void {
  stream.getTracks().forEach((track) => {
    track.stop();
  });
}
