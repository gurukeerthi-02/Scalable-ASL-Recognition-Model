'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { ASLClient } from '@/lib/asl-client';
import { captureVideoFrame } from '@/lib/webrtc';
import { ASLRecognitionResponse } from '@/types';

interface UseASLStreamProps {
  enabled: boolean;
  videoElement: HTMLVideoElement | null;
  onRecognition: (result: ASLRecognitionResponse) => void;
  fps?: number;
}

export function useASLStream({
  enabled,
  videoElement,
  onRecognition,
  fps = 10,
}: UseASLStreamProps) {
  const [isStreaming, setIsStreaming] = useState(false);
  const clientRef = useRef<ASLClient | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const frameCountRef = useRef(0);

  useEffect(() => {
    if (!enabled || !videoElement) {
      if (clientRef.current) {
        clientRef.current.disconnect();
        clientRef.current = null;
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setIsStreaming(false);
      return;
    }

    // primary server: using [IP_ADDRESS]
    // fallback server: http://localhost:5000

    const ASL_BACKEND_URL = process.env.NEXT_PUBLIC_ASL_BACKEND_URL_PRIMARY || 
      process.env.NEXT_PUBLIC_ASL_BACKEND_URL_FALLBACK || 
        'http://localhost:5000';

    clientRef.current = new ASLClient(ASL_BACKEND_URL);
    clientRef.current.connect(onRecognition);

    const captureInterval = 1000 / fps;

    intervalRef.current = setInterval(() => {
      if (clientRef.current && clientRef.current.isConnected()) {
        const frameData = captureVideoFrame(videoElement);
        if (frameData) {
          clientRef.current.sendFrame(frameData);
          frameCountRef.current += 1;
        }
      }
    }, captureInterval);

    setIsStreaming(true);

    return () => {
      if (clientRef.current) {
        clientRef.current.disconnect();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [enabled, videoElement, onRecognition, fps]);

  const getFrameCount = useCallback(() => {
    return frameCountRef.current;
  }, []);

  return {
    isStreaming,
    getFrameCount,
  };
}
