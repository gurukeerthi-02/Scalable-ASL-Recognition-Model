'use client';

import { useEffect, useRef, useCallback, useState } from 'react';

interface UseTTSProps {
  enabled?: boolean;
  rate?: number;
  pitch?: number;
  volume?: number;
}

export function useTTS({
  enabled = true,
  rate = 1.0,
  pitch = 1.0,
  volume = 1.0,
}: UseTTSProps = {}) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const synthRef = useRef<SpeechSynthesis | null>(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      synthRef.current = window.speechSynthesis;
      setIsSupported(true);
    } else {
      console.warn('SpeechSynthesis API not supported in this browser');
      setIsSupported(false);
    }

    return () => {
      if (synthRef.current) {
        synthRef.current.cancel();
      }
    };
  }, []);

  const speak = useCallback(
    (text: string) => {
      if (!enabled || !isSupported || !synthRef.current || !text.trim()) {
        return;
      }

      synthRef.current.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = rate;
      utterance.pitch = pitch;
      utterance.volume = volume;

      utterance.onstart = () => {
        setIsSpeaking(true);
      };

      utterance.onend = () => {
        setIsSpeaking(false);
      };

      utterance.onerror = (event) => {
        console.error('Speech synthesis error:', event);
        setIsSpeaking(false);
      };

      synthRef.current.speak(utterance);
    },
    [enabled, isSupported, rate, pitch, volume]
  );

  const stop = useCallback(() => {
    if (synthRef.current) {
      synthRef.current.cancel();
      setIsSpeaking(false);
    }
  }, []);

  const pause = useCallback(() => {
    if (synthRef.current) {
      synthRef.current.pause();
    }
  }, []);

  const resume = useCallback(() => {
    if (synthRef.current) {
      synthRef.current.resume();
    }
  }, []);

  return {
    speak,
    stop,
    pause,
    resume,
    isSpeaking,
    isSupported,
  };
}
