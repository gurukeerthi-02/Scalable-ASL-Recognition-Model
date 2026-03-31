'use client';

import { useEffect, useRef, useCallback, useState } from 'react';

export type VoiceGender = 'female' | 'male';

interface UseTTSProps {
  enabled?: boolean;
  rate?: number;
  pitch?: number;
  volume?: number;
  gender?: VoiceGender;
}

/** Heuristically determine the gender of a SpeechSynthesisVoice by its name. */
function guessGender(voice: SpeechSynthesisVoice): VoiceGender {
  const name = voice.name.toLowerCase();
  const femaleKeywords = [
    'female', 'zira', 'hazel', 'susan', 'samantha', 'victoria',
    'karen', 'moira', 'fiona', 'tessa', 'veena', 'woman', 'girl',
  ];
  const maleKeywords = [
    'male', 'david', 'mark', 'james', 'daniel', 'alex', 'fred',
    'tom', 'man', 'guy', 'boy',
  ];

  if (femaleKeywords.some((kw) => name.includes(kw))) return 'female';
  if (maleKeywords.some((kw) => name.includes(kw))) return 'male';

  // Google voices: "Google UK English Female" / "Google UK English Male"
  if (name.includes('google') && name.includes('female')) return 'female';
  if (name.includes('google') && name.includes('male')) return 'male';

  // Microsoft voices: odd-indexed voices in en-US set tend to be female
  return 'female'; // safe default
}

/** Score a voice: prefer Google/Microsoft, then en-US, then other en voices. */
function scoreVoice(voice: SpeechSynthesisVoice): number {
  let score = 0;
  const name = voice.name.toLowerCase();

  if (name.includes('google')) score += 30;
  else if (name.includes('microsoft')) score += 20;

  if (voice.lang === 'en-US') score += 15;
  else if (voice.lang.startsWith('en')) score += 8;

  return score;
}

export function useTTS({
  enabled = true,
  rate = 0.92,      // slightly slower → clearer
  pitch = 1.05,     // slightly higher → more natural
  volume = 1.0,
  gender = 'female',
}: UseTTSProps = {}) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
  const synthRef = useRef<SpeechSynthesis | null>(null);

  // Load and cache voices (they load asynchronously on many browsers)
  useEffect(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      console.warn('SpeechSynthesis API not supported in this browser');
      return;
    }

    const synth = window.speechSynthesis;
    synthRef.current = synth;
    setIsSupported(true);

    const loadVoices = () => {
      const voices = synth.getVoices();
      if (voices.length > 0) {
        // Keep only English voices, sorted by quality score
        const englishVoices = voices
          .filter((v) => v.lang.startsWith('en'))
          .sort((a, b) => scoreVoice(b) - scoreVoice(a));
        setAvailableVoices(englishVoices);
      }
    };

    loadVoices(); // already loaded in some browsers
    synth.addEventListener('voiceschanged', loadVoices);

    return () => {
      synth.removeEventListener('voiceschanged', loadVoices);
      synth.cancel();
    };
  }, []);

  /** Pick the best matching voice for the requested gender. */
  const pickVoice = useCallback(
    (voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null => {
      if (voices.length === 0) return null;

      // Filter to the desired gender
      const genderMatches = voices.filter((v) => guessGender(v) === gender);
      if (genderMatches.length > 0) return genderMatches[0]; // already sorted by score

      // Fallback: use any voice
      return voices[0];
    },
    [gender]
  );

  const speak = useCallback(
    (text: string) => {
      if (!enabled || !isSupported || !synthRef.current || !text.trim()) return;

      synthRef.current.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = rate;
      utterance.pitch = pitch;
      utterance.volume = volume;
      utterance.lang = 'en-US';

      const voice = pickVoice(availableVoices);
      if (voice) utterance.voice = voice;

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = (event) => {
        console.error('Speech synthesis error:', event);
        setIsSpeaking(false);
      };

      synthRef.current.speak(utterance);
    },
    [enabled, isSupported, rate, pitch, volume, availableVoices, pickVoice]
  );

  const stop = useCallback(() => {
    synthRef.current?.cancel();
    setIsSpeaking(false);
  }, []);

  const pause = useCallback(() => synthRef.current?.pause(), []);
  const resume = useCallback(() => synthRef.current?.resume(), []);

  /** Voices split by heuristic gender — useful for a picker UI. */
  const femaleVoices = availableVoices.filter((v) => guessGender(v) === 'female');
  const maleVoices = availableVoices.filter((v) => guessGender(v) === 'male');

  return {
    speak,
    stop,
    pause,
    resume,
    isSpeaking,
    isSupported,
    availableVoices,
    femaleVoices,
    maleVoices,
  };
}
