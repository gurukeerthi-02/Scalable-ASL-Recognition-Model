'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { useRouter, useParams, useSearchParams } from 'next/navigation';
import { VideoGrid } from '@/components/video/VideoGrid';
import { CallControls } from '@/components/video/CallControls';
import { ASLIndicator } from '@/components/video/ASLIndicator';
import { useWebRTC } from '@/hooks/useWebRTC';
import { useASLStream } from '@/hooks/useASLStream';
import { useTTS } from '@/hooks/useTTS';
import { ASLRecognitionResponse } from '@/types';
import { supabase } from '@/lib/supabase';
import { Card, CardContent } from '@/components/ui/card';
import { AlertCircle, Video, Users, Copy, Check, Info, Settings, Maximize2, Hand, ChevronRight, MessageSquare, Send } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export default function CallPage() {
  const router = useRouter();
  const params = useParams();
  const searchParams = useSearchParams();

  const roomId = params.roomId as string;
  const displayName = searchParams.get('name') || 'Anonymous';
  const peerId = useRef(`peer-${Math.random().toString(36).substring(7)}`).current;

  const [roomName, setRoomName] = useState<string>('');
  const [isCopied, setIsCopied] = useState(false);
  const [aslEnabled, setAslEnabled] = useState(false);
  const [localVideoElement, setLocalVideoElement] = useState<HTMLVideoElement | null>(null);
  const [lastRecognition, setLastRecognition] = useState<ASLRecognitionResponse | null>(null);
  const [participantId, setParticipantId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sentenceBuffer, setSentenceBuffer] = useState<string>('');
  const [receivedMessages, setReceivedMessages] = useState<Array<{ peerId: string, text: string, timestamp: number, displayName?: string }>>([]);
  const [lastRecognizedLetter, setLastRecognizedLetter] = useState<string>('');
  const [voiceOutEnabled, setVoiceOutEnabled] = useState(false);
  const [handRaised, setHandRaised] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const [showASLPanel, setShowASLPanel] = useState(true);
  const lastRecognizedLetterRef = useRef<string>('');

  const { toast } = useToast();
  const { speak } = useTTS({ enabled: true, rate: 1.0, volume: 1.0 });

  useEffect(() => {
    const fetchRoomDetails = async () => {
      const { data, error } = await supabase
        .from('rooms')
        .select('name')
        .eq('id', roomId)
        .single();

      if (data && !error) {
        setRoomName(data.name);
      }
    };

    fetchRoomDetails();
  }, [roomId]);

  const convertToSpeechText = useCallback((text: string) => {
    const wordMappings: { [key: string]: string } = {
      'HI': 'hi', 'BYE': 'bye', 'YES': 'yes', 'NO': 'no', 'OK': 'okay',
      'HELP': 'help', 'PLEASE': 'please', 'THANK': 'thank', 'YOU': 'you',
      'ME': 'me', 'GOOD': 'good', 'BAD': 'bad', 'HELLO': 'hello',
      'SORRY': 'sorry', 'LOVE': 'love', 'WATER': 'water', 'FOOD': 'food',
      'HOME': 'home', 'WORK': 'work', 'SCHOOL': 'school'
    };

    const upperText = text.toUpperCase();
    if (wordMappings[upperText]) {
      return wordMappings[upperText];
    }
    // If it's a single word (no spaces), speak it as a word, not as individual letters.
    // If it's already multiple words, keep it as is.
    return text;
  }, []);

  const handleRemoteASLToggle = useCallback(
    (remotePeerId: string, enabled: boolean) => {
      console.log(`Remote peer ${remotePeerId} ${enabled ? 'enabled' : 'disabled'} ASL mode`);
    },
    []
  );

  const handleTextMessage = useCallback(
    (remotePeerId: string, text: string, remoteDisplayName?: string) => {
      setReceivedMessages(prev => [...prev, {
        peerId: remotePeerId,
        text,
        timestamp: Date.now(),
        displayName: remoteDisplayName || 'Participant'
      }]);

      const speechText = convertToSpeechText(text);
      speak(speechText);
    },
    [speak, convertToSpeechText]
  );

  const {
    localStream,
    peers,
    audioEnabled,
    videoEnabled,
    toggleAudio,
    toggleVideo,
    notifyASLToggle,
    sendTextMessage,
    leaveRoom: originalLeaveRoom,
  } = useWebRTC({
    roomId,
    peerId,
    displayName,
    onRemoteASLToggle: handleRemoteASLToggle,
    onTextMessage: handleTextMessage,
  });

  const handleASLRecognition = useCallback(
    (result: ASLRecognitionResponse) => {
      setLastRecognition(result);

      if (result.text && result.text !== 'None') {
        if (result.text !== lastRecognizedLetterRef.current) {
          console.log(`[ASL] Appending to buffer: ${result.text}`);
          setSentenceBuffer(prev => prev + result.text);
          lastRecognizedLetterRef.current = result.text;

          // if (result.text.length > 1) {
          //   speak(convertToSpeechText(result.text));
          // }
        }
      } else {
        lastRecognizedLetterRef.current = '';
      }

      if (participantId && result.text) {
        supabase
          .from('call_logs')
          .insert({
            room_id: roomId,
            participant_id: participantId,
            event_type: 'asl_recognition',
            event_data: result,
          })
          .then(({ error }) => {
            if (error) console.error('Error logging ASL:', error);
          });
      }
    },
    [roomId, participantId, speak, convertToSpeechText]
  );

  const handleSendSentence = useCallback(() => {
    if (sentenceBuffer.trim()) {
      const message = sentenceBuffer.trim();
      sendTextMessage(message);

      // Speak the message when sent
      speak(convertToSpeechText(message));

      // Show confirmation toast
      toast({
        title: "Message Sent",
        description: `"${message}" has been shared with participants`,
        duration: 3000,
      });

      // Add to own message list
      setReceivedMessages(prev => [...prev, {
        peerId,
        text: message,
        timestamp: Date.now(),
        displayName: 'You'
      }]);

      setSentenceBuffer('');
    }
  }, [sentenceBuffer, sendTextMessage, speak, convertToSpeechText, toast]);

  const handleClearSentence = useCallback(() => {
    setSentenceBuffer('');
    // Remove lastRecognizedLetterRef clearing to prevent re-adding current sign
  }, []);

  const handleAddSpace = useCallback(() => {
    setSentenceBuffer(prev => prev + ' ');
    // Remove lastRecognizedLetterRef clearing to prevent re-adding current sign
  }, []);

  const handleToggleVoiceOut = useCallback(() => {
    const newState = !voiceOutEnabled;
    setVoiceOutEnabled(newState);
    setAslEnabled(newState);
    notifyASLToggle(newState);
  }, [voiceOutEnabled, notifyASLToggle]);

  const handleToggleHandRaise = useCallback(() => {
    setHandRaised(!handRaised);
  }, [handRaised]);

  const { isStreaming, getFrameCount } = useASLStream({
    enabled: voiceOutEnabled,
    videoElement: localVideoElement,
    onRecognition: handleASLRecognition,
    fps: 10,
  });

  useEffect(() => {
    const joinRoom = async () => {
      try {
        const { data: roomData, error: roomError } = await supabase
          .from('rooms')
          .select('*')
          .eq('id', roomId)
          .maybeSingle();

        if (roomError) throw roomError;

        if (!roomData) {
          setError('Room not found or has ended');
          return;
        }

        if (!roomData.is_active) {
          setError('This call has ended');
          return;
        }

        const { data: participantData, error: participantError } = await supabase
          .from('participants')
          .insert({
            room_id: roomId,
            peer_id: peerId,
            display_name: displayName,
            is_asl_enabled: false,
          })
          .select()
          .single();

        if (participantError) throw participantError;

        setParticipantId(participantData.id);

        await supabase.from('call_logs').insert({
          room_id: roomId,
          participant_id: participantData.id,
          event_type: 'join',
          event_data: { display_name: displayName },
        });
      } catch (error) {
        console.error('Error joining room:', error);
        setError('Failed to join room');
      }
    };

    joinRoom();
  }, [roomId, peerId, displayName]);

  const handleLeave = useCallback(async () => {
    if (participantId) {
      const { data: participantData } = await supabase
        .from('participants')
        .select('display_name')
        .eq('id', participantId)
        .single();

      if (participantData) {
        const { data: roomData } = await supabase
          .from('rooms')
          .select('created_by')
          .eq('id', roomId)
          .single();

        if (roomData && roomData.created_by === participantData.display_name) {
          await supabase
            .from('rooms')
            .update({ is_active: false })
            .eq('id', roomId);
        }
      }

      await supabase
        .from('participants')
        .update({ left_at: new Date().toISOString() })
        .eq('id', participantId);

      await supabase.from('call_logs').insert({
        room_id: roomId,
        participant_id: participantId,
        event_type: 'leave',
      });
    }

    originalLeaveRoom();
    router.push('/');
  }, [participantId, roomId, originalLeaveRoom, router]);

  const copyRoomId = async () => {
    try {
      await navigator.clipboard.writeText(roomId);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-yellow-400 via-yellow-300 to-yellow-400 flex items-center justify-center p-4">
        <Card className="p-8 bg-white border-0 shadow-2xl max-w-md animate-fade-in">
          <div className="flex flex-col items-center text-center">
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mb-6">
              <AlertCircle className="w-8 h-8 text-red-600" />
            </div>
            <h2 className="text-2xl font-bold mb-2 text-black">Session Error</h2>
            <p className="text-black/70 mb-8 leading-relaxed">{error}</p>
            <Button
              onClick={() => router.push('/')}
              className="w-full h-12 bg-black text-yellow-400 hover:bg-black/90"
            >
              Back to Home
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <TooltipProvider>
      <div className="flex flex-col h-[100dvh] bg-gradient-to-br from-yellow-400 via-yellow-300 to-yellow-400 overflow-hidden">

        {/* Professional Header */}
        <header className="h-14 md:h-18 border-b border-black/10 flex items-center justify-between px-3 md:px-6 bg-white/90 backdrop-blur-xl z-50 flex-shrink-0 shadow-sm">
          <div className="flex items-center gap-2 md:gap-3 min-w-0 flex-1">
            <div className="flex items-center gap-1.5 md:gap-2">
              <div className="p-1.5 bg-black rounded-lg md:rounded-xl flex-shrink-0">
                <Hand className="w-4 h-4 md:w-5 md:h-5 text-yellow-400" />
              </div>
              <span className="hidden sm:block text-base md:text-lg font-bold text-black">ASL Meet</span>
            </div>

            <div className="h-8 w-px bg-black/10 mx-2 hidden md:block" />

            <div className="min-w-0 flex flex-col">
              <div className="flex items-center gap-2">
                <h1 className="text-xs md:text-base font-bold truncate text-black max-w-[100px] sm:max-w-none">
                  {roomName || 'Active Session'}
                </h1>
                <Badge className="bg-green-500 text-white hover:bg-green-600 px-1.5 md:px-2 h-4 md:h-5 text-[9px] md:text-[10px] font-bold">
                  LIVE
                </Badge>
              </div>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-[10px] md:text-xs text-black/50 font-mono">
                  ID: {roomId}
                </span>
                <button
                  onClick={copyRoomId}
                  className="flex items-center gap-1 hover:text-black transition-colors"
                >
                  <span className="text-[10px] text-black/50 hover:text-black underline">
                    {isCopied ? 'Copied!' : 'Copy'}
                  </span>
                  {isCopied ? <Check className="w-3 h-3 text-green-600" /> : <Copy className="w-3 h-3 text-black/50" />}
                </button>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 md:gap-3 flex-shrink-0">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowASLPanel(!showASLPanel)}
              className={`rounded-full ${showASLPanel ? 'bg-yellow-400 text-black' : 'bg-black/5 text-black hover:bg-black/10'}`}
            >
              <MessageSquare className="w-5 h-5" />
            </Button>

            <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 bg-black/5 rounded-full">
              <Users className="w-4 h-4 text-black" />
              <span className="text-xs font-semibold text-black">{peers.size + 1}</span>
            </div>

            <Button
              onClick={handleLeave}
              className="rounded-full px-4 md:px-6 h-9 md:h-10 font-bold bg-red-600 hover:bg-red-700 text-white shadow-lg"
            >
              <span className="hidden sm:inline">Leave</span>
              <span className="sm:hidden">End</span>
            </Button>
          </div>
        </header>

        {/* Main Video Area */}
        <main className="flex-1 flex flex-col lg:flex-row gap-0 min-h-0">
          {/* Video Grid */}
          <div className="flex-1 min-h-0 h-full w-full bg-gray-300 ">
            <VideoGrid
              localStream={localStream}
              localDisplayName={displayName}
              audioEnabled={audioEnabled}
              videoEnabled={videoEnabled}
              isAslEnabled={voiceOutEnabled}
              handRaised={handRaised}
              peers={peers}
              onLocalVideoReady={setLocalVideoElement}
            />

            {/* Video Off State */}
            {!videoEnabled && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-gray-900 to-black z-20">
                <div className="w-20 h-20 bg-yellow-400/20 rounded-full flex items-center justify-center mb-4 border-2 border-yellow-400/30">
                  <UserIcon className="w-10 h-10 text-yellow-400" />
                </div>
                <p className="text-white/70 font-medium">Camera is Off</p>
              </div>
            )}

            {/* Participant Count (Mobile) */}
            <div className="lg:hidden absolute top-4 right-4 z-30 px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-full border border-white/10">
              <div className="flex items-center gap-2">
                <Users className="w-4 h-4 text-white" />
                <span className="text-xs font-semibold text-white">{peers.size + 1}</span>
              </div>
            </div>
          </div>

          {/* ASL Panel - Desktop Sidebar / Mobile Bottom Sheet */}
          {showASLPanel && (
            <>
              {/* Desktop Sidebar */}
              <div className={`hidden lg:flex lg:w-[360px] xl:w-[400px] flex-shrink-0 bg-white border-l border-black/10 flex-col shadow-2xl transition-all duration-300`}>
                <div className="p-4 border-b border-black/10 flex items-center justify-between bg-yellow-400/10">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${voiceOutEnabled ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
                    <span className="text-sm font-bold text-black">ASL Interpreter & Chat</span>
                  </div>
                  {voiceOutEnabled && <Badge className="text-[10px] bg-black text-yellow-400">Recognition Active</Badge>}
                </div>
                <div className="flex-1 overflow-hidden">
                  <ASLIndicator
                    isActive={voiceOutEnabled}
                    lastRecognition={lastRecognition || undefined}
                    frameCount={getFrameCount()}
                    sentenceBuffer={sentenceBuffer}
                    onSendSentence={handleSendSentence}
                    onClearSentence={handleClearSentence}
                    onAddSpace={handleAddSpace}
                    receivedMessages={receivedMessages}
                  />
                </div>
              </div>

              {/* Mobile Bottom Sheet */}
              <div className="lg:hidden absolute bottom-24 left-2 right-2 z-30 animate-in slide-in-from-bottom duration-300">
                <div className="bg-white rounded-2xl shadow-2xl border border-black/10 flex flex-col overflow-hidden max-h-[50vh]">
                  <div className="p-3 border-b border-black/10 flex items-center justify-between bg-yellow-400/5">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${voiceOutEnabled ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
                      <span className="text-xs font-bold text-black uppercase tracking-wider">Interpreter & Chat</span>
                    </div>
                    {voiceOutEnabled && <Badge className="text-[9px] bg-black text-yellow-400 border-0 h-4">LIVE</Badge>}
                    <Button variant="ghost" size="icon" onClick={() => setShowASLPanel(false)} className="h-6 w-6 rounded-full">
                      <ChevronRight className="w-4 h-4 rotate-90" />
                    </Button>
                  </div>
                  <div className="flex-1 overflow-y-auto">
                    <ASLIndicator
                      isActive={voiceOutEnabled}
                      lastRecognition={lastRecognition || undefined}
                      frameCount={getFrameCount()}
                      sentenceBuffer={sentenceBuffer}
                      onSendSentence={handleSendSentence}
                      onClearSentence={handleClearSentence}
                      onAddSpace={handleAddSpace}
                      receivedMessages={receivedMessages}
                    />
                  </div>
                </div>
              </div>
            </>
          )}
        </main>

        {/* Control Bar */}
        <footer className="h-16 md:h-24 bg-white/95 backdrop-blur-xl border-t border-black/10 flex items-center justify-center px-4 z-40 shadow-lg">
          <div className="max-w-4xl w-full">
            <CallControls
              audioEnabled={audioEnabled}
              videoEnabled={videoEnabled}
              voiceOutEnabled={voiceOutEnabled}
              handRaised={handRaised}
              onToggleAudio={toggleAudio}
              onToggleVideo={toggleVideo}
              onToggleVoiceOut={handleToggleVoiceOut}
              onToggleHandRaise={handleToggleHandRaise}
              onLeave={handleLeave}
            />
          </div>
        </footer>
      </div>
    </TooltipProvider>
  );
}

function UserIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
      />
    </svg>
  );
}