'use client';

import { useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { SignalingMessage } from '@/types';

interface UseSignalingProps {
  roomId: string;
  peerId: string;
  displayName: string;
  onOffer: (peerId: string, offer: RTCSessionDescriptionInit) => void;
  onAnswer: (peerId: string, answer: RTCSessionDescriptionInit) => void;
  onIceCandidate: (peerId: string, candidate: RTCIceCandidateInit) => void;
  onPeerJoined: (peerId: string, displayName: string) => void;
  onPeerLeft: (peerId: string) => void;
  onASLToggle: (peerId: string, enabled: boolean) => void;
  onMediaToggle?: (peerId: string, type: 'audio' | 'video', enabled: boolean) => void;
  onTextMessage?: (peerId: string, text: string, displayName?: string) => void;
}

export function useSignaling({
  roomId,
  peerId,
  displayName,
  onOffer,
  onAnswer,
  onIceCandidate,
  onPeerJoined,
  onPeerLeft,
  onASLToggle,
  onMediaToggle,
  onTextMessage,
}: UseSignalingProps) {
  const socketRef = useRef<Socket | null>(null);

  // primary server: using [IP_ADDRESS]
  // fallback server: http://localhost:3001

  useEffect(() => {
    const SOCKET_URL = process.env.NEXT_PUBLIC_SOCKET_URL || 'http://localhost:3001';

    socketRef.current = io(SOCKET_URL, {
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    const socket = socketRef.current;

    socket.on('connect', () => {
      console.log('Socket connected:', socket.id);
      socket.emit('join-room', { roomId, peerId, displayName });
    });

    socket.on('peer-joined', ({ peerId: remotePeerId, displayName: remoteDisplayName }) => {
      console.log('Peer joined:', remotePeerId);
      onPeerJoined(remotePeerId, remoteDisplayName);
    });

    socket.on('peer-left', ({ peerId: remotePeerId }) => {
      console.log('Peer left:', remotePeerId);
      onPeerLeft(remotePeerId);
    });

    socket.on('offer', ({ peerId: remotePeerId, offer }) => {
      console.log('Received offer from:', remotePeerId);
      onOffer(remotePeerId, offer);
    });

    socket.on('answer', ({ peerId: remotePeerId, answer }) => {
      console.log('Received answer from:', remotePeerId);
      onAnswer(remotePeerId, answer);
    });

    socket.on('ice-candidate', ({ peerId: remotePeerId, candidate }) => {
      console.log('Received ICE candidate from:', remotePeerId);
      onIceCandidate(remotePeerId, candidate);
    });

    socket.on('asl-toggle', ({ peerId: remotePeerId, enabled }) => {
      console.log('ASL toggle from:', remotePeerId, enabled);
      onASLToggle(remotePeerId, enabled);
    });

    socket.on('media-toggle', ({ peerId: remotePeerId, type, enabled }) => {
      console.log('Media toggle from:', remotePeerId, type, enabled);
      onMediaToggle?.(remotePeerId, type, enabled);
    });

    socket.on('text-message', ({ peerId: remotePeerId, text, displayName: remoteDisplayName }) => {
      console.log('Text message from:', remotePeerId, text);
      onTextMessage?.(remotePeerId, text, remoteDisplayName);
    });

    socket.on('disconnect', () => {
      console.log('Socket disconnected');
    });

    return () => {
      socket.disconnect();
    };
  }, [roomId, peerId, displayName, onOffer, onAnswer, onIceCandidate, onPeerJoined, onPeerLeft, onASLToggle, onMediaToggle, onTextMessage]);

  const sendOffer = useCallback(
    (targetPeerId: string, offer: RTCSessionDescriptionInit) => {
      socketRef.current?.emit('offer', {
        roomId,
        peerId,
        targetPeerId,
        offer,
      });
    },
    [roomId, peerId]
  );

  const sendAnswer = useCallback(
    (targetPeerId: string, answer: RTCSessionDescriptionInit) => {
      socketRef.current?.emit('answer', {
        roomId,
        peerId,
        targetPeerId,
        answer,
      });
    },
    [roomId, peerId]
  );

  const sendIceCandidate = useCallback(
    (targetPeerId: string, candidate: RTCIceCandidateInit) => {
      socketRef.current?.emit('ice-candidate', {
        roomId,
        peerId,
        targetPeerId,
        candidate,
      });
    },
    [roomId, peerId]
  );

  const notifyASLToggle = useCallback(
    (enabled: boolean) => {
      socketRef.current?.emit('asl-toggle', {
        roomId,
        peerId,
        enabled,
      });
    },
    [roomId, peerId]
  );

  const notifyMediaToggle = useCallback(
    (type: 'audio' | 'video', enabled: boolean) => {
      socketRef.current?.emit('media-toggle', {
        roomId,
        peerId,
        type,
        enabled,
      });
    },
    [roomId, peerId]
  );

  const sendTextMessage = useCallback(
    (text: string) => {
      socketRef.current?.emit('text-message', {
        roomId,
        peerId,
        displayName,
        text,
      });
    },
    [roomId, peerId, displayName]
  );

  const leaveRoom = useCallback(() => {
    socketRef.current?.emit('leave-room', { roomId, peerId });
    socketRef.current?.disconnect();
  }, [roomId, peerId]);

  return {
    sendOffer,
    sendAnswer,
    sendIceCandidate,
    notifyASLToggle,
    notifyMediaToggle,
    sendTextMessage,
    leaveRoom,
  };
}
