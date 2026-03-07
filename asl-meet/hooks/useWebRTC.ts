'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { PeerConnection } from '@/types';
import { createPeerConnection, getUserMedia, toggleAudioTrack, toggleVideoTrack, stopMediaStream } from '@/lib/webrtc';
import { useSignaling } from './useSignaling';

interface UseWebRTCProps {
  roomId: string;
  peerId: string;
  displayName: string;
  onRemoteASLToggle: (peerId: string, enabled: boolean) => void;
  onTextMessage?: (peerId: string, text: string, displayName?: string) => void;
}

export function useWebRTC({ roomId, peerId, displayName, onRemoteASLToggle, onTextMessage }: UseWebRTCProps) {
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [peers, setPeers] = useState<Map<string, PeerConnection>>(new Map());
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [videoEnabled, setVideoEnabled] = useState(true);

  const peersRef = useRef<Map<string, PeerConnection>>(new Map());
  const localStreamRef = useRef<MediaStream | null>(null);
  const pendingCandidates = useRef<Map<string, RTCIceCandidateInit[]>>(new Map());

  // Refs for signaling functions to avoid circular dependencies
  const sendOfferRef = useRef<((targetPeerId: string, offer: RTCSessionDescriptionInit) => void) | null>(null);
  const sendAnswerRef = useRef<((targetPeerId: string, answer: RTCSessionDescriptionInit) => void) | null>(null);
  const sendIceCandidateRef = useRef<((targetPeerId: string, candidate: RTCIceCandidateInit) => void) | null>(null);
  const notifyMediaToggleRef = useRef<((type: 'audio' | 'video', enabled: boolean) => void) | null>(null);

  const handlePeerLeft = useCallback((remotePeerId: string) => {
    const peerData = peersRef.current.get(remotePeerId);
    if (peerData) {
      peerData.connection.close();
      peersRef.current.delete(remotePeerId);
      pendingCandidates.current.delete(remotePeerId);
      setPeers(new Map(peersRef.current));
    }
  }, []);

  const handlePeerJoined = useCallback(async (remotePeerId: string, remoteDisplayName: string) => {
    console.log('Creating offer for new peer:', remotePeerId);

    const peerConnection = createPeerConnection();

    if (localStreamRef.current) {
      console.log('Adding local tracks to peer connection for:', remotePeerId);
      localStreamRef.current.getTracks().forEach((track) => {
        peerConnection.addTrack(track, localStreamRef.current!);
      });
    }

    const peerData: PeerConnection = {
      peerId: remotePeerId,
      connection: peerConnection,
      displayName: remoteDisplayName,
      isAslEnabled: false,
      handRaised: false,
      audioEnabled: true,
      videoEnabled: true,
    };

    peersRef.current.set(remotePeerId, peerData);
    setPeers(new Map(peersRef.current));

    peerConnection.onicecandidate = (event) => {
      if (event.candidate) {
        sendIceCandidateRef.current?.(remotePeerId, event.candidate.toJSON());
      }
    };

    peerConnection.ontrack = (event) => {
      console.log('Received remote track from:', remotePeerId, event.streams[0]);
      peerData.stream = event.streams[0];
      setPeers(new Map(peersRef.current));
    };

    peerConnection.oniceconnectionstatechange = () => {
      const state = peerConnection.iceConnectionState;
      console.log(`ICE connection state for ${remotePeerId}:`, state);

      if (state === 'failed') {
        handlePeerLeft(remotePeerId);
      }
    };

    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    sendOfferRef.current?.(remotePeerId, offer);
  }, [handlePeerLeft]);

  const handleExistingPeers = useCallback(async (existingPeers: Array<{ peerId: string, displayName: string }>) => {
    existingPeers.forEach(peer => {
      if (!peersRef.current.has(peer.peerId)) {
        handlePeerJoined(peer.peerId, peer.displayName);
      }
    });
  }, [handlePeerJoined]);

  const handleOffer = useCallback(async (remotePeerId: string, offer: RTCSessionDescriptionInit) => {
    console.log('Handling offer from:', remotePeerId);
    let peerData = peersRef.current.get(remotePeerId);

    if (!peerData) {
      const peerConnection = createPeerConnection();
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((track) => {
          peerConnection.addTrack(track, localStreamRef.current!);
        });
      }

      peerData = {
        peerId: remotePeerId,
        connection: peerConnection,
        displayName: 'Remote User',
        isAslEnabled: false,
        handRaised: false,
        audioEnabled: true,
        videoEnabled: true,
      };

      peersRef.current.set(remotePeerId, peerData);

      peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          sendIceCandidateRef.current?.(remotePeerId, event.candidate.toJSON());
        }
      };

      peerConnection.ontrack = (event) => {
        peerData!.stream = event.streams[0];
        setPeers(new Map(peersRef.current));
      };

      peerConnection.oniceconnectionstatechange = () => {
        if (peerConnection.iceConnectionState === 'failed') {
          handlePeerLeft(remotePeerId);
        }
      };
    }

    await peerData.connection.setRemoteDescription(new RTCSessionDescription(offer));
    const answer = await peerData.connection.createAnswer();
    await peerData.connection.setLocalDescription(answer);
    sendAnswerRef.current?.(remotePeerId, answer);

    setPeers(new Map(peersRef.current));

    const pending = pendingCandidates.current.get(remotePeerId) || [];
    for (const candidate of pending) {
      await peerData.connection.addIceCandidate(new RTCIceCandidate(candidate));
    }
    pendingCandidates.current.delete(remotePeerId);
  }, [handlePeerLeft]);

  const handleAnswer = useCallback(async (remotePeerId: string, answer: RTCSessionDescriptionInit) => {
    const peerData = peersRef.current.get(remotePeerId);
    if (peerData) {
      await peerData.connection.setRemoteDescription(new RTCSessionDescription(answer));
      const pending = pendingCandidates.current.get(remotePeerId) || [];
      for (const candidate of pending) {
        await peerData.connection.addIceCandidate(new RTCIceCandidate(candidate));
      }
      pendingCandidates.current.delete(remotePeerId);
    }
  }, []);

  const handleIceCandidate = useCallback(async (remotePeerId: string, candidate: RTCIceCandidateInit) => {
    const peerData = peersRef.current.get(remotePeerId);
    if (peerData && peerData.connection.remoteDescription) {
      try {
        await peerData.connection.addIceCandidate(new RTCIceCandidate(candidate));
      } catch (error) {
        console.error('Error adding ICE candidate:', error);
      }
    } else {
      const pending = pendingCandidates.current.get(remotePeerId) || [];
      pending.push(candidate);
      pendingCandidates.current.set(remotePeerId, pending);
    }
  }, []);

  const handleASLToggle = useCallback((remotePeerId: string, enabled: boolean) => {
    const peerData = peersRef.current.get(remotePeerId);
    if (peerData) {
      peerData.isAslEnabled = enabled;
      setPeers(new Map(peersRef.current));
      onRemoteASLToggle(remotePeerId, enabled);
    }
  }, [onRemoteASLToggle]);

  const handleMediaToggle = useCallback((remotePeerId: string, type: 'audio' | 'video', enabled: boolean) => {
    const peerData = peersRef.current.get(remotePeerId);
    if (peerData) {
      if (type === 'audio') peerData.audioEnabled = enabled;
      if (type === 'video') peerData.videoEnabled = enabled;
      setPeers(new Map(peersRef.current));
    }
  }, []);

  const { sendOffer, sendAnswer, sendIceCandidate, notifyASLToggle, notifyMediaToggle, sendTextMessage, leaveRoom } = useSignaling({
    roomId,
    peerId,
    displayName,
    onOffer: handleOffer,
    onAnswer: handleAnswer,
    onIceCandidate: handleIceCandidate,
    onPeerJoined: handlePeerJoined,
    onExistingPeers: handleExistingPeers,
    onPeerLeft: handlePeerLeft,
    onASLToggle: handleASLToggle,
    onMediaToggle: handleMediaToggle,
    onTextMessage,
  });

  // Sync signaling functions to refs
  useEffect(() => {
    sendOfferRef.current = sendOffer;
    sendAnswerRef.current = sendAnswer;
    sendIceCandidateRef.current = sendIceCandidate;
    notifyMediaToggleRef.current = notifyMediaToggle;
  }, [sendOffer, sendAnswer, sendIceCandidate, notifyMediaToggle]);

  useEffect(() => {
    const initLocalStream = async () => {
      try {
        const stream = await getUserMedia(audioEnabled, videoEnabled);
        setLocalStream(stream);
        localStreamRef.current = stream;
      } catch (error) {
        console.error('Failed to get user media:', error);
      }
    };

    initLocalStream();

    return () => {
      if (localStreamRef.current) {
        stopMediaStream(localStreamRef.current);
      }
      peersRef.current.forEach((peer) => {
        peer.connection.close();
      });
      leaveRoom();
    };
  }, []);

  const toggleAudio = useCallback(() => {
    if (localStreamRef.current) {
      const newState = !audioEnabled;
      toggleAudioTrack(localStreamRef.current, newState);
      setAudioEnabled(newState);
      notifyMediaToggleRef.current?.('audio', newState);
    }
  }, [audioEnabled]);

  const toggleVideo = useCallback(() => {
    if (localStreamRef.current) {
      const newState = !videoEnabled;
      toggleVideoTrack(localStreamRef.current, newState);
      setVideoEnabled(newState);
      notifyMediaToggleRef.current?.('video', newState);
    }
  }, [videoEnabled]);

  return {
    localStream,
    peers,
    audioEnabled,
    videoEnabled,
    toggleAudio,
    toggleVideo,
    notifyASLToggle,
    sendTextMessage,
    leaveRoom,
  };
}
