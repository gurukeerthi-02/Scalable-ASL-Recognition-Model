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

  const handlePeerJoined = useCallback(async (remotePeerId: string, remoteDisplayName: string) => {
    console.log('Creating offer for new peer:', remotePeerId);

    const peerConnection = createPeerConnection();

    if (localStreamRef.current) {
      console.log('Adding local tracks to peer connection for:', remotePeerId);
      console.log('Local stream tracks:', localStreamRef.current.getTracks().map(t => ({ kind: t.kind, enabled: t.enabled })));
      localStreamRef.current.getTracks().forEach((track) => {
        console.log('Adding track:', track.kind, track.enabled);
        peerConnection.addTrack(track, localStreamRef.current!);
      });
    } else {
      console.log('No local stream available when creating peer connection for:', remotePeerId);
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
        sendIceCandidate(remotePeerId, event.candidate.toJSON());
      }
    };

    peerConnection.ontrack = (event) => {
      console.log('Received remote track from:', remotePeerId, event.streams[0]);
      console.log('Track kind:', event.track.kind, 'enabled:', event.track.enabled);
      console.log('Stream tracks:', event.streams[0].getTracks().map(t => ({ kind: t.kind, enabled: t.enabled })));
      peerData.stream = event.streams[0];
      setPeers(new Map(peersRef.current));
    };

    peerConnection.oniceconnectionstatechange = () => {
      console.log(`ICE connection state for ${remotePeerId}:`, peerConnection.iceConnectionState);
      if (peerConnection.iceConnectionState === 'disconnected' ||
        peerConnection.iceConnectionState === 'failed') {
        handlePeerLeft(remotePeerId);
      }
    };

    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    sendOffer(remotePeerId, offer);
  }, []);

  const handleOffer = useCallback(async (remotePeerId: string, offer: RTCSessionDescriptionInit) => {
    console.log('Handling offer from:', remotePeerId);

    let peerData = peersRef.current.get(remotePeerId);

    if (!peerData) {
      const peerConnection = createPeerConnection();

      if (localStreamRef.current) {
        console.log('Adding local tracks to peer connection for:', remotePeerId);
        console.log('Local stream tracks:', localStreamRef.current.getTracks().map(t => ({ kind: t.kind, enabled: t.enabled })));
        localStreamRef.current.getTracks().forEach((track) => {
          console.log('Adding track:', track.kind, track.enabled);
          peerConnection.addTrack(track, localStreamRef.current!);
        });
      } else {
        console.log('No local stream available when handling offer from:', remotePeerId);
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
          sendIceCandidate(remotePeerId, event.candidate.toJSON());
        }
      };

      peerConnection.ontrack = (event) => {
        console.log('Received remote track from:', remotePeerId, event.streams[0]);
        console.log('Track kind:', event.track.kind, 'enabled:', event.track.enabled);
        console.log('Stream tracks:', event.streams[0].getTracks().map(t => ({ kind: t.kind, enabled: t.enabled })));
        peerData!.stream = event.streams[0];
        setPeers(new Map(peersRef.current));
      };

      peerConnection.oniceconnectionstatechange = () => {
        console.log(`ICE connection state for ${remotePeerId}:`, peerConnection.iceConnectionState);
        if (peerConnection.iceConnectionState === 'disconnected' ||
          peerConnection.iceConnectionState === 'failed') {
          handlePeerLeft(remotePeerId);
        }
      };
    }

    await peerData.connection.setRemoteDescription(new RTCSessionDescription(offer));
    const answer = await peerData.connection.createAnswer();
    await peerData.connection.setLocalDescription(answer);
    sendAnswer(remotePeerId, answer);

    setPeers(new Map(peersRef.current));
  }, []);

  const handleAnswer = useCallback(async (remotePeerId: string, answer: RTCSessionDescriptionInit) => {
    const peerData = peersRef.current.get(remotePeerId);
    if (peerData) {
      await peerData.connection.setRemoteDescription(new RTCSessionDescription(answer));
    }
  }, []);

  const handleIceCandidate = useCallback(async (remotePeerId: string, candidate: RTCIceCandidateInit) => {
    const peerData = peersRef.current.get(remotePeerId);
    if (peerData) {
      try {
        await peerData.connection.addIceCandidate(new RTCIceCandidate(candidate));
      } catch (error) {
        console.error('Error adding ICE candidate:', error);
      }
    }
  }, []);

  const handlePeerLeft = useCallback((remotePeerId: string) => {
    const peerData = peersRef.current.get(remotePeerId);
    if (peerData) {
      peerData.connection.close();
      peersRef.current.delete(remotePeerId);
      setPeers(new Map(peersRef.current));
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

  const { sendOffer, sendAnswer, sendIceCandidate, notifyASLToggle, sendTextMessage, leaveRoom } = useSignaling({
    roomId,
    peerId,
    displayName,
    onOffer: handleOffer,
    onAnswer: handleAnswer,
    onIceCandidate: handleIceCandidate,
    onPeerJoined: handlePeerJoined,
    onPeerLeft: handlePeerLeft,
    onASLToggle: handleASLToggle,
    onTextMessage,
  });

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
    }
  }, [audioEnabled]);

  const toggleVideo = useCallback(() => {
    if (localStreamRef.current) {
      const newState = !videoEnabled;
      toggleVideoTrack(localStreamRef.current, newState);
      setVideoEnabled(newState);
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
