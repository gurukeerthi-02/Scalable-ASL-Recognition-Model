export interface Room {
  id: string;
  name: string;
  created_at: string;
  max_participants: number;
  is_active: boolean;
}

export interface Participant {
  id: string;
  room_id: string;
  peer_id: string;
  display_name: string;
  joined_at: string;
  left_at?: string;
  is_asl_enabled: boolean;
}

export interface CallLog {
  id: string;
  room_id: string;
  participant_id?: string;
  event_type: string;
  event_data?: Record<string, any>;
  created_at: string;
}

export interface PeerConnection {
  peerId: string;
  connection: RTCPeerConnection;
  stream?: MediaStream;
  displayName: string;
  isAslEnabled: boolean;
  handRaised: boolean;
  audioEnabled?: boolean;
  videoEnabled?: boolean;
}

export interface SignalingMessage {
  type: 'offer' | 'answer' | 'ice-candidate' | 'join' | 'leave' | 'asl-toggle';
  roomId: string;
  peerId: string;
  displayName?: string;
  data?: any;
}

export interface ASLRecognitionResponse {
  text: string;
  confidence: number;
  mode?: number;
  motion?: number;
  buffer_size?: number;
  stable_count?: number;
}

export interface MediaControls {
  audioEnabled: boolean;
  videoEnabled: boolean;
  aslEnabled: boolean;
}
