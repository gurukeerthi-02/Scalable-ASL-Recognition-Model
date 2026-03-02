# Architecture Documentation

This document provides a deep technical overview of the ASL Video Call application architecture.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Browser                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐       │
│  │   Next.js App  │  │  WebRTC Peers  │  │  Socket.IO     │       │
│  │   (React)      │◄─┤  (P2P Streams) ├─►│  Client        │       │
│  └────────────────┘  └────────────────┘  └────────────────┘       │
│         │                                         │                  │
└─────────┼─────────────────────────────────────────┼─────────────────┘
          │                                         │
          │ HTTP/WS (ASL Frames)                    │ WS (Signaling)
          │                                         │
          ▼                                         ▼
┌─────────────────────┐                 ┌─────────────────────┐
│  Flask ASL Backend  │                 │  Socket.IO Server   │
│  - MediaPipe        │                 │  - Room Management  │
│  - TensorFlow       │                 │  - SDP Exchange     │
│  - Gesture Model    │                 │  - ICE Negotiation  │
└─────────────────────┘                 └─────────────────────┘
          │
          │ Recognition Results
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Supabase PostgreSQL                             │
│  - Rooms                                                             │
│  - Participants                                                      │
│  - Call Logs                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Next.js Frontend

**Technology Stack**:
- Next.js 13 (App Router)
- React 18 (Client Components)
- TypeScript
- Tailwind CSS

**Key Responsibilities**:
- User interface rendering
- WebRTC peer connection management
- Socket.IO signaling client
- ASL frame capture and streaming
- Text-to-speech synthesis
- State management

**Component Hierarchy**:
```
app/
├── page.tsx (Landing Page)
│   ├── Room creation form
│   ├── Room join form
│   └── Feature showcase
│
└── call/[roomId]/page.tsx (Call Page)
    ├── VideoGrid
    │   └── VideoTile[] (1-4 participants)
    ├── CallControls
    │   ├── Audio toggle
    │   ├── Video toggle
    │   ├── ASL toggle
    │   └── Leave button
    └── ASLIndicator
        ├── Recognition status
        └── Frame counter
```

### 2. WebRTC Architecture

**Peer Connection Flow**:

```
User A                    Socket.IO Server              User B
  │                              │                         │
  ├─ join-room ────────────────►│                         │
  │                              ├─ peer-joined ─────────►│
  │                              │                         │
  ├─ create offer ──────────────┤                         │
  ├─ set local description      │                         │
  ├─ offer ────────────────────►│                         │
  │                              ├─ offer ───────────────►│
  │                              │                         ├─ set remote desc
  │                              │                         ├─ create answer
  │                              │                         ├─ set local desc
  │                              │◄─ answer ──────────────┤
  │◄─ answer ────────────────────┤                         │
  ├─ set remote description     │                         │
  │                              │                         │
  ├─ ICE candidate ─────────────►│                         │
  │                              ├─ ICE candidate ────────►│
  │                              │                         ├─ add candidate
  │◄─ ICE candidate ─────────────┤◄─ ICE candidate ───────┤
  ├─ add candidate              │                         │
  │                              │                         │
  ├───────────────── P2P Media Stream ──────────────────►│
  │◄──────────────── P2P Media Stream ─────────────────────┤
```

**ICE Configuration**:
- **STUN Servers**: Google public STUN servers for NAT traversal
- **No TURN**: Currently no TURN server (can be added for strict NATs)

**Media Constraints**:
```typescript
{
  audio: true,
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: 'user'
  }
}
```

### 3. Socket.IO Signaling

**Server Architecture** (`server/socket-server.js`):

```javascript
// Room Structure
Map<roomId, Map<peerId, { socketId, displayName }>>

// Events Handled:
- join-room      // Peer joins a room
- offer          // WebRTC offer (SDP)
- answer         // WebRTC answer (SDP)
- ice-candidate  // ICE candidate for NAT traversal
- asl-toggle     // ASL mode enabled/disabled
- leave-room     // Peer leaves room
- disconnect     // Socket disconnection
```

**Event Flow**:

1. **Join Room**:
   - Client emits `join-room` with `{ roomId, peerId, displayName }`
   - Server adds peer to room map
   - Server broadcasts `peer-joined` to all existing peers
   - Existing peers initiate offer to new peer

2. **Offer/Answer Exchange**:
   - Peer A creates offer and emits to server
   - Server forwards offer to Peer B
   - Peer B creates answer and emits to server
   - Server forwards answer to Peer A

3. **ICE Candidates**:
   - Each peer discovers ICE candidates
   - Candidates sent to server
   - Server forwards to target peer
   - Target peer adds candidates to connection

### 4. ASL Recognition Pipeline

**Frame Capture Flow**:

```
Local Video Element
        │
        │ captureVideoFrame() @ 10 FPS
        ▼
Canvas (HTML5)
        │
        │ toDataURL('image/jpeg', 0.8)
        ▼
Base64 JPEG String
        │
        │ WebSocket / HTTP POST
        ▼
Flask Backend
        │
        │ 1. Decode base64
        │ 2. Convert to OpenCV Mat
        │ 3. MediaPipe hand landmarks
        │ 4. Feature extraction
        │ 5. TensorFlow model inference
        │ 6. Gesture classification
        ▼
{ text: "HELLO", confidence: 0.92 }
        │
        │ WebSocket / HTTP Response
        ▼
Next.js Frontend
        │
        │ useTTS hook
        ▼
Browser SpeechSynthesis API
        │
        ▼
Audio Output (Remote Peers Only)
```

**Frame Throttling**:
- **FPS**: 10 frames per second (configurable)
- **Compression**: JPEG quality 80%
- **Resolution**: Matches video stream (1280x720)
- **Bandwidth**: ~50-100 KB/s per frame

**ASL Backend Interface**:

WebSocket Endpoint:
```
ws://localhost:5000/asl-stream

// Send:
{
  "frame": "data:image/jpeg;base64,/9j/4AAQ..."
}

// Receive:
{
  "text": "HELLO",
  "confidence": 0.92
}
```

HTTP Endpoint:
```
POST /recognize
Content-Type: application/json

{
  "frame": "data:image/jpeg;base64,/9j/4AAQ..."
}

Response:
{
  "text": "HELLO",
  "confidence": 0.92
}
```

### 5. Custom Hooks Architecture

#### useWebRTC

**Purpose**: Manages WebRTC peer connections and local media stream

**State**:
- `localStream`: MediaStream | null
- `peers`: Map<peerId, PeerConnection>
- `audioEnabled`: boolean
- `videoEnabled`: boolean

**Methods**:
- `toggleAudio()`: Enable/disable local audio track
- `toggleVideo()`: Enable/disable local video track
- `notifyASLToggle(enabled)`: Notify peers of ASL mode change
- `leaveRoom()`: Close all connections and leave

**Lifecycle**:
```typescript
useEffect(() => {
  // 1. Get user media (camera + mic)
  getUserMedia() → setLocalStream()

  // 2. Cleanup on unmount
  return () => {
    stopMediaStream(localStream)
    peers.forEach(peer => peer.connection.close())
    leaveRoom()
  }
}, [])
```

#### useSignaling

**Purpose**: Socket.IO client for WebRTC signaling

**Props**:
- `roomId`: string
- `peerId`: string
- `displayName`: string
- `onOffer`: (peerId, offer) => void
- `onAnswer`: (peerId, answer) => void
- `onIceCandidate`: (peerId, candidate) => void
- `onPeerJoined`: (peerId, displayName) => void
- `onPeerLeft`: (peerId) => void
- `onASLToggle`: (peerId, enabled) => void

**Methods**:
- `sendOffer(targetPeerId, offer)`
- `sendAnswer(targetPeerId, answer)`
- `sendIceCandidate(targetPeerId, candidate)`
- `notifyASLToggle(enabled)`
- `leaveRoom()`

**Event Handlers**:
```typescript
socket.on('peer-joined', ({ peerId, displayName }) => {
  // Trigger offer creation to new peer
  onPeerJoined(peerId, displayName)
})

socket.on('offer', ({ peerId, offer }) => {
  // Handle incoming offer, create answer
  onOffer(peerId, offer)
})
```

#### useASLStream

**Purpose**: Captures video frames and streams to ASL backend

**State**:
- `isStreaming`: boolean

**Props**:
- `enabled`: boolean
- `videoElement`: HTMLVideoElement | null
- `onRecognition`: (result) => void
- `fps`: number (default: 10)

**Lifecycle**:
```typescript
useEffect(() => {
  if (!enabled || !videoElement) return

  // 1. Create ASL client
  const client = new ASLClient(backendUrl)
  client.connect(onRecognition)

  // 2. Start frame capture interval
  const interval = setInterval(() => {
    const frame = captureVideoFrame(videoElement)
    client.sendFrame(frame)
  }, 1000 / fps)

  // 3. Cleanup
  return () => {
    client.disconnect()
    clearInterval(interval)
  }
}, [enabled, videoElement])
```

#### useTTS

**Purpose**: Text-to-speech using browser SpeechSynthesis API

**State**:
- `isSpeaking`: boolean
- `isSupported`: boolean

**Props**:
- `enabled`: boolean
- `rate`: number (0.1 - 10, default: 1.0)
- `pitch`: number (0 - 2, default: 1.0)
- `volume`: number (0 - 1, default: 1.0)

**Methods**:
- `speak(text)`: Synthesize and play audio
- `stop()`: Cancel current speech
- `pause()`: Pause current speech
- `resume()`: Resume paused speech

**Implementation**:
```typescript
const speak = (text: string) => {
  const synth = window.speechSynthesis
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.rate = rate
  utterance.pitch = pitch
  utterance.volume = volume
  synth.speak(utterance)
}
```

### 6. Database Schema

**Entity Relationship Diagram**:

```
┌─────────────────┐
│     rooms       │
├─────────────────┤
│ id (PK)         │
│ name            │
│ created_at      │
│ max_participants│
│ is_active       │
└─────────────────┘
         │
         │ 1:N
         ▼
┌─────────────────┐
│  participants   │
├─────────────────┤
│ id (PK)         │
│ room_id (FK)    │───┐
│ peer_id         │   │
│ display_name    │   │
│ joined_at       │   │
│ left_at         │   │
│ is_asl_enabled  │   │
└─────────────────┘   │
         │            │
         │ 1:N        │
         ▼            │
┌─────────────────┐   │
│   call_logs     │   │
├─────────────────┤   │
│ id (PK)         │   │
│ room_id (FK)    │───┘
│ participant_id  │
│ event_type      │
│ event_data      │
│ created_at      │
└─────────────────┘
```

**Row Level Security (RLS)**:

All tables have RLS enabled with public access policies suitable for demo/research:

```sql
-- Example policy
CREATE POLICY "Anyone can view rooms"
  ON rooms FOR SELECT
  TO anon, authenticated
  USING (true);
```

**Production RLS** (recommended):

```sql
-- Restrict room access to participants
CREATE POLICY "Users can view their rooms"
  ON rooms FOR SELECT
  TO authenticated
  USING (
    id IN (
      SELECT room_id FROM participants
      WHERE peer_id = current_setting('app.peer_id')
    )
  );
```

## Data Flow Scenarios

### Scenario 1: User Creates Room

```
1. User fills form on landing page
   ├─ Display name: "Alice"
   └─ Room name: "Meeting"

2. Click "Create Room"
   ├─ supabase.from('rooms').insert({
   │    name: "Meeting",
   │    max_participants: 4,
   │    is_active: true
   │  })
   └─ Returns: { id: "abc-123-..." }

3. Navigate to /call/abc-123-...?name=Alice
   ├─ useWebRTC hook initializes
   │  ├─ Generate peerId: "peer-xyz"
   │  ├─ getUserMedia() → localStream
   │  └─ Connect to Socket.IO server
   │
   └─ Insert participant record
      └─ supabase.from('participants').insert({
           room_id: "abc-123-...",
           peer_id: "peer-xyz",
           display_name: "Alice",
           is_asl_enabled: false
         })

4. Display video grid with local stream
```

### Scenario 2: Second User Joins

```
1. User enters Room ID on landing page
   ├─ Display name: "Bob"
   └─ Room ID: "abc-123-..."

2. Verify room exists
   ├─ supabase.from('rooms')
   │    .select('*')
   │    .eq('id', 'abc-123-...')
   └─ Check participant count < max_participants

3. Navigate to /call/abc-123-...?name=Bob
   ├─ Generate peerId: "peer-uvw"
   └─ Connect to Socket.IO

4. Socket.IO handshake
   ├─ Bob emits: join-room
   ├─ Server notifies Alice: peer-joined
   └─ Alice creates offer to Bob

5. WebRTC negotiation
   ├─ Alice → offer → Server → Bob
   ├─ Bob → answer → Server → Alice
   └─ ICE candidates exchanged

6. P2P connection established
   └─ Both see each other's video
```

### Scenario 3: ASL Mode Activated

```
1. Alice clicks ASL button
   ├─ setAslEnabled(true)
   ├─ notifyASLToggle(true)
   └─ Socket.IO broadcasts to Bob

2. useASLStream hook activates
   ├─ Connect to Flask backend WebSocket
   └─ Start frame capture interval (10 FPS)

3. Frame capture loop
   ├─ Every 100ms:
   │  ├─ captureVideoFrame(videoElement)
   │  ├─ Convert to base64 JPEG
   │  └─ Send to Flask backend
   │
   └─ Flask processes frame:
      ├─ Decode base64
      ├─ MediaPipe hand landmarks
      ├─ TensorFlow model inference
      └─ Return: { text: "HELLO", confidence: 0.92 }

4. Recognition received
   ├─ onRecognition({ text: "HELLO", confidence: 0.92 })
   ├─ Update ASLIndicator component
   └─ useTTS.speak("HELLO")

5. Text-to-speech
   ├─ Browser SpeechSynthesis API
   └─ Audio plays for Bob (not Alice)

6. Log event
   └─ supabase.from('call_logs').insert({
        room_id: "abc-123-...",
        participant_id: alice_id,
        event_type: "asl_recognition",
        event_data: { text: "HELLO", confidence: 0.92 }
      })
```

## Performance Optimizations

### 1. Frame Capture Throttling

**Problem**: Capturing every frame is CPU-intensive and bandwidth-heavy.

**Solution**:
```typescript
// Throttle to 10 FPS instead of 30 FPS
const captureInterval = 1000 / 10; // 100ms

setInterval(() => {
  const frame = captureVideoFrame(videoElement);
  client.sendFrame(frame);
}, captureInterval);
```

**Result**: 67% reduction in CPU usage and bandwidth.

### 2. JPEG Compression

**Problem**: PNG frames are 500KB - 1MB each.

**Solution**:
```typescript
canvas.toDataURL('image/jpeg', 0.8); // 80% quality
```

**Result**: Frames reduced to 50-100 KB each.

### 3. Peer Connection Reuse

**Problem**: Creating new peer connections for each event is expensive.

**Solution**:
```typescript
// Store peer connections in a map
const peers = new Map<peerId, RTCPeerConnection>();

// Reuse existing connection
if (!peers.has(peerId)) {
  peers.set(peerId, createPeerConnection());
}
```

### 4. Socket.IO Room Isolation

**Problem**: Broadcasting to all clients wastes bandwidth.

**Solution**:
```javascript
// Server-side: Only send to specific room
socket.to(roomId).emit('peer-joined', { peerId });
```

### 5. Lazy Component Loading

**Problem**: Large initial bundle size.

**Solution**:
```typescript
// Dynamic import for call page
const CallPage = dynamic(() => import('./call/[roomId]/page'));
```

## Security Considerations

### 1. Database Security

**Current**: Public access RLS policies for demo
**Production**: Restrict to authenticated users with proper row-level checks

```sql
-- Example: Restrict participant queries
CREATE POLICY "Users can only see their own data"
  ON participants FOR SELECT
  TO authenticated
  USING (peer_id = current_setting('app.peer_id'));
```

### 2. WebRTC Security

**Current**: No authentication on peer connections
**Production**: Implement credential-based authentication

```typescript
const configuration = {
  iceServers: [
    {
      urls: 'turn:turn.example.com:3478',
      username: 'user',
      credential: 'password',
    },
  ],
};
```

### 3. Socket.IO Security

**Current**: No authentication on signaling messages
**Production**: Implement JWT-based authentication

```typescript
// Client
const socket = io(socketUrl, {
  auth: {
    token: jwt_token,
  },
});

// Server
io.use((socket, next) => {
  const token = socket.handshake.auth.token;
  if (verifyToken(token)) {
    next();
  } else {
    next(new Error('Authentication error'));
  }
});
```

### 4. ASL Backend Security

**Current**: No authentication on API requests
**Production**: Implement API key authentication

```typescript
// Client
fetch(aslBackendUrl, {
  headers: {
    'X-API-Key': process.env.NEXT_PUBLIC_ASL_API_KEY,
  },
});
```

## Scalability Considerations

### Horizontal Scaling

**Socket.IO Server**:
- Use Redis adapter for multiple server instances
- Implement sticky sessions for WebSocket connections

**Flask Backend**:
- Use gunicorn with multiple workers
- Implement load balancer (nginx)
- Consider GPU acceleration for model inference

### Vertical Scaling

**Database**:
- Supabase automatically handles scaling
- Add read replicas if needed

**Frontend**:
- Deploy to CDN (Vercel, Netlify)
- Implement edge caching

## Future Enhancements

### 1. End-to-End Encryption

Implement using Insertable Streams API:

```typescript
const sender = peerConnection.addTrack(track, stream);
const streams = sender.createEncodedStreams();

streams.readableStream
  .pipeThrough(new TransformStream({
    transform: encryptFrame,
  }))
  .pipeTo(streams.writableStream);
```

### 2. Simulcast

Enable multiple quality streams:

```typescript
sender.setParameters({
  encodings: [
    { rid: 'h', maxBitrate: 900000 },
    { rid: 'm', maxBitrate: 300000, scaleResolutionDownBy: 2 },
    { rid: 'l', maxBitrate: 100000, scaleResolutionDownBy: 4 },
  ],
});
```

### 3. SFU Architecture

Replace mesh P2P with Selective Forwarding Unit:

```
User A ──┐
User B ──┼──► SFU ──┬──► User C
User D ──┘           └──► User E
```

Benefits:
- Better scalability (>4 participants)
- Lower client bandwidth usage
- Server-side transcoding/recording

## Conclusion

This architecture provides a solid foundation for a real-time video calling application with ASL recognition. The modular design allows for easy extension and modification while maintaining clear separation of concerns.
