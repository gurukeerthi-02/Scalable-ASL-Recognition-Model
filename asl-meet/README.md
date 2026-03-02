# ASL Video Call Application

A real-time video calling application with integrated ASL (American Sign Language) recognition, built with Next.js, WebRTC, and Socket.IO.

## Features

- **Peer-to-Peer Video Calling**: WebRTC-based video and audio streaming
- **Up to 4 Participants**: Each room supports a maximum of 4 simultaneous participants
- **ASL Recognition Integration**: Real-time sign language recognition with text-to-speech conversion
- **Low Latency**: Direct peer-to-peer connections for minimal delay
- **Persistent Room Management**: Supabase database for room and participant tracking

## Architecture

### Frontend (Next.js)
- **Framework**: Next.js 13 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI primitives (via shadcn/ui)

### Backend Services
1. **Supabase**: PostgreSQL database for room management and call logging
2. **Socket.IO Server**: WebRTC signaling server (port 3001)
3. **Flask ASL Backend**: Python-based ASL recognition service (port 5000) - **NOT included, must be running separately**

### WebRTC Architecture
- **Signaling**: Socket.IO for SDP exchange and ICE candidate negotiation
- **Media**: Peer-to-peer video/audio streams
- **STUN Servers**: Google STUN servers for NAT traversal

## Project Structure

```
/app
  /call/[roomId]
    page.tsx          # Main video call page
  page.tsx            # Landing page (create/join rooms)
  layout.tsx
  globals.css

/components
  /video
    VideoGrid.tsx     # Grid layout for video tiles
    VideoTile.tsx     # Individual video tile component
    CallControls.tsx  # Mute, video, ASL, and leave buttons
    ASLIndicator.tsx  # ASL mode status indicator
  /ui                 # shadcn/ui components

/hooks
  useWebRTC.ts        # WebRTC peer connection management
  useSignaling.ts     # Socket.IO signaling logic
  useASLStream.ts     # ASL frame capture and streaming
  useTTS.ts           # Text-to-speech synthesis

/lib
  webrtc.ts           # WebRTC utilities (ICE config, media capture)
  asl-client.ts       # ASL backend API client
  supabase.ts         # Supabase client configuration

/types
  index.ts            # TypeScript type definitions

/server
  socket-server.js    # Socket.IO signaling server
```

## Database Schema

### Tables

**rooms**
- `id` (uuid): Unique room identifier
- `name` (text): Optional room name
- `created_at` (timestamptz): Room creation timestamp
- `max_participants` (integer): Maximum allowed participants (default: 4)
- `is_active` (boolean): Whether room is currently active

**participants**
- `id` (uuid): Unique participant identifier
- `room_id` (uuid): Reference to room
- `peer_id` (text): WebRTC peer identifier
- `display_name` (text): Participant display name
- `joined_at` (timestamptz): When participant joined
- `left_at` (timestamptz, nullable): When participant left
- `is_asl_enabled` (boolean): Whether ASL mode is active

**call_logs**
- `id` (uuid): Unique log identifier
- `room_id` (uuid): Reference to room
- `participant_id` (uuid, nullable): Reference to participant
- `event_type` (text): Event type (join, leave, asl_on, asl_off, asl_recognition)
- `event_data` (jsonb, nullable): Additional event metadata
- `created_at` (timestamptz): Event timestamp

## Setup Instructions

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment Variables

Create a `.env.local` file based on `.env.local.example`:

```bash
cp .env.local.example .env.local
```

Fill in your Supabase credentials:
- `NEXT_PUBLIC_SUPABASE_URL`: Your Supabase project URL
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`: Your Supabase anonymous key

Optional (defaults shown):
- `NEXT_PUBLIC_SOCKET_URL`: Socket.IO server URL (default: http://localhost:3001)
- `NEXT_PUBLIC_ASL_BACKEND_URL`: ASL recognition backend URL (default: http://localhost:5000)

### 3. Database Setup

The database schema is already applied via Supabase migrations. Verify tables exist:
- `rooms`
- `participants`
- `call_logs`

### 4. Start the Socket.IO Signaling Server

In a separate terminal:

```bash
npm run socket-server
```

This starts the WebRTC signaling server on port 3001.

### 5. Start the ASL Recognition Backend

**Note**: The ASL recognition backend is NOT included in this repository. You must have a separate Flask application running that provides:

- **WebSocket endpoint**: `ws://localhost:5000/asl-stream`
  - Accepts video frames as base64-encoded JPEG images
  - Returns JSON: `{ "text": "HELLO", "confidence": 0.92 }`

- **HTTP endpoint**: `POST http://localhost:5000/recognize`
  - Accepts JSON: `{ "frame": "data:image/jpeg;base64,..." }`
  - Returns JSON: `{ "text": "HELLO", "confidence": 0.92 }`

### 6. Start the Next.js Development Server

```bash
npm run dev
```

The application will be available at `http://localhost:3000`.

## Usage

### Creating a Room

1. Navigate to the landing page (`http://localhost:3000`)
2. Enter your display name
3. Optionally enter a room name
4. Click "Create Room"
5. Share the Room ID with other participants (visible in the URL)

### Joining a Room

1. Navigate to the landing page
2. Enter your display name
3. Enter the Room ID
4. Click "Join Room"

### During a Call

**Basic Controls**:
- **Microphone**: Toggle audio on/off
- **Camera**: Toggle video on/off
- **Leave**: Exit the call and return to landing page

**ASL Mode**:
1. Click the "Hand" button to enable ASL Mode
2. Your local video frames are sent to the ASL backend at 10 FPS
3. Recognized gestures appear in the ASL Indicator panel
4. Text is converted to speech using the browser's Speech Synthesis API
5. Speech is played ONLY to other participants, not to you

## Technical Details

### WebRTC Flow

1. **Join Room**: Client connects to Socket.IO server
2. **Get User Media**: Request camera and microphone access
3. **Peer Discovery**: Socket.IO notifies about existing peers
4. **Create Offers**: Initiate WebRTC connections to all peers
5. **Exchange SDP**: Offer/Answer exchange via Socket.IO
6. **ICE Negotiation**: Exchange ICE candidates for NAT traversal
7. **Connected**: Direct peer-to-peer media streaming

### ASL Recognition Flow

1. **Enable ASL Mode**: User clicks ASL button
2. **Frame Capture**: Extract frames from local video element at 10 FPS
3. **Base64 Encoding**: Convert frames to JPEG and encode as base64
4. **WebSocket Streaming**: Send frames to Flask backend
5. **Recognition**: Backend processes frames and returns text + confidence
6. **Text-to-Speech**: Convert recognized text to audio
7. **Audio Playback**: Play audio to remote participants only

### Custom Hooks

**useWebRTC**
- Manages local media stream
- Creates and maintains peer connections
- Handles offer/answer/ICE candidate exchange
- Provides audio/video toggle functions

**useSignaling**
- Establishes Socket.IO connection
- Handles room join/leave events
- Sends and receives signaling messages
- Notifies about peer events

**useASLStream**
- Captures video frames at configurable FPS
- Manages WebSocket connection to ASL backend
- Throttles frame transmission
- Triggers recognition callback

**useTTS**
- Wraps browser Speech Synthesis API
- Configurable rate, pitch, and volume
- Handles speech queue management

## Performance Considerations

- **Frame Throttling**: ASL frames limited to 10 FPS to reduce bandwidth and processing load
- **Video Quality**: Ideal resolution 1280x720, adjustable based on network conditions
- **Peer Limit**: Maximum 4 participants to maintain quality and performance
- **JPEG Compression**: Frames compressed to 80% quality before transmission

## Browser Compatibility

- **Chrome/Edge**: Full support
- **Firefox**: Full support
- **Safari**: Full support (iOS requires HTTPS)

**Required APIs**:
- WebRTC (RTCPeerConnection)
- getUserMedia
- WebSocket
- Speech Synthesis API (for TTS)

## Known Limitations

1. **ASL Backend Required**: The Flask ASL recognition service must be running separately
2. **STUN Only**: No TURN server for strict NAT/firewall scenarios
3. **No Recording**: Call recording not implemented
4. **No Chat**: Text chat not implemented
5. **No Screen Sharing**: Limited to camera video only

## Development

### Type Checking

```bash
npm run typecheck
```

### Build Production

```bash
npm run build
```

### Linting

```bash
npm run lint
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_SUPABASE_URL` | Yes | - | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Yes | - | Supabase anonymous key |
| `NEXT_PUBLIC_SOCKET_URL` | No | `http://localhost:3001` | Socket.IO server URL |
| `NEXT_PUBLIC_ASL_BACKEND_URL` | No | `http://localhost:5000` | ASL recognition backend URL |

## Security Considerations

- **Database**: Row Level Security (RLS) enabled with public access policies (suitable for demo/research)
- **WebRTC**: No authentication on peer connections (add TURN auth for production)
- **ASL Backend**: No authentication (add API keys for production)
- **Room Access**: No password protection (add for production)

## Future Enhancements

- [ ] Add TURN server support for better connectivity
- [ ] Implement end-to-end encryption
- [ ] Add text chat feature
- [ ] Add screen sharing
- [ ] Add call recording
- [ ] Add room password protection
- [ ] Add admin controls for room management
- [ ] Improve ASL recognition accuracy display
- [ ] Add gesture history panel
- [ ] Add multi-language support

## License

MIT

## Credits

Built with Next.js, WebRTC, Socket.IO, Supabase, and shadcn/ui.
