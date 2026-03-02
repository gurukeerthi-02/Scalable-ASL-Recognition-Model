const { createServer } = require('http');
const { Server } = require('socket.io');

const httpServer = createServer();

const io = new Server(httpServer, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
  },
});

const rooms = new Map();

io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('join-room', ({ roomId, peerId, displayName }) => {
    console.log(`Peer ${peerId} (${displayName}) joining room ${roomId}`);

    socket.join(roomId);

    if (!rooms.has(roomId)) {
      rooms.set(roomId, new Map());
    }

    const roomPeers = rooms.get(roomId);

    // Notify new peer about existing peers
    const existingPeers = Array.from(roomPeers.entries()).map(([id, info]) => ({
      peerId: id,
      displayName: info.displayName,
    }));

    console.log(`Existing peers in room ${roomId}:`, existingPeers);

    // Notify existing peers about new peer
    roomPeers.forEach((peerInfo, existingPeerId) => {
      console.log(`Notifying existing peer ${existingPeerId} about new peer ${peerId}`);
      io.to(peerInfo.socketId).emit('peer-joined', {
        peerId,
        displayName,
      });
    });

    // Add new peer to room
    roomPeers.set(peerId, {
      socketId: socket.id,
      displayName,
    });

    socket.roomId = roomId;
    socket.peerId = peerId;

    console.log(`Room ${roomId} now has ${roomPeers.size} peers`);
  });

  socket.on('offer', ({ roomId, peerId, targetPeerId, offer }) => {
    console.log(`Offer from ${peerId} to ${targetPeerId}`);

    const roomPeers = rooms.get(roomId);
    if (roomPeers) {
      const targetPeer = roomPeers.get(targetPeerId);
      if (targetPeer) {
        io.to(targetPeer.socketId).emit('offer', {
          peerId,
          offer,
        });
      }
    }
  });

  socket.on('answer', ({ roomId, peerId, targetPeerId, answer }) => {
    console.log(`Answer from ${peerId} to ${targetPeerId}`);

    const roomPeers = rooms.get(roomId);
    if (roomPeers) {
      const targetPeer = roomPeers.get(targetPeerId);
      if (targetPeer) {
        io.to(targetPeer.socketId).emit('answer', {
          peerId,
          answer,
        });
      }
    }
  });

  socket.on('ice-candidate', ({ roomId, peerId, targetPeerId, candidate }) => {
    console.log(`ICE candidate from ${peerId} to ${targetPeerId}`);

    const roomPeers = rooms.get(roomId);
    if (roomPeers) {
      const targetPeer = roomPeers.get(targetPeerId);
      if (targetPeer) {
        io.to(targetPeer.socketId).emit('ice-candidate', {
          peerId,
          candidate,
        });
      }
    }
  });

  socket.on('asl-toggle', ({ roomId, peerId, enabled }) => {
    console.log(`ASL toggle from ${peerId}: ${enabled}`);

    socket.to(roomId).emit('asl-toggle', {
      peerId,
      enabled,
    });
  });

  socket.on('text-message', ({ roomId, peerId, text }) => {
    console.log(`Text message from ${peerId}: ${text}`);

    socket.to(roomId).emit('text-message', {
      peerId,
      text,
    });
  });

  socket.on('leave-room', ({ roomId, peerId }) => {
    handlePeerLeave(socket, roomId, peerId);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);

    if (socket.roomId && socket.peerId) {
      handlePeerLeave(socket, socket.roomId, socket.peerId);
    }
  });
});

function handlePeerLeave(socket, roomId, peerId) {
  console.log(`Peer ${peerId} leaving room ${roomId}`);

  const roomPeers = rooms.get(roomId);
  if (roomPeers) {
    roomPeers.delete(peerId);

    socket.to(roomId).emit('peer-left', {
      peerId,
    });

    if (roomPeers.size === 0) {
      rooms.delete(roomId);
      console.log(`Room ${roomId} is now empty and removed`);
    } else {
      console.log(`Room ${roomId} now has ${roomPeers.size} peers`);
    }
  }

  socket.leave(roomId);
}

const PORT = process.env.PORT || 3001;
const HOST = '0.0.0.0'; // Bind to all interfaces

httpServer.listen(PORT, HOST, () => {
  console.log(`Socket.IO signaling server running on ${HOST}:${PORT}`);
});
