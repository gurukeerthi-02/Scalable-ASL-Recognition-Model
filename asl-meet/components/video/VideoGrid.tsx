'use client';

import { VideoTile } from './VideoTile';
import { PeerConnection } from '@/types';

interface VideoGridProps {
  localStream: MediaStream | null;
  localDisplayName: string;
  audioEnabled: boolean;
  videoEnabled: boolean;
  isAslEnabled: boolean;
  handRaised?: boolean;
  peers: Map<string, PeerConnection>;
  onLocalVideoReady?: (videoElement: HTMLVideoElement) => void;
}

export function VideoGrid({
  localStream,
  localDisplayName,
  audioEnabled,
  videoEnabled,
  isAslEnabled,
  handRaised = false,
  peers,
  onLocalVideoReady,
}: VideoGridProps) {
  const totalParticipants = 1 + peers.size;

  const getGridClass = () => {
    if (totalParticipants === 1) {
      // Single participant - full screen
      return 'grid-cols-1';
    } else if (totalParticipants === 2) {
      // Two participants - side by side on desktop, stacked on mobile
      return 'grid-cols-1 lg:grid-cols-2';
    } else if (totalParticipants === 3) {
      // Three participants - responsive layout
      return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3';
    } else if (totalParticipants === 4) {
      // Four participants - 2x2 grid
      return 'grid-cols-2 lg:grid-cols-2';
    } else if (totalParticipants <= 6) {
      // 5-6 participants - 2 columns on mobile, 3 on desktop
      return 'grid-cols-2 lg:grid-cols-3';
    } else if (totalParticipants <= 9) {
      // 7-9 participants - 2 columns on mobile, 3 on tablet, 3 on desktop
      return 'grid-cols-2 md:grid-cols-3';
    } else {
      // 10+ participants - 2 columns on mobile, 3 on tablet, 4 on desktop
      return 'grid-cols-2 md:grid-cols-3 lg:grid-cols-4';
    }
  };

  const getContainerClass = () => {
    // Adjust padding and gap based on number of participants
    if (totalParticipants === 1) {
      return 'p-4 md:p-8 gap-0';
    } else if (totalParticipants <= 4) {
      return 'p-3 md:p-6 gap-3 md:gap-4';
    } else {
      return 'p-2 md:p-4 gap-2 md:gap-3';
    }
  };

  return (
    <div className={`grid w-full h-full content-center items-center ${getGridClass()} ${getContainerClass()}`}>
      {/* Local Video */}
      <VideoTile
        stream={localStream}
        displayName={localDisplayName}
        isLocal={true}
        isPrimary={totalParticipants === 1}
        audioEnabled={audioEnabled}
        videoEnabled={videoEnabled}
        isAslEnabled={isAslEnabled}
        handRaised={handRaised}
        onVideoElementReady={onLocalVideoReady}
      />

      {/* Remote Participants */}
      {Array.from(peers.values()).map((peer) => {
        return (
          <VideoTile
            key={peer.peerId}
            stream={peer.stream || null}
            displayName={peer.displayName}
            isLocal={false}
            audioEnabled={peer.audioEnabled ?? true}
            videoEnabled={peer.videoEnabled ?? true}
            isAslEnabled={peer.isAslEnabled}
            handRaised={peer.handRaised}
          />
        );
      })}
    </div>
  );
}