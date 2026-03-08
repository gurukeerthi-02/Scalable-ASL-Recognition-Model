'use client';

import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { MicOff, VideoOff, Hand, Pin, Maximize2, User } from 'lucide-react';

interface VideoTileProps {
  stream: MediaStream | null;
  displayName: string;
  isLocal?: boolean;
  isPrimary?: boolean;
  audioEnabled?: boolean;
  videoEnabled?: boolean;
  isAslEnabled?: boolean;
  handRaised?: boolean;
  onVideoElementReady?: (videoElement: HTMLVideoElement) => void;
  onPin?: () => void;
  onFullscreen?: () => void;
}

export function VideoTile({
  stream,
  displayName,
  isLocal = false,
  isPrimary = false,
  audioEnabled = true,
  videoEnabled = true,
  isAslEnabled = false,
  handRaised = false,
  onVideoElementReady,
  onPin,
  onFullscreen,
}: VideoTileProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;

      if (onVideoElementReady) {
        onVideoElementReady(videoRef.current);
      }
    }
  }, [stream, onVideoElementReady]);

  return (
    <div
      className="relative overflow-hidden bg-black rounded-2xl w-full h-full group border-2 border-gray-800 shadow-2xl transition-all duration-300 hover:border-yellow-400/50 hover:shadow-yellow-400/20"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted={isLocal}
        className={`w-full h-full object-contain transition-all duration-500 ${isLocal ? 'scale-x-[-1]' : ''} ${!videoEnabled ? 'opacity-0 scale-95' : 'opacity-100 scale-100'}`}
      />

      {/* Camera Off State */}
      {(!stream || !videoEnabled) && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-gray-900 via-black to-gray-900">
          <div className="relative">
            <div className="absolute inset-0 bg-yellow-400/10 blur-3xl rounded-full" />
            <div className="relative w-24 h-24 md:w-28 md:h-28 rounded-full bg-gradient-to-br from-gray-800 to-black border-2 border-gray-700 flex items-center justify-center mb-4 shadow-xl">
              <User className="w-10 h-10 md:w-12 md:h-12 text-gray-600 group-hover:text-yellow-400 transition-colors duration-300" />
              {!videoEnabled && (
                <div className="absolute -bottom-1 -right-1 bg-red-500 rounded-full p-2 border-4 border-black shadow-lg">
                  <VideoOff className="w-3 h-3 md:w-4 md:h-4 text-white" />
                </div>
              )}
            </div>
          </div>
          <h3 className="text-white font-bold text-base md:text-lg mb-1">
            {displayName} {isLocal && "(You)"}
          </h3>
          <p className="text-gray-500 text-xs font-medium uppercase tracking-wide flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse" />
            Camera Off
          </p>
        </div>
      )}

      {/* Gradient Overlay for Better Text Readability */}
      <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-black/90 via-black/50 to-transparent pointer-events-none" />

      {/* Bottom Info Bar */}
      <div className="absolute bottom-4 left-4 right-4 flex items-end justify-between pointer-events-none">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            {/* Name Badge */}
            <div className="bg-black/70 backdrop-blur-md px-3 md:px-4 py-2 rounded-xl flex items-center gap-2 md:gap-3 border border-white/10 shadow-xl pointer-events-auto transition-all hover:bg-black/80">
              <div className={`w-2 h-2 rounded-full ${audioEnabled ? 'bg-green-500 shadow-lg shadow-green-500/50' : 'bg-red-500'}`} />
              <span className="text-xs md:text-sm font-bold text-white">
                {displayName}
              </span>
              {!audioEnabled && (
                <MicOff className="w-3 md:w-3.5 h-3 md:h-3.5 text-red-400" />
              )}
            </div>

            {/* ASL Enabled Badge */}
            {isAslEnabled && (
              <div className="bg-yellow-400 px-3 py-2 rounded-xl flex items-center gap-2 shadow-lg shadow-yellow-400/30 border border-yellow-500 animate-in slide-in-from-left duration-300 pointer-events-auto">
                <Hand className="w-3 md:w-3.5 h-3 md:h-3.5 text-black fill-current" />
                <span className="text-[10px] md:text-xs font-bold text-black tracking-wide uppercase">ASL Active</span>
              </div>
            )}
          </div>
        </div>

        {/* Hand Raised Indicator */}
        {handRaised && (
          <div className="bg-yellow-400 p-2.5 md:p-3 rounded-xl shadow-xl shadow-yellow-400/50 animate-bounce pointer-events-auto border-2 border-yellow-500">
            <Hand className="w-5 h-5 md:w-6 md:h-6 text-black fill-current" />
          </div>
        )}
      </div>

      {/* Hover Controls */}
      <div className={`absolute top-4 right-4 flex flex-col gap-2 transition-all duration-300 ${isHovered ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-4 pointer-events-none'}`}>
        {onPin && (
          <ControlButton
            icon={<Pin className="w-4 h-4" />}
            onClick={onPin}
            label="Pin"
          />
        )}
        {onFullscreen && (
          <ControlButton
            icon={<Maximize2 className="w-4 h-4" />}
            onClick={onFullscreen}
            label="Fullscreen"
          />
        )}
      </div>

      {/* Primary Indicator */}
      {isPrimary && (
        <div className="absolute top-4 left-4 bg-yellow-400 px-3 py-1.5 rounded-lg shadow-lg">
          <span className="text-xs font-bold text-black uppercase tracking-wide">Primary</span>
        </div>
      )}
    </div>
  );
}

function ControlButton({
  icon,
  onClick,
  label
}: {
  icon: React.ReactNode;
  onClick: () => void;
  label?: string;
}) {
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={onClick}
      className="bg-black/70 hover:bg-yellow-400 text-white hover:text-black rounded-xl w-10 h-10 md:w-11 md:h-11 backdrop-blur-md border border-white/10 hover:border-yellow-400 transition-all hover:scale-110 active:scale-95 shadow-xl pointer-events-auto"
      title={label}
    >
      {icon}
    </Button>
  );
}