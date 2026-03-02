'use client';

import { Button } from '@/components/ui/button';
import {
  Mic,
  MicOff,
  Video,
  VideoOff,
  PhoneOff,
  Hand,
  Sparkles
} from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface CallControlsProps {
  audioEnabled: boolean;
  videoEnabled: boolean;
  voiceOutEnabled: boolean;
  handRaised: boolean;
  onToggleAudio: () => void;
  onToggleVideo: () => void;
  onToggleVoiceOut: () => void;
  onToggleHandRaise: () => void;
  onLeave: () => void;
}

export function CallControls({
  audioEnabled,
  videoEnabled,
  voiceOutEnabled,
  handRaised,
  onToggleAudio,
  onToggleVideo,
  onToggleVoiceOut,
  onToggleHandRaise,
  onLeave,
}: CallControlsProps) {
  return (
    <TooltipProvider>
      <div className="flex items-center justify-center gap-2 sm:gap-3 md:gap-4">
        {/* Audio Toggle */}
        <ControlToggle
          active={audioEnabled}
          activeIcon={<Mic className="w-4 h-4 sm:w-5 sm:h-5" />}
          inactiveIcon={<MicOff className="w-4 h-4 sm:w-5 sm:h-5" />}
          onClick={onToggleAudio}
          tooltip={audioEnabled ? 'Mute Microphone' : 'Unmute Microphone'}
          variant="status"
        />

        {/* Video Toggle */}
        <ControlToggle
          active={videoEnabled}
          activeIcon={<Video className="w-4 h-4 sm:w-5 sm:h-5" />}
          inactiveIcon={<VideoOff className="w-4 h-4 sm:w-5 sm:h-5" />}
          onClick={onToggleVideo}
          tooltip={videoEnabled ? 'Turn Off Camera' : 'Turn On Camera'}
          variant="status"
        />

        {/* ASL Recognition Toggle */}
        <ControlToggle
          active={voiceOutEnabled}
          activeIcon={<Sparkles className="w-4 h-4 sm:w-5 sm:h-5" />}
          inactiveIcon={<Sparkles className="w-4 h-4 sm:w-5 sm:h-5" />}
          onClick={onToggleVoiceOut}
          tooltip={voiceOutEnabled ? 'Disable ASL Interpreter' : 'Enable ASL Interpreter'}
          variant="brand"
        />

        {/* Hand Raise */}
        <ControlToggle
          active={handRaised}
          activeIcon={<Hand className="w-4 h-4 sm:w-5 sm:h-5 fill-current" />}
          inactiveIcon={<Hand className="w-4 h-4 sm:w-5 sm:h-5" />}
          onClick={onToggleHandRaise}
          tooltip={handRaised ? 'Lower Hand' : 'Raise Hand'}
          variant="warning"
        />

        {/* Divider */}
        <div className="w-px h-8 sm:h-10 bg-black/10 mx-1 sm:mx-2" />

        {/* Leave Button */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="destructive"
              onClick={onLeave}
              className="rounded-full w-12 h-12 sm:w-14 sm:h-14 flex items-center justify-center transition-all bg-red-600 hover:bg-red-700 shadow-lg hover:shadow-xl border-0 hover:scale-105 active:scale-95"
            >
              <PhoneOff className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
            </Button>
          </TooltipTrigger>
          <TooltipContent className="bg-black text-yellow-400 border-yellow-400/20 font-semibold">
            End Call
          </TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
}

function ControlToggle({
  active,
  activeIcon,
  inactiveIcon,
  onClick,
  tooltip,
  variant = 'status'
}: {
  active: boolean,
  activeIcon: React.ReactNode,
  inactiveIcon: React.ReactNode,
  onClick: () => void,
  tooltip: string,
  variant?: 'status' | 'brand' | 'warning'
}) {

  const getColors = () => {
    if (active) {
      if (variant === 'brand') {
        // ASL mode active - yellow background
        return 'bg-yellow-400 text-black shadow-lg shadow-yellow-400/30 hover:bg-yellow-500 border-2 border-yellow-500';
      }
      if (variant === 'warning') {
        // Hand raised - yellow background
        return 'bg-yellow-400 text-black shadow-lg shadow-yellow-400/30 hover:bg-yellow-500 border-2 border-yellow-500 animate-pulse';
      }
      // Audio/Video active - white background
      return 'bg-white text-black hover:bg-gray-100 border-2 border-gray-200 shadow-md';
    }
    // Inactive state - red for disabled
    return 'bg-red-500/20 text-red-600 border-2 border-red-500/40 hover:bg-red-500/30 shadow-sm';
  };

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          onClick={onClick}
          className={`rounded-full w-12 h-12 sm:w-14 sm:h-14 flex items-center justify-center transition-all duration-200 hover:scale-105 active:scale-95 ${getColors()}`}
          variant="ghost"
        >
          {active ? activeIcon : inactiveIcon}
        </Button>
      </TooltipTrigger>
      <TooltipContent className="bg-black text-yellow-400 border-yellow-400/20 font-semibold">
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}