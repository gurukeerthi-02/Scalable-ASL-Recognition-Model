'use client';

import { Button } from '@/components/ui/button';
import {
  Mic,
  MicOff,
  Video,
  VideoOff,
  PhoneOff,
  Hand,
  Sparkles,
  MessageSquare
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
  showASLPanel: boolean;
  onToggleAudio: () => void;
  onToggleVideo: () => void;
  onToggleVoiceOut: () => void;
  onToggleHandRaise: () => void;
  onToggleASLPanel: () => void;
  onLeave: () => void;
}

export function CallControls({
  audioEnabled,
  videoEnabled,
  voiceOutEnabled,
  handRaised,
  showASLPanel,
  onToggleAudio,
  onToggleVideo,
  onToggleVoiceOut,
  onToggleHandRaise,
  onToggleASLPanel,
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

        {/* Chat / ASL Panel Toggle */}
        <ControlToggle
          active={showASLPanel}
          activeIcon={<MessageSquare className="w-4 h-4 sm:w-5 sm:h-5" />}
          inactiveIcon={<MessageSquare className="w-4 h-4 sm:w-5 sm:h-5" />}
          onClick={onToggleASLPanel}
          tooltip={showASLPanel ? 'Hide Chat & Interpreter' : 'Show Chat & Interpreter'}
          variant="secondary"
        />

        {/* Divider */}
        <div className="w-px h-8 sm:h-10 bg-palette-dark/10 mx-1 sm:mx-2" />

        {/* Leave Button */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="destructive"
              onClick={onLeave}
              className="rounded-full w-10 h-10 sm:w-14 sm:h-14 flex items-center justify-center transition-all bg-red-600 hover:bg-red-700 shadow-lg hover:shadow-xl border-0 hover:scale-105 active:scale-95"
            >
              <PhoneOff className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
            </Button>
          </TooltipTrigger>
          <TooltipContent className="bg-palette-dark text-palette-offwhite border-palette-light/40 font-semibold">
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
  variant?: 'status' | 'brand' | 'warning' | 'secondary'
}) {

  const getColors = () => {
    // Audio/Video turned OFF
    if (!active && variant === 'status') {
      return 'bg-red-500 text-white border-0 hover:bg-red-600 shadow-md';
    }
    
    // Everything else (Active AV, Toggles, etc) -> simple white background with black icons
    let classes = 'bg-white text-black border border-gray-200 hover:shadow-sm';
    
    // Subtle indicators for active toggle states (Chat, ASL)
    if (active && (variant === 'brand' || variant === 'secondary')) {
      classes = 'bg-gray-900 text-white border border-gray-300 hover:shadow-inner';
    }
    
    // Alert state for hand raised
    if (active && variant === 'warning') {
      classes = 'bg-gray-900 text-white border-0 hover:bg-gray-800 shadow-md';
    }

    return classes;
  };

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          onClick={onClick}
          className={`rounded-full w-10 h-10 sm:w-14 sm:h-14 flex items-center justify-center transition-all duration-200 hover:scale-105 active:scale-95 ${getColors()}`}
          variant="ghost"
        >
          {active ? activeIcon : inactiveIcon}
        </Button>
      </TooltipTrigger>
      <TooltipContent className="bg-palette-dark text-palette-offwhite border-palette-light/40 font-semibold">
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}