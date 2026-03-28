'use client';

import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Hand, Send, Trash2, Space, Sparkles } from 'lucide-react';

interface ASLPanelProps {
  isActive: boolean;
  lastRecognition?: {
    text: string;
    confidence: number;
  };
  frameCount?: number;
  sentenceBuffer?: string;
  onSendSentence?: () => void;
  onClearSentence?: () => void;
  onAddSpace?: () => void;
  localVideoElement?: HTMLVideoElement | null;
}

export function ASLPanel({
  isActive,
  lastRecognition,
  frameCount = 0,
  sentenceBuffer = '',
  onSendSentence,
  onClearSentence,
  onAddSpace,
  localVideoElement,
}: ASLPanelProps) {
  if (!isActive) {
    return (
      <Card className="bg-white border-0 shadow-lg p-8">
        <div className="text-center">
          <div className="w-16 h-16 bg-palette-light/30 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Hand className="w-8 h-8 text-palette-dark/40" />
          </div>
          <h3 className="text-lg font-bold text-palette-dark mb-2">
            ASL Interpreter
          </h3>
          <p className="text-palette-dark/60 text-sm">
            Enable ASL mode to start real-time translation
          </p>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-4 p-4">
      {/* ASL Camera Feed */}
      <Card className="bg-white border-0 shadow-lg overflow-hidden">
        <div className="bg-palette-medium px-4 py-3 border-b border-palette-medium">
          <h3 className="text-palette-dark font-bold flex items-center gap-2 text-sm">
            <Hand className="w-4 h-4 text-palette-dark" />
            Live ASL Recognition
          </h3>
        </div>
        
        <div className="aspect-video bg-palette-dark relative">
          <video
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover scale-x-[-1]"
            ref={(video) => {
              if (video && localVideoElement?.srcObject) {
                video.srcObject = localVideoElement.srcObject;
              }
            }}
          />
          
          {/* Live Recognition Overlay */}
          {lastRecognition && lastRecognition.text !== 'None' && (
            <div className="absolute top-4 left-4 right-4">
              <div className="bg-palette-medium backdrop-blur-sm rounded-xl px-4 py-3 shadow-xl border-2 border-palette-medium">
                <div className="text-3xl font-bold text-palette-dark mb-1 text-center">
                  {lastRecognition.text}
                </div>
                <div className="flex items-center justify-center gap-2">
                  <div className="h-1.5 flex-1 bg-black/20 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-palette-dark rounded-full transition-all duration-300"
                      style={{ width: `${lastRecognition.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-bold text-palette-dark">
                    {(lastRecognition.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          )}
          
          {/* Status Indicator */}
          <div className="absolute bottom-4 left-4 bg-palette-dark/70 backdrop-blur-md rounded-lg px-3 py-2 flex items-center gap-2 border border-white/10">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-xs font-semibold text-white">
              Active
            </span>
          </div>

          {/* Frame Counter */}
          <div className="absolute bottom-4 right-4 bg-palette-dark/70 backdrop-blur-md rounded-lg px-3 py-2 border border-white/10">
            <span className="text-xs font-semibold text-white">
              {frameCount} frames
            </span>
          </div>
        </div>
      </Card>

      {/* Sentence Builder */}
      <Card className="bg-white border-0 shadow-lg">
        <div className="bg-palette-light/40 px-4 py-3 border-b border-palette-light/50">
          <h4 className="text-palette-dark font-bold text-sm flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-palette-dark" />
            Sentence Builder
          </h4>
        </div>
        
        <div className="p-4">
          {/* Buffer Display */}
          <div className="min-h-[80px] p-4 bg-gray-50 border-2 border-gray-200 rounded-xl text-palette-dark mb-4 font-mono text-lg">
            {sentenceBuffer || (
              <span className="text-palette-dark/40 font-sans text-sm">
                Your signed letters will appear here...
              </span>
            )}
          </div>
          
          {/* Action Buttons */}
          <div className="flex gap-2">
            <Button
              onClick={onSendSentence}
              disabled={!sentenceBuffer.trim()}
              className="flex-1 bg-palette-dark hover:bg-palette-dark/90 text-palette-offwhite font-bold shadow-lg disabled:opacity-50 disabled:cursor-not-allowed h-11"
              size="sm"
            >
              <Send className="w-4 h-4 mr-2" />
              Send to Chat
            </Button>
            
            <Button
              onClick={onAddSpace}
              className="border-2 border-black/20 text-palette-dark hover:bg-palette-dark/5 hover:border-palette-dark/30 h-11 w-11 p-0"
              variant="outline"
              size="sm"
              title="Add Space"
            >
              <Space className="w-4 h-4" />
            </Button>
            
            <Button
              onClick={onClearSentence}
              disabled={!sentenceBuffer.trim()}
              className="border-2 border-red-200 text-red-600 hover:bg-red-50 hover:border-red-300 disabled:opacity-50 disabled:cursor-not-allowed h-11 w-11 p-0"
              variant="outline"
              size="sm"
              title="Clear"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>

          {/* Quick Tips */}
          <div className="mt-4 p-3 bg-palette-light/30 rounded-lg border border-palette-light/40">
            <p className="text-xs text-palette-dark/70">
              <strong className="text-palette-dark">Tip:</strong> Sign letters to build your message, then click "Send to Chat" to share with participants.
            </p>
          </div>
        </div>
      </Card>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-3">
        <Card className="bg-white border-0 shadow-md p-4">
          <div className="text-2xl font-bold text-palette-dark mb-1">
            {sentenceBuffer.length}
          </div>
          <div className="text-xs text-palette-dark/60 font-semibold">Characters</div>
        </Card>
        <Card className="bg-white border-0 shadow-md p-4">
          <div className="text-2xl font-bold text-palette-dark mb-1">
            {sentenceBuffer.split(' ').filter(w => w.length > 0).length}
          </div>
          <div className="text-xs text-palette-dark/60 font-semibold">Words</div>
        </Card>
      </div>
    </div>
  );
}