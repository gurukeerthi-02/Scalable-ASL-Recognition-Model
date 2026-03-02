'use client';

import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Hand, Send, Trash2, Space, Zap, MessageSquare, Activity } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

import { ASLRecognitionResponse } from '@/types';

interface ASLIndicatorProps {
  isActive: boolean;
  lastRecognition?: ASLRecognitionResponse;
  frameCount?: number;
  sentenceBuffer?: string;
  onSendSentence?: () => void;
  onClearSentence?: () => void;
  onAddSpace?: () => void;
  receivedMessages?: Array<{ peerId: string, text: string, timestamp: number }>;
}

export function ASLIndicator({
  isActive,
  lastRecognition,
  frameCount = 0,
  sentenceBuffer = '',
  onSendSentence,
  onClearSentence,
  onAddSpace,
  receivedMessages = []
}: ASLIndicatorProps) {
  if (!isActive) {
    return null;
  }

  return (
    <div className="flex flex-col h-full bg-white overflow-hidden">
      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">

        {/* Real-time Recognition Card */}
        <Card className="p-4 bg-yellow-400/10 border-2 border-yellow-400/30 shadow-lg">
          <div className="flex items-start gap-3">
            <div className="relative flex-shrink-0">
              <div className="w-12 h-12 bg-yellow-400 rounded-xl flex items-center justify-center shadow-md">
                <Activity className="w-6 h-6 text-black" />
              </div>
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white shadow-sm animate-pulse" />
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <h3 className="text-xs font-bold uppercase tracking-wide text-black">Live Recognition</h3>
                  {lastRecognition && (
                    <Badge className={`text-[9px] font-black border-0 px-1.5 h-4 ${lastRecognition.mode === 1 ? 'bg-blue-600 text-white' :
                      lastRecognition.mode === 2 ? 'bg-purple-600 text-white' :
                        lastRecognition.mode === 3 ? 'bg-green-600 text-white' :
                          'bg-gray-400 text-white'
                      }`}>
                      {lastRecognition.mode === 1 ? 'STATIC' :
                        lastRecognition.mode === 2 ? 'DYNAMIC' :
                          lastRecognition.mode === 3 ? 'RESULT' : 'IDLE'}
                    </Badge>
                  )}
                </div>
                <span className="text-[10px] font-mono text-black/60">{frameCount} frames</span>
              </div>

              {lastRecognition && lastRecognition.text ? (
                <div className="space-y-3">
                  <div className="flex items-baseline gap-2 flex-wrap">
                    <span className="text-3xl md:text-4xl font-black text-black tracking-tight">
                      {lastRecognition.text}
                    </span>
                    <Badge className="bg-black text-yellow-400 border-0 text-xs font-bold">
                      {(lastRecognition.confidence * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  <div className="h-2 bg-black/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-yellow-400 transition-all duration-300 rounded-full"
                      style={{ width: `${lastRecognition.confidence * 100}%` }}
                    />
                  </div>
                </div>
              ) : (
                <div className="py-2 space-y-3">
                  {/* Debug Metrics */}
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-white p-2 rounded-lg border border-gray-200">
                      <p className="text-[8px] uppercase text-black/50 font-bold mb-0.5">Motion</p>
                      <p className="text-xs font-mono text-black">{(lastRecognition?.motion || 0).toFixed(4)}</p>
                    </div>
                    <div className="bg-white p-2 rounded-lg border border-gray-200">
                      <p className="text-[8px] uppercase text-black/50 font-bold mb-0.5">Mode</p>
                      <p className="text-xs font-mono text-black">
                        {lastRecognition?.mode === 0 ? 'IDLE' :
                          lastRecognition?.mode === 1 ? 'STATIC' :
                            lastRecognition?.mode === 2 ? 'DYNAMIC' : 'HOLD'}
                      </p>
                    </div>
                    <div className="bg-white p-2 rounded-lg border border-gray-200">
                      <p className="text-[8px] uppercase text-black/50 font-bold mb-0.5">Stability</p>
                      <p className="text-xs font-mono text-black">{lastRecognition?.stable_count || 0}/5</p>
                    </div>
                    <div className="bg-white p-2 rounded-lg border border-gray-200">
                      <p className="text-[8px] uppercase text-black/50 font-bold mb-0.5">Buffer</p>
                      <p className="text-xs font-mono text-black">{lastRecognition?.buffer_size || 0}/30</p>
                    </div>
                  </div>

                  <div className="h-1.5 w-full bg-black/10 overflow-hidden rounded-full">
                    <div
                      className="h-full bg-yellow-400 transition-all duration-300 rounded-full"
                      style={{ width: `${((lastRecognition?.buffer_size || 0) / 30) * 100}%` }}
                    />
                  </div>
                  <p className="text-xs text-black/60 font-medium">Analyzing hand gestures...</p>
                </div>
              )}
            </div>
          </div>
        </Card>

        {/* Translation Buffer */}
        <div className="space-y-2">
          <h4 className="text-[10px] font-bold uppercase tracking-wider text-black/60 ml-1">Message Buffer</h4>
          <div className="relative group">
            <div className="min-h-[80px] md:min-h-[100px] p-3 md:p-4 bg-gray-50 border-2 border-gray-200 rounded-xl text-base md:text-lg font-medium text-black leading-relaxed group-hover:border-gray-300 transition-colors">
              {sentenceBuffer || (
                <span className="text-black/40 italic text-sm font-normal">Start signing to build your message...</span>
              )}
            </div>
            <div className="absolute bottom-2 right-2 flex gap-1">
              <Button
                size="icon"
                variant="ghost"
                onClick={onAddSpace}
                className="h-8 w-8 text-black/50 hover:text-black hover:bg-black/5 rounded-lg"
                title="Add Space"
              >
                <Space className="w-4 h-4" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                onClick={onClearSentence}
                disabled={!sentenceBuffer.trim()}
                className="h-8 w-8 text-black/50 hover:text-red-600 hover:bg-red-50 rounded-lg disabled:opacity-30"
                title="Clear Buffer"
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Character count */}
          {sentenceBuffer && (
            <div className="flex items-center justify-between text-xs text-black/50 px-1">
              <span>{sentenceBuffer.length} characters</span>
              <span>{sentenceBuffer.split(' ').filter(w => w.length > 0).length} words</span>
            </div>
          )}
        </div>

        {/* Recent Messages */}
        {receivedMessages.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-[10px] font-bold uppercase tracking-wider text-black/60 ml-1 flex items-center gap-1.5">
              <MessageSquare className="w-3 h-3" />
              Recent Messages
            </h4>
            <div className="space-y-2">
              {receivedMessages.slice(-5).reverse().map((msg, idx) => (
                <div key={idx} className="p-3 bg-yellow-400/5 border border-yellow-400/20 rounded-xl animate-in slide-in-from-right duration-300">
                  <p className="text-sm font-medium text-black">{msg.text}</p>
                  <p className="text-[10px] text-black/50 mt-1 uppercase font-semibold tracking-wide">From participant</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Quick Stats */}
        <div className="grid grid-cols-2 gap-2">
          <Card className="p-3 bg-white border border-gray-200 shadow-sm">
            <div className="text-2xl font-bold text-black mb-0.5">
              {sentenceBuffer.length}
            </div>
            <div className="text-[10px] text-black/60 font-semibold uppercase tracking-wide">Characters</div>
          </Card>
          <Card className="p-3 bg-white border border-gray-200 shadow-sm">
            <div className="text-2xl font-bold text-black mb-0.5">
              {frameCount}
            </div>
            <div className="text-[10px] text-black/60 font-semibold uppercase tracking-wide">Frames</div>
          </Card>
        </div>
      </div>

      {/* Action Footer */}
      <div className="p-3 md:p-4 bg-gradient-to-t from-gray-50 to-white border-t border-gray-200">
        <Button
          onClick={onSendSentence}
          disabled={!sentenceBuffer.trim()}
          className="w-full h-10 md:h-12 rounded-xl bg-black hover:bg-black/90 text-yellow-400 font-bold shadow-lg transition-all active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Send className="w-4 h-4 mr-2" />
          Send Message
        </Button>

        {/* Helper text */}
        <p className="text-center text-[10px] text-black/50 mt-2">
          Message will be sent to all participants
        </p>
      </div>
    </div>
  );
}