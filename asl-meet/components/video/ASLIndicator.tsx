'use client';

import { useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { COMMON_WORDS } from '@/lib/dictionary';
import { Button } from '@/components/ui/button';
import { Hand, Send, Trash2, Space, Delete, Zap, MessageSquare, Activity, User, MessageCircle, Sparkles } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

import { ASLRecognitionResponse } from '@/types';

interface ASLIndicatorProps {
  isActive: boolean;
  lastRecognition?: ASLRecognitionResponse;
  frameCount?: number;
  sentenceBuffer?: string;
  onSendSentence?: () => void;
  onClearSentence?: () => void;
  onAddSpace?: () => void;
  onBackspace?: () => void;
  onSelectSuggestion?: (word: string) => void;
  receivedMessages?: Array<{ peerId: string, text: string, timestamp: number, displayName?: string }>;
}

export function ASLIndicator({
  isActive,
  lastRecognition,
  frameCount = 0,
  sentenceBuffer = '',
  onSendSentence,
  onClearSentence,
  onAddSpace,
  onBackspace,
  onSelectSuggestion,
  receivedMessages = []
}: ASLIndicatorProps) {

  const suggestions = useMemo(() => {
    const words = sentenceBuffer.split(' ');
    const lastWord = words[words.length - 1].toUpperCase();
    if (!lastWord || lastWord.length < 2) return [];
    
    const matches = COMMON_WORDS.filter(w => 
      w.toUpperCase().startsWith(lastWord) && w.toUpperCase() !== lastWord
    );
    return matches.slice(0, 3).map(w => w.toUpperCase());
  }, [sentenceBuffer]);

  return (
    <div className="flex flex-col h-full bg-white overflow-hidden">
      <Tabs defaultValue="interpreter" className="flex-1 flex flex-col overflow-hidden">
        <div className="px-4 pt-2 bg-gray-50/50">
          <TabsList className="grid w-full grid-cols-2 bg-palette-dark/5 p-1 rounded-xl">
            <TabsTrigger
              value="interpreter"
              className="rounded-lg data-[state=active]:bg-white data-[state=active]:text-palette-dark data-[state=active]:shadow-sm font-bold text-xs flex items-center gap-2"
            >
              <Activity className="w-3.5 h-3.5" />
              Interpreter
            </TabsTrigger>
            <TabsTrigger
              value="chat"
              className="rounded-lg data-[state=active]:bg-white data-[state=active]:text-palette-dark data-[state=active]:shadow-sm font-bold text-xs flex items-center gap-2"
            >
              <MessageCircle className="w-3.5 h-3.5" />
              Chat
              {receivedMessages.length > 0 && (
                <Badge variant="secondary" className="ml-1 h-4 min-w-4 p-0 px-1 flex items-center justify-center text-[9px] bg-red-500 text-white border-0 font-bold">
                  {receivedMessages.length}
                </Badge>
              )}
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="interpreter" className="flex-1 overflow-y-auto m-0 p-4 space-y-5 focus-visible:outline-none focus-visible:ring-0">
          {!isActive && (
            <div className="bg-gray-50 border border-gray-200 rounded-2xl p-6 text-center">
              <div className="w-12 h-12 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-3">
                <Sparkles className="w-6 h-6 text-gray-400" />
              </div>
              <h4 className="text-sm font-bold text-gray-900 mb-1">Interpreter Disabled</h4>
              <p className="text-xs text-gray-500 mb-4">Enable the interpreter in the call controls to start recognition.</p>
            </div>
          )}

          <div className={!isActive ? 'opacity-40 pointer-events-none grayscale' : ''}>
            {/* Real-time Recognition Card */}
            <Card className="p-4 bg-palette-light/30 border-2 border-palette-light/50 shadow-lg">
              <div className="flex items-start gap-3">
                <div className="relative flex-shrink-0">
                  <div className="w-12 h-12 bg-palette-medium rounded-xl flex items-center justify-center shadow-md">
                    <Activity className="w-6 h-6 text-palette-dark" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white shadow-sm animate-pulse" />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <h3 className="text-xs font-bold uppercase tracking-wide text-palette-dark">Live Recognition</h3>
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
                    <span className="text-[10px] font-mono text-palette-dark/60">{frameCount} frames</span>
                  </div>

                  {lastRecognition && lastRecognition.text ? (
                    <div className="space-y-3">
                      <div className="flex items-baseline gap-2 flex-wrap">
                        <span className="text-3xl md:text-4xl font-black text-palette-dark tracking-tight">
                          {lastRecognition.text}
                        </span>
                        <Badge className="bg-palette-dark text-palette-offwhite border-0 text-xs font-bold">
                          {(lastRecognition.confidence * 100).toFixed(0)}%
                        </Badge>
                      </div>
                      <div className="h-2 bg-palette-dark/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-palette-medium transition-all duration-300 rounded-full"
                          style={{ width: `${lastRecognition.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="py-2 space-y-3">
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-white p-2 rounded-lg border border-gray-200">
                          <p className="text-[8px] uppercase text-palette-dark/50 font-bold mb-0.5">Motion</p>
                          <p className="text-xs font-mono text-palette-dark">{(lastRecognition?.motion || 0).toFixed(4)}</p>
                        </div>
                        <div className="bg-white p-2 rounded-lg border border-gray-200">
                          <p className="text-[8px] uppercase text-palette-dark/50 font-bold mb-0.5">Mode</p>
                          <p className="text-xs font-mono text-palette-dark">
                            {lastRecognition?.mode === 0 ? 'IDLE' :
                              lastRecognition?.mode === 1 ? 'STATIC' :
                                lastRecognition?.mode === 2 ? 'DYNAMIC' : 'HOLD'}
                          </p>
                        </div>
                      </div>
                      <div className="h-1.5 w-full bg-palette-dark/10 overflow-hidden rounded-full">
                        <div
                          className="h-full bg-palette-medium transition-all duration-300 rounded-full"
                          style={{ width: `${((lastRecognition?.buffer_size || 0) / 30) * 100}%` }}
                        />
                      </div>
                      <p className="text-xs text-palette-dark/60 font-medium italic animate-pulse">Analyzing hand gestures...</p>
                    </div>
                  )}
                </div>
              </div>
            </Card>

            {/* Translation Buffer */}
            <div className="space-y-3">
              <div className="flex items-center justify-between ml-1 mb-1">
                <h4 className="text-[10px] font-bold uppercase tracking-wider text-palette-dark/60">Current Sentence</h4>
                <div className="flex flex-wrap gap-2">
                  {suggestions.map((sug) => (
                    <button
                      key={sug}
                      onClick={() => onSelectSuggestion?.(sug)}
                      className="px-3 py-1 bg-palette-light/20 hover:bg-palette-medium hover:text-white text-palette-dark text-xs font-bold rounded-full shadow-sm border border-palette-medium/30 transition-all active:scale-95 flex items-center justify-center whitespace-nowrap"
                    >
                      {sug}
                    </button>
                  ))}
                </div>
              </div>
              <div className="relative group">
                <div className="min-h-[120px] p-4 bg-gray-50 border-2 border-gray-200 rounded-2xl text-lg font-medium text-palette-dark leading-relaxed group-hover:border-gray-300 transition-colors shadow-inner">
                  {sentenceBuffer || (
                    <span className="text-black/30 italic text-sm font-normal">Signs will appear here as you make them...</span>
                  )}
                </div>
                <div className="absolute bottom-3 right-3 flex gap-1.5">
                  <Button
                    size="icon"
                    variant="secondary"
                    onClick={onBackspace}
                    disabled={!sentenceBuffer.length}
                    className="h-9 w-9 bg-white text-palette-dark border border-gray-200 hover:bg-orange-500 hover:border-orange-500 hover:text-white rounded-xl shadow-sm transition-all disabled:opacity-30"
                    title="Backspace"
                  >
                    <Delete className="w-4 h-4" />
                  </Button>
                  <Button
                    size="icon"
                    variant="secondary"
                    onClick={onAddSpace}
                    className="h-9 w-9 bg-white text-palette-dark border border-gray-200 hover:bg-palette-dark hover:text-white rounded-xl shadow-sm transition-all"
                    title="Add Space"
                  >
                    <Space className="w-4 h-4" />
                  </Button>
                  <Button
                    size="icon"
                    variant="secondary"
                    onClick={onClearSentence}
                    disabled={!sentenceBuffer.trim()}
                    className="h-9 w-9 bg-white text-palette-dark border border-gray-200 hover:bg-red-600 hover:text-white hover:border-red-600 rounded-xl shadow-sm transition-all disabled:opacity-30"
                    title="Clear Buffer"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 pt-2">
              <div className="bg-gray-50 p-3 rounded-xl border border-gray-100">
                <p className="text-[9px] font-bold text-palette-dark/40 uppercase tracking-widest mb-1">Words</p>
                <p className="text-xl font-black text-palette-dark">{sentenceBuffer.split(' ').filter(w => w.length > 0).length}</p>
              </div>
              <div className="bg-gray-50 p-3 rounded-xl border border-gray-100">
                <p className="text-[9px] font-bold text-palette-dark/40 uppercase tracking-widest mb-1">Characters</p>
                <p className="text-xl font-black text-palette-dark">{sentenceBuffer.length}</p>
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="chat" className="flex-1 overflow-y-auto m-0 p-4 space-y-3 focus-visible:outline-none focus-visible:ring-0">
          {receivedMessages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-8">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                <MessageSquare className="w-8 h-8 text-black/20" />
              </div>
              <h3 className="text-sm font-bold text-palette-dark mb-1">No messages yet</h3>
              <p className="text-xs text-palette-dark/40 max-w-[200px]">Sent and received messages will be displayed here.</p>
            </div>
          ) : (
            <div className="space-y-3 pb-4">
              {receivedMessages.slice().reverse().map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex flex-col ${msg.displayName === 'You' ? 'items-end' : 'items-start'} animate-in slide-in-from-bottom-2 duration-300`}
                >
                  <div className="flex items-center gap-1.5 mb-1 px-1">
                    <span className="text-[10px] font-bold text-palette-dark/60 uppercase tracking-wide">
                      {msg.displayName || 'Participant'}
                    </span>
                    <span className="text-[9px] text-black/30">
                      {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                  <div className={`max-w-[85%] p-3 rounded-2xl shadow-sm text-sm font-medium ${msg.displayName === 'You'
                    ? 'bg-palette-dark text-palette-offwhite rounded-tr-none'
                    : 'bg-palette-light/40 text-palette-dark border border-palette-light/50 rounded-tl-none'
                    }`}>
                    {msg.text}
                  </div>
                </div>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Persistent Footer Actions */}
      <div className="p-4 bg-white border-t border-gray-100">
        <Button
          onClick={onSendSentence}
          disabled={!sentenceBuffer.trim()}
          className="w-full h-12 rounded-xl bg-palette-dark hover:bg-palette-dark/90 text-palette-offwhite font-bold shadow-lg transition-all active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed group"
        >
          <Send className="w-4 h-4 mr-2 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
          Send Integrated Message
        </Button>
      </div>
    </div>
  );
}