'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Copy, Video, Users, ArrowLeft, Check, Sparkles, Hand, Shield } from 'lucide-react';
import { supabase } from '@/lib/supabase';

export default function CreateCallPage() {
  const router = useRouter();
  const [displayName, setDisplayName] = useState('');
  const [roomReason, setRoomReason] = useState('');
  const [roomId, setRoomId] = useState('');
  const [roomName, setRoomName] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [copied, setCopied] = useState(false);

  const generateRoomId = () => {
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < 6; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
  };

  const createRoom = async () => {
    if (!displayName.trim() || !roomReason.trim()) return;

    setIsCreating(true);
    const newRoomId = generateRoomId();
    const finalRoomName = roomReason.trim();

    try {
      const { error } = await supabase
        .from('rooms')
        .insert({
          id: newRoomId,
          name: finalRoomName,
          max_participants: 10,
          created_by: displayName,
          is_active: true
        });

      if (error) throw error;

      setRoomId(newRoomId);
      setRoomName(finalRoomName);
    } catch (error) {
      console.error('Error creating room:', error);
      alert('Failed to create room');
    } finally {
      setIsCreating(false);
    }
  };

  const copyRoomId = async () => {
    await navigator.clipboard.writeText(roomId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-palette-offwhite flex flex-col relative overflow-hidden">
      {/* Decorative Elements */}
      <div className="absolute top-20 left-10 w-64 h-64 bg-palette-dark/5 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-palette-dark/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      
      {/* Navigation */}
      <nav className="p-6 flex items-center max-w-7xl mx-auto w-full z-10">
        <div className="flex items-center gap-2.5 font-bold text-xl tracking-tight">
          <div className="bg-gradient-to-br from-palette-medium to-palette-light p-2 rounded-xl shadow-lg shadow-palette-medium/30">
            <Hand className="w-5 h-5 text-palette-offwhite" />
          </div>
          <span className="text-palette-dark font-extrabold tracking-wide">Voiceout</span>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex-1 flex items-center justify-center p-4 relative z-10">
        <div className="w-full max-w-lg">
          <Button
            variant="ghost"
            className="mb-6 text-palette-dark/80 hover:text-palette-dark hover:bg-palette-dark/10"
            onClick={() => router.push('/')}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>

          {!roomId ? (
            /* Create Room Form */
            <Card className="p-8 md:p-10 bg-white border-0 shadow-2xl">
              <div className="text-center mb-8">
                <div className="w-16 h-16 bg-palette-light/40 flex items-center justify-center rounded-2xl mx-auto mb-4">
                  <Video className="w-8 h-8 text-palette-dark" />
                </div>
                <h1 className="text-3xl font-bold text-palette-dark mb-2">Create Room</h1>
                <p className="text-palette-dark/60">Start your Voiceout session</p>
              </div>

              <div className="space-y-6 mb-8">
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-palette-dark flex items-center gap-2">
                    Your Name
                    <span className="text-palette-dark/40 font-normal text-xs">(required)</span>
                  </label>
                  <Input
                    placeholder="Enter your display name"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    className="h-12 border-2 border-gray-200 hover:border-palette-dark/30 focus-visible:border-palette-dark focus-visible:ring-0 text-palette-dark placeholder:text-palette-dark/40 bg-white"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-semibold text-palette-dark flex items-center gap-2">
                    Room Purpose
                    <span className="text-palette-dark/40 font-normal text-xs">(required)</span>
                  </label>
                  <Input
                    placeholder="e.g., Team Meeting, Client Call"
                    value={roomReason}
                    onChange={(e) => setRoomReason(e.target.value)}
                    className="h-12 border-2 border-gray-200 hover:border-palette-dark/30 focus-visible:border-palette-dark focus-visible:ring-0 text-palette-dark placeholder:text-palette-dark/40 bg-white"
                  />
                </div>
              </div>

              {/* Features */}
              <div className="space-y-3 mb-8 p-5 bg-palette-light/30 rounded-xl border border-palette-light/40">
                <p className="text-xs font-bold uppercase tracking-wide text-palette-dark/60 mb-3">What's included</p>
                <div className="flex items-center gap-3 text-sm">
                  <Check className="w-4 h-4 text-palette-dark flex-shrink-0" />
                  <span className="text-palette-dark/80">Real-time ASL translation</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Check className="w-4 h-4 text-palette-dark flex-shrink-0" />
                  <span className="text-palette-dark/80">End-to-end encryption</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Check className="w-4 h-4 text-palette-dark flex-shrink-0" />
                  <span className="text-palette-dark/80">Up to 10 participants</span>
                </div>
              </div>

              <Button
                onClick={createRoom}
                disabled={!displayName.trim() || !roomReason.trim() || isCreating}
                className="w-full h-14 text-base font-semibold bg-palette-dark hover:bg-palette-dark/90 text-palette-offwhite transition-all hover:-translate-y-0.5 shadow-xl shadow-palette-dark/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
              >
                {isCreating ? (
                  <>
                    <div className="w-4 h-4 border-2 border-palette-medium border-t-transparent rounded-full animate-spin mr-2" />
                    Creating Room...
                  </>
                ) : (
                  <>
                    Create Room
                    <Sparkles className="w-4 h-4 ml-2" />
                  </>
                )}
              </Button>

              <p className="text-xs text-center text-palette-dark/50 mt-4">
                By creating a room, you agree to our privacy policy
              </p>
            </Card>
          ) : (
            /* Room Created Success */
            <Card className="p-10 bg-white border-0 shadow-2xl max-w-xl mx-auto">
              <div className="text-center mb-8 animate-slide-up">
                <div className="w-20 h-20 bg-gradient-to-br from-palette-medium to-palette-light flex items-center justify-center rounded-2xl mx-auto mb-4 shadow-lg shadow-palette-light/50">
                  <Check className="w-10 h-10 text-palette-dark" />
                </div>
                <h1 className="text-3xl font-bold text-palette-dark mb-2">Room Created!</h1>
                <p className="text-palette-dark/60">Your session is ready to go</p>
              </div>

              <div className="space-y-6">
                {/* Room Info */}
                <div className="p-6 bg-palette-light/30 rounded-2xl border-2 border-palette-light/50">
                  <div className="flex items-center gap-3 mb-3">
                    <Video className="w-5 h-5 text-palette-dark" />
                    <span className="text-sm font-semibold text-palette-dark/60">Room Name</span>
                  </div>
                  <p className="text-xl font-bold text-palette-dark">{roomName}</p>
                </div>

                {/* Room Code */}
                <div className="space-y-3">
                  <label className="text-xs font-bold uppercase tracking-widest text-palette-dark/60">
                    Share this code
                  </label>
                  <div className="flex items-center gap-3 p-4 bg-palette-dark rounded-xl">
                    <code className="flex-1 text-3xl font-mono text-center tracking-[0.3em] font-bold text-palette-medium">
                      {roomId}
                    </code>
                    <Button
                      size="icon"
                      onClick={copyRoomId}
                      className="h-12 w-12 bg-palette-medium hover:bg-palette-medium text-palette-dark rounded-lg flex-shrink-0"
                    >
                      {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
                    </Button>
                  </div>
                  {copied && (
                    <p className="text-sm text-palette-dark/60 text-center animate-fade-in">
                      ✓ Code copied to clipboard!
                    </p>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="flex flex-col gap-3 pt-4">
                  <Button
                    onClick={() => router.push(`/call/${roomId}?name=${encodeURIComponent(displayName)}`)}
                    className="w-full h-14 text-base font-semibold bg-palette-dark hover:bg-palette-dark/90 text-palette-offwhite shadow-xl shadow-palette-dark/20"
                  >
                    Join Your Room
                  </Button>
                  <Button
                    onClick={() => router.push('/')}
                    variant="ghost"
                    className="w-full h-12 text-palette-dark hover:bg-palette-dark/5"
                  >
                    Back to Home
                  </Button>
                </div>

                {/* Info Footer */}
                <div className="pt-4 border-t border-palette-dark/10">
                  <div className="flex items-center justify-center gap-6 text-xs text-palette-dark/60">
                    <div className="flex items-center gap-1.5">
                      <Users className="w-4 h-4" />
                      <span>Up to 10 people</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <Shield className="w-4 h-4" />
                      <span>Encrypted</span>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}