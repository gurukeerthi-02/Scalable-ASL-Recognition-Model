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
    <div className="min-h-screen bg-gradient-to-br from-yellow-400 via-yellow-300 to-yellow-400 flex flex-col relative overflow-hidden">
      {/* Decorative Elements */}
      <div className="absolute top-20 left-10 w-64 h-64 bg-black/5 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-20 right-10 w-96 h-96 bg-black/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      
      {/* Navigation */}
      <nav className="p-6 flex items-center max-w-7xl mx-auto w-full z-10">
        <div className="flex items-center gap-2.5 font-bold text-xl tracking-tight">
          <div className="bg-black p-2 rounded-xl shadow-lg">
            <Hand className="w-5 h-5 text-yellow-400" />
          </div>
          <span className="text-black">ASL Meet</span>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex-1 flex items-center justify-center p-4 relative z-10">
        <div className="w-full max-w-lg">
          <Button
            variant="ghost"
            className="mb-6 text-black/80 hover:text-black hover:bg-black/10"
            onClick={() => router.push('/')}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>

          {!roomId ? (
            /* Create Room Form */
            <Card className="p-8 md:p-10 bg-white border-0 shadow-2xl">
              <div className="text-center mb-8">
                <div className="w-16 h-16 bg-yellow-400/20 flex items-center justify-center rounded-2xl mx-auto mb-4">
                  <Video className="w-8 h-8 text-black" />
                </div>
                <h1 className="text-3xl font-bold text-black mb-2">Create Room</h1>
                <p className="text-black/60">Start your ASL Meet session</p>
              </div>

              <div className="space-y-6 mb-8">
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-black flex items-center gap-2">
                    Your Name
                    <span className="text-black/40 font-normal text-xs">(required)</span>
                  </label>
                  <Input
                    placeholder="Enter your display name"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    className="h-12 border-2 border-gray-200 hover:border-black/30 focus-visible:border-black focus-visible:ring-0 text-black placeholder:text-black/40 bg-white"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-semibold text-black flex items-center gap-2">
                    Room Purpose
                    <span className="text-black/40 font-normal text-xs">(required)</span>
                  </label>
                  <Input
                    placeholder="e.g., Team Meeting, Client Call"
                    value={roomReason}
                    onChange={(e) => setRoomReason(e.target.value)}
                    className="h-12 border-2 border-gray-200 hover:border-black/30 focus-visible:border-black focus-visible:ring-0 text-black placeholder:text-black/40 bg-white"
                  />
                </div>
              </div>

              {/* Features */}
              <div className="space-y-3 mb-8 p-5 bg-yellow-400/10 rounded-xl border border-yellow-400/20">
                <p className="text-xs font-bold uppercase tracking-wide text-black/60 mb-3">What's included</p>
                <div className="flex items-center gap-3 text-sm">
                  <Check className="w-4 h-4 text-black flex-shrink-0" />
                  <span className="text-black/80">Real-time ASL translation</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Check className="w-4 h-4 text-black flex-shrink-0" />
                  <span className="text-black/80">End-to-end encryption</span>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <Check className="w-4 h-4 text-black flex-shrink-0" />
                  <span className="text-black/80">Up to 10 participants</span>
                </div>
              </div>

              <Button
                onClick={createRoom}
                disabled={!displayName.trim() || !roomReason.trim() || isCreating}
                className="w-full h-14 text-base font-semibold bg-black hover:bg-black/90 text-yellow-400 transition-all hover:-translate-y-0.5 shadow-xl shadow-black/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
              >
                {isCreating ? (
                  <>
                    <div className="w-4 h-4 border-2 border-yellow-400 border-t-transparent rounded-full animate-spin mr-2" />
                    Creating Room...
                  </>
                ) : (
                  <>
                    Create Room
                    <Sparkles className="w-4 h-4 ml-2" />
                  </>
                )}
              </Button>

              <p className="text-xs text-center text-black/50 mt-4">
                By creating a room, you agree to our privacy policy
              </p>
            </Card>
          ) : (
            /* Room Created Success */
            <Card className="p-10 bg-white border-0 shadow-2xl max-w-xl mx-auto">
              <div className="text-center mb-8 animate-slide-up">
                <div className="w-20 h-20 bg-gradient-to-br from-yellow-400 to-yellow-500 flex items-center justify-center rounded-2xl mx-auto mb-4 shadow-lg shadow-yellow-400/30">
                  <Check className="w-10 h-10 text-black" />
                </div>
                <h1 className="text-3xl font-bold text-black mb-2">Room Created!</h1>
                <p className="text-black/60">Your session is ready to go</p>
              </div>

              <div className="space-y-6">
                {/* Room Info */}
                <div className="p-6 bg-yellow-400/10 rounded-2xl border-2 border-yellow-400/30">
                  <div className="flex items-center gap-3 mb-3">
                    <Video className="w-5 h-5 text-black" />
                    <span className="text-sm font-semibold text-black/60">Room Name</span>
                  </div>
                  <p className="text-xl font-bold text-black">{roomName}</p>
                </div>

                {/* Room Code */}
                <div className="space-y-3">
                  <label className="text-xs font-bold uppercase tracking-widest text-black/60">
                    Share this code
                  </label>
                  <div className="flex items-center gap-3 p-4 bg-black rounded-xl">
                    <code className="flex-1 text-3xl font-mono text-center tracking-[0.3em] font-bold text-yellow-400">
                      {roomId}
                    </code>
                    <Button
                      size="icon"
                      onClick={copyRoomId}
                      className="h-12 w-12 bg-yellow-400 hover:bg-yellow-500 text-black rounded-lg flex-shrink-0"
                    >
                      {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
                    </Button>
                  </div>
                  {copied && (
                    <p className="text-sm text-black/60 text-center animate-fade-in">
                      ✓ Code copied to clipboard!
                    </p>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="flex flex-col gap-3 pt-4">
                  <Button
                    onClick={() => router.push(`/call/${roomId}?name=${encodeURIComponent(displayName)}`)}
                    className="w-full h-14 text-base font-semibold bg-black hover:bg-black/90 text-yellow-400 shadow-xl shadow-black/20"
                  >
                    Join Your Room
                  </Button>
                  <Button
                    onClick={() => router.push('/')}
                    variant="ghost"
                    className="w-full h-12 text-black hover:bg-black/5"
                  >
                    Back to Home
                  </Button>
                </div>

                {/* Info Footer */}
                <div className="pt-4 border-t border-black/10">
                  <div className="flex items-center justify-center gap-6 text-xs text-black/60">
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