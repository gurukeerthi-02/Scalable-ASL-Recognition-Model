'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Users, AlertCircle, ArrowLeft, Key, User, Hand, Check } from 'lucide-react';
import { supabase } from '@/lib/supabase';

export default function JoinCallPage() {
  const router = useRouter();
  const [displayName, setDisplayName] = useState('');
  const [roomId, setRoomId] = useState('');
  const [isJoining, setIsJoining] = useState(false);
  const [error, setError] = useState('');

  const joinCall = async () => {
    if (!displayName.trim() || !roomId.trim()) return;

    setIsJoining(true);
    setError('');

    try {
      const { data: roomData, error: roomError } = await supabase
        .from('rooms')
        .select('*')
        .eq('id', roomId.trim())
        .eq('is_active', true)
        .maybeSingle();

      if (roomError) throw roomError;

      if (!roomData) {
        setError('Room not found or session has ended');
        return;
      }

      router.push(`/call/${roomId.trim()}?name=${encodeURIComponent(displayName)}`);
    } catch (error) {
      console.error('Error joining room:', error);
      setError('Connection failed. Please check your room ID.');
    } finally {
      setIsJoining(false);
    }
  };

  return (
    <div className="min-h-screen bg-palette-offwhite flex flex-col relative overflow-hidden">
      {/* Decorative Elements */}
      <div className="absolute top-20 right-10 w-64 h-64 bg-palette-dark/5 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-20 left-10 w-96 h-96 bg-palette-dark/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      
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

          <Card className="p-8 md:p-10 bg-white border-0 shadow-2xl">
            <div className="text-center mb-8">
              <div className="w-16 h-16 bg-palette-light/40 flex items-center justify-center rounded-2xl mx-auto mb-4">
                <Users className="w-8 h-8 text-palette-dark" />
              </div>
              <h1 className="text-3xl font-bold text-palette-dark mb-2">Join Room</h1>
              <p className="text-palette-dark/60">Enter your details to join the session</p>
            </div>

            <div className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-palette-dark flex items-center gap-2">
                  <User className="w-4 h-4 text-palette-dark/50" />
                  Your Name
                  <span className="text-palette-dark/40 font-normal text-xs">(required)</span>
                </label>
                <Input
                  placeholder="Enter your display name"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  className="h-12 border-2 border-gray-200 hover:border-palette-dark/30 focus-visible:border-palette-dark focus-visible:ring-0 text-palette-dark placeholder:text-palette-dark/40 bg-white"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && displayName.trim() && roomId.trim() && !isJoining) {
                      joinCall();
                    }
                  }}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-semibold text-palette-dark flex items-center gap-2">
                  <Key className="w-4 h-4 text-palette-dark/50" />
                  Room Code
                  <span className="text-palette-dark/40 font-normal text-xs">(required)</span>
                </label>
                <Input
                  placeholder="e.g., abc123"
                  value={roomId}
                  onChange={(e) => setRoomId(e.target.value.toLowerCase())}
                  className="h-12 border-2 border-gray-200 hover:border-palette-dark/30 focus-visible:border-palette-dark focus-visible:ring-0 text-palette-dark placeholder:text-palette-dark/40 bg-white font-mono text-center tracking-[0.3em] text-lg uppercase"
                  maxLength={6}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && displayName.trim() && roomId.trim() && !isJoining) {
                      joinCall();
                    }
                  }}
                />
                <p className="text-xs text-palette-dark/50 ml-1">
                  Ask the host for the 6-character room code
                </p>
              </div>

              {error && (
                <div className="flex items-center gap-3 p-4 bg-red-50 border-2 border-red-200 rounded-xl animate-fade-in">
                  <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-600" />
                  <span className="text-sm font-medium text-red-700">{error}</span>
                </div>
              )}

              {/* Quick Info */}
              <div className="p-4 bg-palette-light/30 rounded-xl border border-palette-light/40">
                <p className="text-xs font-semibold text-palette-dark mb-2 flex items-center gap-2">
                  <Check className="w-4 h-4" />
                  You're about to join
                </p>
                <ul className="space-y-1.5 ml-6">
                  <li className="text-xs text-palette-dark/70">• A secure, encrypted video call</li>
                  <li className="text-xs text-palette-dark/70">• With real-time ASL translation</li>
                  <li className="text-xs text-palette-dark/70">• HIPAA compliant session</li>
                </ul>
              </div>

              <Button
                onClick={joinCall}
                disabled={!displayName.trim() || !roomId.trim() || isJoining}
                className="w-full h-14 text-base font-semibold bg-palette-dark hover:bg-palette-dark/90 text-palette-offwhite transition-all hover:-translate-y-0.5 shadow-xl shadow-palette-dark/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
              >
                {isJoining ? (
                  <>
                    <div className="w-4 h-4 border-2 border-palette-medium border-t-transparent rounded-full animate-spin mr-2" />
                    Joining Room...
                  </>
                ) : (
                  <>
                    Join Room
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </>
                )}
              </Button>

              <p className="text-xs text-center text-palette-dark/50">
                By joining, you agree to our privacy policy
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function ArrowRight({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
    </svg>
  );
}