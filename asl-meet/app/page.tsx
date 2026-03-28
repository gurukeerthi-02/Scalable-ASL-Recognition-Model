'use client';

import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Video, Plus, Users, Hand, MessageSquare, Shield, Zap, ArrowRight, Check, Activity, Cpu, Webhook } from 'lucide-react';

export default function HomePage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-palette-offwhite text-palette-dark flex flex-col font-sans">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-palette-offwhite/90 backdrop-blur-xl border-b border-palette-medium/20 z-50 shadow-sm transition-all duration-300">
        <div className="p-6 flex justify-between items-center max-w-7xl mx-auto w-full">
          <div className="flex items-center gap-2.5 font-bold text-xl tracking-tight">
            <div className="bg-gradient-to-br from-palette-medium to-palette-light p-2 rounded-xl shadow-lg shadow-palette-medium/30">
              <Hand className="w-5 h-5 text-palette-offwhite" />
            </div>
            <span className="text-palette-dark font-extrabold tracking-wide">Voiceout</span>
          </div>
          <div className="hidden md:flex gap-8 text-sm font-semibold text-palette-dark/70">
            <a href="#features" className="hover:text-palette-medium transition-colors">Features</a>
            <a href="#how-it-works" className="hover:text-palette-medium transition-colors">How It Works</a>
            <a href="#metrics" className="hover:text-palette-medium transition-colors">Performance</a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 py-32 pt-40 text-center max-w-5xl mx-auto">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-palette-light/30 border border-palette-light text-sm font-semibold mb-8 animate-fade-in shadow-sm">
          <Activity className="w-4 h-4 text-palette-dark" />
          <span className="text-palette-dark">Powered by Hybrid Modal-Switching (HMS)</span>
        </div>

        <h1 className="text-4xl sm:text-6xl md:text-7xl lg:text-8xl font-black tracking-tighter mb-6 leading-[1.1] animate-slide-up">
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-palette-dark to-palette-medium">
            Communication
          </span>
          <br />
          <span className="text-palette-dark">
            without barriers
          </span>
        </h1>

        <p className="text-lg md:text-xl text-palette-dark/80 mb-10 max-w-3xl animate-slide-up delay-100 leading-relaxed font-medium">
          An end-to-end solution for real-time American Sign Language (ASL) recognition integrated directly within a multi-party WebRTC video conferencing environment.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto animate-slide-up delay-200 mb-12">
          <Button
            size="lg"
            className="group px-8 h-14 text-base bg-palette-dark hover:bg-palette-dark/90 text-palette-offwhite font-bold shadow-xl shadow-palette-dark/20 hover:shadow-2xl hover:shadow-palette-dark/30 transition-all hover:-translate-y-0.5 border-0"
            onClick={() => router.push('/create')}
          >
            <Plus className="w-5 h-5 mr-2" />
            Start a Call
            <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
          </Button>
          <Button
            size="lg"
            className="px-8 h-14 bg-palette-offwhite text-base border-2 border-palette-medium hover:border-palette-dark hover:bg-palette-light/30 text-palette-dark font-bold shadow-lg shadow-palette-light/20 transition-all"
            onClick={() => router.push('/join')}
          >
            <Users className="w-5 h-5 mr-2 text-palette-medium" />
            Join Existing
          </Button>
        </div>

        {/* Tech Stack Indicators */}
        <div className="flex flex-wrap items-center justify-center gap-6 text-sm text-palette-dark/80 mb-20 animate-slide-up delay-300">
          <div className="flex items-center gap-2">
            <div className="bg-palette-light/50 p-1 rounded-full">
              <Check className="w-4 h-4 text-palette-dark" />
            </div>
            <span className="font-semibold">Next.js Frontend</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="bg-palette-light/50 p-1 rounded-full">
              <Check className="w-4 h-4 text-palette-dark" />
            </div>
            <span className="font-semibold">Node.js/Socket.IO</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="bg-palette-light/50 p-1 rounded-full">
              <Check className="w-4 h-4 text-palette-dark" />
            </div>
            <span className="font-semibold">Flask Inference Engine</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="bg-palette-light/50 p-1 rounded-full">
              <Check className="w-4 h-4 text-palette-dark" />
            </div>
            <span className="font-semibold">Supabase PostgreSQL</span>
          </div>
        </div>

        {/* Feature Cards */}
        <div id="features" className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full animate-slide-up delay-400">
          <FeatureCard
            icon={<Cpu className="w-7 h-7" />}
            title="HMS Architecture"
            description="Dynamically switches between a DNN-based Static Gesture Network (SGN) and an LSTM-based Dynamic Sequence Network (DSN)."
            accentColor="primary"
          />
          <FeatureCard
            icon={<Webhook className="w-7 h-7" />}
            title="WebRTC Integration"
            description="Browser-native ASL translation powered by a distributed four-layer stack, achieving ultra-low latency within secure multi-party calls."
            accentColor="secondary"
          />
          <FeatureCard
            icon={<Activity className="w-7 h-7" />}
            title="Robust Generalisation"
            description="Proven through Leave-One-Person-Out CV on 48,000 samples across 24 ASL classes, demonstrating robust generalisation to unseen signers."
            accentColor="primary"
          />
        </div>
      </main>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-24 px-4 max-w-7xl mx-auto w-full">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-black text-palette-dark mb-4 tracking-tight">How It Works</h2>
          <p className="text-palette-dark/70 text-lg max-w-2xl mx-auto font-medium">A seamless pipeline designed for ultra-low latency sign language translation.</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="p-8 rounded-3xl bg-white border border-palette-medium/20 shadow-lg relative">
            <div className="absolute -top-6 left-8 w-12 h-12 bg-palette-dark text-white rounded-full flex items-center justify-center text-xl font-bold shadow-xl border-4 border-palette-offwhite">1</div>
            <h3 className="text-xl font-black text-palette-dark mb-3 mt-4">Capture & Optimize</h3>
            <p className="text-palette-dark/70 font-medium leading-relaxed">Browser-native WebRTC captures video safely and efficiently using JPEG compression and 10Hz frame throttling to drastically reduce payload overhead.</p>
          </div>
          <div className="p-8 rounded-3xl bg-white border border-palette-medium/20 shadow-lg relative">
            <div className="absolute -top-6 left-8 w-12 h-12 bg-palette-dark text-white rounded-full flex items-center justify-center text-xl font-bold shadow-xl border-4 border-palette-offwhite">2</div>
            <h3 className="text-xl font-black text-palette-dark mb-3 mt-4">Hybrid Inference</h3>
            <p className="text-palette-dark/70 font-medium leading-relaxed">Our custom Flask engine dynamically routes standard ASL alphabet gestures to the SGN, while fluid motions automatically trigger the LSTM-powered DSN.</p>
          </div>
          <div className="p-8 rounded-3xl bg-white border border-palette-medium/20 shadow-lg relative">
            <div className="absolute -top-6 left-8 w-12 h-12 bg-palette-dark text-white rounded-full flex items-center justify-center text-xl font-bold shadow-xl border-4 border-palette-offwhite">3</div>
            <h3 className="text-xl font-black text-palette-dark mb-3 mt-4">Real-Time Sync</h3>
            <p className="text-palette-dark/70 font-medium leading-relaxed">Accurate results are persisted via Supabase and broadcasted globally through Node.js/Socket.IO, ensuring all participants sync simultaneously.</p>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section id="metrics" className="py-20 px-4 bg-palette-light/20 border-y border-palette-medium/20 mt-16">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 text-center">
            <div className="animate-slide-up">
              <div className="text-5xl md:text-6xl font-black text-palette-dark mb-2 tracking-tight">
                97.36%
              </div>
              <div className="text-palette-dark/70 font-bold uppercase tracking-wider text-sm">Static Network (SGN) Accuracy</div>
            </div>
            <div className="animate-slide-up delay-100">
              <div className="text-5xl md:text-6xl font-black text-palette-dark mb-2 tracking-tight">
                20.9<span className="text-4xl">ms</span>
              </div>
              <div className="text-palette-dark/70 font-bold uppercase tracking-wider text-sm">Steady-State Latency</div>
            </div>
            <div className="animate-slide-up delay-200">
              <div className="text-5xl md:text-6xl font-black text-palette-dark mb-2 tracking-tight">
                89.08%
              </div>
              <div className="text-palette-dark/70 font-bold uppercase tracking-wider text-sm">Dynamic Network (DSN) Accuracy</div>
            </div>
          </div>
        </div>
      </section>

      <footer className="p-8 bg-palette-dark text-center text-sm text-palette-offwhite/80 mt-auto border-t border-palette-dark/20">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center gap-2 mb-4">
            <div className="bg-gradient-to-br from-palette-medium to-palette-light p-1.5 rounded-lg shadow-md">
              <Hand className="w-5 h-5 text-palette-offwhite" />
            </div>
            <span className="text-palette-offwhite font-bold text-lg tracking-wide">Voiceout</span>
          </div>
          <p className="font-medium text-palette-light">© 2026 Voiceout. An advanced sign language recognition platform.</p>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description, accentColor }: {
  icon: React.ReactNode;
  title: string;
  description: string;
  accentColor: 'primary' | 'secondary';
}) {
  const isPrimary = accentColor === 'primary';

  return (
    <Card className="group relative p-8 text-left bg-palette-offwhite border-palette-medium/30 hover:shadow-2xl hover:shadow-palette-medium/20 hover:border-palette-medium transition-all duration-300 overflow-hidden shadow-lg shadow-palette-medium/5">
      {/* Gradient overlay on hover */}
      <div className="absolute inset-0 bg-gradient-to-br from-palette-light/0 to-palette-medium/0 group-hover:from-palette-light/10 group-hover:to-palette-medium/10 transition-all duration-500" />

      <div className="relative z-10">
        <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-6 shadow-sm group-hover:scale-110 group-hover:rotate-3 transition-transform duration-500 ${isPrimary
            ? 'bg-gradient-to-br from-palette-medium to-palette-light text-palette-dark'
            : 'bg-palette-dark text-palette-offwhite'
          }`}>
          {icon}
        </div>
        <h3 className="font-black text-2xl mb-3 text-palette-dark tracking-tight">{title}</h3>
        <p className="text-palette-dark/70 text-base leading-relaxed font-medium">
          {description}
        </p>
      </div>
    </Card>
  );
}