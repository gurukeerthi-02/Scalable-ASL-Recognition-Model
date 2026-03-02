'use client';

import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Video, Plus, Users, Hand, MessageSquare, Shield, Zap, ArrowRight, Check } from 'lucide-react';

export default function HomePage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-gradient-to-br from-yellow-400 via-yellow-300 to-yellow-400 text-black flex flex-col">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-yellow-400/90 backdrop-blur-xl border-b border-yellow-500/30 z-50 shadow-sm">
        <div className="p-6 flex justify-between items-center max-w-7xl mx-auto w-full">
          <div className="flex items-center gap-2.5 font-bold text-xl tracking-tight">
            <div className="bg-gradient-to-br from-yellow-400 to-yellow-500 p-2 rounded-xl shadow-lg shadow-yellow-400/30">
              <Hand className="w-5 h-5 text-gray-900" />
            </div>
            <span className="text-black font-bold">ASL Meet</span>
          </div>
          <div className="hidden md:flex gap-8 text-sm font-medium text-black/80">
            <a href="#features" className="hover:text-black transition-colors">Features</a>
            <a href="#about" className="hover:text-black transition-colors">About</a>
            <a href="#safety" className="hover:text-black transition-colors">Safety</a>
          </div>
          <Button
            // variant="outline" 
            size="sm"
            onClick={() => router.push('/join')}
            className="border-2 bg-yellow-300 border-black text-black hover:bg-black hover:text-yellow-400 font-semibold transition-all"
          >
            Sign In
          </Button>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 py-32 pt-40 text-center max-w-5xl mx-auto">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-black/10 border border-black/20 text-sm font-medium mb-8 animate-fade-in">
          <Zap className="w-4 h-4 text-black fill-black" />
          <span className="text-black">Powered by Real-time AI Recognition</span>
        </div>

        <h1 className="text-4xl sm:text-6xl md:text-7xl lg:text-8xl font-extrabold tracking-tight mb-6 leading-[1.1] animate-slide-up">
          <span className="text-black">
            Communication
          </span>
          <br />
          <span className="text-black">
            without barriers
          </span>
        </h1>

        <p className="text-lg md:text-xl text-black/80 mb-10 max-w-2xl animate-slide-up delay-100 leading-relaxed">
          Experience the next generation of video conferencing. ASL Meet translates American Sign Language gestures into text and speech in real-time.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto animate-slide-up delay-200 mb-12">
          <Button
            size="lg"
            className="group px-8 h-14 text-base bg-black hover:bg-black/90 text-yellow-400 font-semibold shadow-xl shadow-black/30 hover:shadow-2xl hover:shadow-black/40 transition-all hover:-translate-y-0.5 border-0"
            onClick={() => router.push('/create')}
          >
            <Plus className="w-5 h-5 mr-2" />
            Start a Call
            <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
          </Button>
          <Button
            size="lg"
            // variant="outline"
            className="px-8 h-14 bg-yellow-300 text-base border-2 border-black hover:border-black hover:bg-black hover:text-yellow-400 text-black font-semibold shadow-lg transition-all"
            onClick={() => router.push('/join')}
          >
            <Users className="w-5 h-5 mr-2" />
            Join Existing
          </Button>
        </div>

        {/* Trust Indicators */}
        <div className="flex flex-wrap items-center justify-center gap-6 text-sm text-black/80 mb-20 animate-slide-up delay-300">
          <div className="flex items-center gap-2">
            <Check className="w-5 h-5 text-black" />
            <span className="font-medium">End-to-end encrypted</span>
          </div>
          <div className="flex items-center gap-2">
            <Check className="w-5 h-5 text-black" />
            <span className="font-medium">HIPAA compliant</span>
          </div>
          <div className="flex items-center gap-2">
            <Check className="w-5 h-5 text-black" />
            <span className="font-medium">Real-time translation</span>
          </div>
        </div>

        {/* Feature Cards */}
        <div id="features" className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full animate-slide-up delay-400">
          <FeatureCard
            icon={<Hand className="w-7 h-7" />}
            title="ASL Recognition"
            description="Our advanced AI processes gestures instantly, enabling seamless signing to speech."
            accentColor="yellow"
          />
          <FeatureCard
            icon={<Shield className="w-7 h-7" />}
            title="Private & Secure"
            description="End-to-end encrypted calls ensure your conversations remain completely private."
            accentColor="gray"
          />
          <FeatureCard
            icon={<MessageSquare className="w-7 h-7" />}
            title="Auto Captioning"
            description="Real-time text-to-speech and captioning for inclusive communication."
            accentColor="yellow"
          />
        </div>
      </main>

      {/* Stats Section */}
      <section className="py-20 px-4 bg-white border-y border-gray-200">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 text-center">
            <div className="animate-slide-up">
              <div className="text-5xl md:text-6xl font-bold text-black mb-2">
                99.8%
              </div>
              <div className="text-black/70 font-medium">Recognition Accuracy</div>
            </div>
            <div className="animate-slide-up delay-100">
              <div className="text-5xl md:text-6xl font-bold text-black mb-2">
                &lt;100ms
              </div>
              <div className="text-black/70 font-medium">Translation Latency</div>
            </div>
            <div className="animate-slide-up delay-200">
              <div className="text-5xl md:text-6xl font-bold text-black mb-2">
                50K+
              </div>
              <div className="text-black/70 font-medium">Active Users</div>
            </div>
          </div>
        </div>
      </section>

      <footer className="p-8 bg-black text-center text-sm text-yellow-400/80 mt-auto">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center gap-2 mb-3">
            <div className="bg-gradient-to-br from-yellow-400 to-yellow-500 p-1.5 rounded-lg">
              <Hand className="w-4 h-4 text-black" />
            </div>
            <span className="text-yellow-400 font-bold">ASL Meet</span>
          </div>
          <p>© 2026 ASL Meet. Bridging the gap through technology.</p>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description, accentColor }: {
  icon: React.ReactNode;
  title: string;
  description: string;
  accentColor: 'yellow' | 'gray';
}) {
  const isYellow = accentColor === 'yellow';

  return (
    <Card className="group relative p-8 text-left bg-white border-gray-200 hover:shadow-2xl hover:border-black/20 transition-all duration-300 overflow-hidden">
      {/* Gradient overlay on hover */}
      <div className="absolute inset-0 bg-gradient-to-br from-yellow-400/0 to-yellow-400/0 group-hover:from-yellow-400/10 group-hover:to-yellow-400/20 transition-all duration-300" />

      <div className="relative">
        <div className={`w-14 h-14 rounded-xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform duration-300 ${isYellow
            ? 'bg-yellow-400/20 text-black'
            : 'bg-black/5 text-black'
          }`}>
          {icon}
        </div>
        <h3 className="font-bold text-xl mb-3 text-black">{title}</h3>
        <p className="text-black/70 text-sm leading-relaxed">
          {description}
        </p>
      </div>
    </Card>
  );
}