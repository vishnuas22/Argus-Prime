'use client';

import Link from 'next/link';
import { ArrowRight, Shield, Video, Music, FileText, Zap, Lock, Globe, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

const features = [
  {
    icon: Video,
    title: 'Video Detection',
    description: 'Advanced AI models analyze facial inconsistencies, artifacts, and temporal coherence.',
    gradient: 'from-blue-500 to-indigo-500',
  },
  {
    icon: Music,
    title: 'Audio Analysis',
    description: 'Detect synthetic speech patterns and vocoder artifacts with high precision.',
    gradient: 'from-purple-500 to-pink-500',
  },
  {
    icon: FileText,
    title: 'Text Forensics',
    description: 'Perplexity and burstiness analysis to identify AI-generated content.',
    gradient: 'from-amber-500 to-orange-500',
  },
  {
    icon: Shield,
    title: 'Multi-Modal Fusion',
    description: 'Combine insights from video, audio, and text for comprehensive analysis.',
    gradient: 'from-emerald-500 to-teal-500',
  },
  {
    icon: Zap,
    title: 'Real-time Processing',
    description: 'Fast analysis with WebSocket progress streaming and instant results.',
    gradient: 'from-red-500 to-rose-500',
  },
  {
    icon: Lock,
    title: 'Enterprise Security',
    description: 'Adversarial defense and secure processing for sensitive content.',
    gradient: 'from-violet-500 to-purple-500',
  },
];

const stats = [
  { value: '99.7%', label: 'Detection Accuracy' },
  { value: '<2s', label: 'Avg. Analysis Time' },
  { value: '500MB', label: 'Max File Size' },
  { value: '24/7', label: 'Active Monitoring' },
];

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-primary-50/30 to-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-lg border-b border-secondary-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-600 to-purple-600 rounded-xl flex items-center justify-center">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
                Argus Prime
              </span>
            </div>
            
            <div className="hidden md:flex items-center gap-8">
              <Link href="#features" className="text-secondary-600 hover:text-secondary-900 transition-colors">
                Features
              </Link>
              <Link href="#how-it-works" className="text-secondary-600 hover:text-secondary-900 transition-colors">
                How It Works
              </Link>
              <Link href="#pricing" className="text-secondary-600 hover:text-secondary-900 transition-colors">
                Pricing
              </Link>
            </div>

            <div className="flex items-center gap-4">
              <Link href="/login">
                <Button variant="ghost">Sign In</Button>
              </Link>
              <Link href="/analyze">
                <Button>
                  Get Started
                  <ArrowRight className="ml-2 w-4 h-4" />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center max-w-4xl mx-auto">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary-50 rounded-full text-primary-700 text-sm font-medium mb-8">
              <Sparkles className="w-4 h-4" />
              <span>Next-Generation Deepfake Detection</span>
            </div>
            
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-secondary-900 mb-6 leading-tight">
              Detect Deepfakes with{' '}
              <span className="bg-gradient-to-r from-primary-600 via-purple-600 to-pink-500 bg-clip-text text-transparent">
                Surgical Precision
              </span>
            </h1>
            
            <p className="text-xl text-secondary-600 mb-10 max-w-2xl mx-auto leading-relaxed">
              Advanced multi-modal AI analysis for video, audio, and text content. 
              Enterprise-grade security with explainable results you can trust.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/analyze">
                <Button size="xl" className="w-full sm:w-auto">
                  Analyze Content
                  <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </Link>
              <Link href="#features">
                <Button variant="outline" size="xl" className="w-full sm:w-auto">
                  Explore Features
                </Button>
              </Link>
            </div>
          </div>

          {/* Stats */}
          <div className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
                  {stat.value}
                </div>
                <div className="text-secondary-500 mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4 sm:px-6 lg:px-8 bg-secondary-50/50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-secondary-900 mb-4">
              Comprehensive Detection Capabilities
            </h2>
            <p className="text-lg text-secondary-600 max-w-2xl mx-auto">
              Our advanced AI models analyze multiple modalities to detect deepfakes 
              with unprecedented accuracy and provide detailed forensic reports.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <Card key={index} hoverable className="group">
                <CardContent className="p-6">
                  <div
                    className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}
                  >
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-secondary-900 mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-secondary-600">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="relative rounded-3xl overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-primary-600 to-purple-600" />
            <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-20" />
            <div className="relative px-8 py-16 sm:px-16 sm:py-20 text-center">
              <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
                Ready to Detect Deepfakes?
              </h2>
              <p className="text-lg text-white/80 mb-8 max-w-2xl mx-auto">
                Start analyzing content with our advanced AI-powered detection platform. 
                Get started for free today.
              </p>
              <Link href="/analyze">
                <Button size="lg" variant="secondary" className="bg-white text-primary-600 hover:bg-white/90">
                  Start Free Analysis
                  <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 sm:px-6 lg:px-8 border-t border-secondary-100">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-primary-600 to-purple-600 rounded-lg flex items-center justify-center">
                <Shield className="w-4 h-4 text-white" />
              </div>
              <span className="font-semibold text-secondary-900">Argus Prime</span>
            </div>
            
            <div className="flex items-center gap-6 text-sm text-secondary-500">
              <Link href="/privacy" className="hover:text-secondary-900 transition-colors">
                Privacy Policy
              </Link>
              <Link href="/terms" className="hover:text-secondary-900 transition-colors">
                Terms of Service
              </Link>
              <Link href="/contact" className="hover:text-secondary-900 transition-colors">
                Contact
              </Link>
            </div>

            <div className="text-sm text-secondary-500">
              © {new Date().getFullYear()} Argus Prime. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
