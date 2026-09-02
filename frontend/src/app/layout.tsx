import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { ToastProvider } from '@/components/ui/toast';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  title: 'Argus Prime | Multi-Modal Deepfake Detection Platform',
  description: 'Advanced AI-powered deepfake detection for video, audio, and text content. Enterprise-grade security with explainable results.',
  keywords: ['deepfake', 'detection', 'AI', 'security', 'forensics', 'video analysis', 'audio analysis'],
  authors: [{ name: 'Argus Prime Team' }],
  openGraph: {
    title: 'Argus Prime | Multi-Modal Deepfake Detection Platform',
    description: 'Advanced AI-powered deepfake detection for video, audio, and text content.',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Argus Prime | Multi-Modal Deepfake Detection Platform',
    description: 'Advanced AI-powered deepfake detection for video, audio, and text content.',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="icon" href="/icon.svg" type="image/svg+xml" />
        <link rel="manifest" href="/manifest.json" />
      </head>
      <body className="min-h-screen bg-white font-sans antialiased">
        <ToastProvider>
          {children}
        </ToastProvider>
      </body>
    </html>
  );
}
