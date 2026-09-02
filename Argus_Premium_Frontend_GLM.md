# Argus Prime Frontend Implementation Guide

**Version:** 1.0.0  
**Date:** January 22, 2026  
**Status:** Production Ready  
**Framework:** Next.js 14 (App Router) + TypeScript + Tailwind CSS

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Backend API Integration](#2-backend-api-integration)
3. [Technology Stack](#3-technology-stack)
4. [Architecture Design](#4-architecture-design)
5. [Component Specifications](#5-component-specifications)
6. [State Management](#6-state-management)
7. [UI/UX Design System](#7-uiux-design-system)
8. [Feature Implementation](#8-feature-implementation)
9. [Performance Optimization](#9-performance-optimization)
10. [Security Implementation](#10-security-implementation)
11. [Testing Strategy](#11-testing-strategy)
12. [Deployment Configuration](#12-deployment-configuration)

---

## 1. Project Overview

### 1.1 Product Description

**Argus Prime** is a production-ready Multi-Modal Deepfake Detection platform that analyzes video, audio, image, and text content to detect deepfakes with forensic precision. The system uses multiple AI models across different modalities and combines results through adaptive fusion to provide accurate, explainable results.

### 1.2 Core Features

- **Multi-Modal Analysis**: Video, audio, and text detection
- **Real-time Progress**: WebSocket-based progress streaming
- **Explainable Results**: Heatmaps, localization, and reasoning
- **Trust Scoring**: Confidence-calibrated trust scores
- **Forensic Reports**: Comprehensive JSON reports with evidence
- **Adversarial Defense**: Protection against adversarial attacks
- **Prometheus Metrics**: Built-in observability

### 1.3 Target Users

- **Security Analysts**: Real-time content verification
- **Forensic Investigators**: Detailed evidence collection
- **Enterprise Security Teams**: Automated content filtering
- **Media Organizations**: Authenticity verification

---

## 2. Backend API Integration

### 2.1 API Base Configuration

```typescript
// config/api.ts
export const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
  timeout: {
    request: 30000,
    upload: 300000,
    websocket: 3600000,
  },
  retries: {
    maxAttempts: 3,
    backoffMultiplier: 2,
    initialDelay: 1000,
  },
};
```

### 2.2 Endpoint Definitions

#### 2.2.1 Analysis Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze` | Submit file for analysis |
| GET | `/api/v1/analyze/{id}/status` | Get analysis progress |
| GET | `/api/v1/analyze/{id}/result` | Get complete analysis result |
| GET | `/api/v1/analyses` | List analyses with pagination |
| GET | `/api/v1/analyses/{id}` | Get detailed analysis |
| WS | `/api/v1/analyze/{id}/stream` | Real-time progress WebSocket |

#### 2.2.2 Health Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Basic health check |
| GET | `/api/v1/health/detailed` | Detailed component status |
| GET | `/api/v1/health/ready` | Readiness probe |
| GET | `/api/v1/health/live` | Liveness probe |
| GET | `/api/v1/config` | Public configuration |
| GET | `/metrics` | Prometheus metrics |

### 2.3 Request/Response Schemas

#### 2.3.1 Analysis Submission

```typescript
// types/api/analysis.ts

interface AnalysisSubmitRequest {
  file: File;
  options?: {
    video?: VideoOptions;
    audio?: AudioOptions;
    text?: TextOptions;
  };
}

interface VideoOptions {
  fps?: number;          // 1-30, default 5
  extractFaces?: boolean; // default true
  generateHeatmaps?: boolean; // default true
  rppgAnalysis?: boolean; // default true
}

interface AudioOptions {
  sampleRate?: number;    // 8000-48000, default 16000
  segmentDuration?: number; // 1-10, default 2
  applyVad?: boolean;     // default true
}

interface TextOptions {
  language?: string;
  includePerplexity?: boolean; // default true
  includeBurstiness?: boolean; // default true
}

interface AnalysisSubmitResponse {
  request_id: string;
  status: 'pending';
  message: string;
  websocket_url: string;
}
```

#### 2.3.2 Analysis Status

```typescript
type AnalysisStatus = 
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed';

interface AnalysisStatusResponse {
  request_id: string;
  status: AnalysisStatus;
  progress: number;        // 0-100
  current_stage?: string;
  estimated_time_remaining?: number;
  error_message?: string;
}

type Verdict = 
  | 'AUTHENTIC'
  | 'MANIPULATED'
  | 'UNCERTAIN';

type ConfidenceLevel = 'high' | 'medium' | 'low';
```

#### 2.3.3 Analysis Result

```typescript
interface ForensicReport {
  report_id: string;
  request_id: string;
  created_at: string;
  file_info: {
    file_name: string;
    file_type: string;
    file_size: number;
    content_type: string;
  };
  executive_summary: {
    verdict: Verdict;
    trust_score: number;
    confidence: number;
  };
  trust_score: {
    overall_score: number;      // 0-100
    confidence: number;          // 0-1
    confidence_level: ConfidenceLevel;
    breakdown: {
      video: { weight: number; contribution: number; percentage: number };
      audio: { weight: number; contribution: number; percentage: number };
      text: { weight: number; contribution: number; percentage: number };
    };
  };
  findings: {
    video?: VideoFinding;
    audio?: AudioFinding;
    text?: TextFinding;
  };
  evidence: {
    heatmaps?: string[];
    spectrograms?: string[];
    localization: {
      timestamps: Array<{ start: number; end: number; confidence: number }>;
      spatial_regions: Array<{ polygon: number[][]; confidence: number }>;
    };
    c2pa_metadata?: C2PAMetadata;
  };
  recommendations: string[];
}

interface VideoFinding {
  frames_analyzed: number;
  faces_detected: number;
  overall_score: number;
  per_frame_scores: Array<{
    frame_number: number;
    timestamp: number;
    face_bbox?: number[];
    manipulation_score: number;
  }>;
  branch_scores: {
    global: number;
    texture: number;
    spectral: number;
  };
}

interface AudioFinding {
  duration_seconds: number;
  segments_analyzed: number;
  overall_score: number;
  per_segment_scores: Array<{
    start_time: number;
    end_time: number;
    manipulation_score: number;
    vocoder_class?: string;
  }>;
}

interface TextFinding {
  word_count: number;
  language_detected: string;
  overall_score: number;
  per_sentence_scores: Array<{
    sentence: string;
    manipulation_score: number;
    perplexity: number;
    burstiness: number;
  }>;
}
```

### 2.4 WebSocket Progress Streaming

```typescript
interface ProgressUpdate {
  request_id: string;
  stage: string;  // 'extracting_frames', 'running_video_detection', etc.
  progress: number; // 0-100
  message: string;
  timestamp: string;
}

type ProgressCallback = (update: ProgressUpdate) => void;
```

---

## 3. Technology Stack

### 3.1 Core Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| Framework | Next.js | 14.x | React framework with App Router |
| Language | TypeScript | 5.x | Type-safe development |
| Styling | Tailwind CSS | 3.x | Utility-first styling |
| State Management | Zustand | 4.x | Lightweight global state |
| Data Fetching | TanStack Query | 5.x | Server state management |
| Charts | Recharts | 2.x | Data visualization |
| Icons | Lucide React | 0.x | Icon system |
| Animations | Framer Motion | 10.x | Complex animations |
| Forms | React Hook Form | 7.x | Form handling |
| Validation | Zod | 3.x | Schema validation |
| HTTP Client | Axios | 1.x | API client |
| Testing | Vitest | 1.x | Unit testing |
| E2E Testing | Playwright | 1.x | Browser testing |

### 3.2 Recommended Design System

For premium UI components, integrate:

```bash
# shadcn/ui for accessible components
npx shadcn@latest init

# Premium icon library
npm install lucide-react

# Chart library
npm install recharts

# Animation library
npm install framer-motion
```

### 3.3 Browser Support

- Chrome (last 2 versions)
- Firefox (last 2 versions)
- Safari (last 2 versions)
- Edge (last 2 versions)

---

## 4. Architecture Design

### 4.1 Folder Structure

```
src/
├── app/                          # Next.js App Router
│   ├── (auth)/                   # Authentication routes
│   │   ├── login/
│   │   └── register/
│   ├── (dashboard)/              # Protected routes
│   │   ├── layout.tsx
│   │   ├── page.tsx             # Dashboard home
│   │   ├── analyze/
│   │   │   ├── page.tsx         # New analysis
│   │   │   └── [id]/            # Analysis details
│   │   │       ├── page.tsx
│   │   │       └── results/
│   │   ├── history/
│   │   │   └── page.tsx
│   │   ├── reports/
│   │   │   └── page.tsx
│   │   └── settings/
│   │       └── page.tsx
│   ├── api/                      # API routes (optional)
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx                  # Landing page
├── components/
│   ├── ui/                       # Base UI components (atoms)
│   │   ├── button/
│   │   ├── card/
│   │   ├── input/
│   │   ├── modal/
│   │   ├── progress/
│   │   ├── select/
│   │   ├── table/
│   │   ├── tabs/
│   │   └── toast/
│   ├── shared/                   # Shared components (molecules)
│   │   ├── header/
│   │   ├── sidebar/
│   │   ├── footer/
│   │   └── loading/
│   ├── analysis/                 # Feature components (organisms)
│   │   ├── upload/
│   │   ├── progress/
│   │   ├── results/
│   │   ├── heatmap/
│   │   ├── charts/
│   │   └── verdict/
│   └── dashboard/                # Dashboard components
├── hooks/                        # Custom React hooks
│   ├── useAnalysis.ts
│   ├── useUpload.ts
│   ├── useWebSocket.ts
│   └── useAuth.ts
├── lib/                          # Utility functions
│   ├── api/
│   │   ├── client.ts
│   │   ├── endpoints.ts
│   │   └── axios.ts
│   ├── utils/
│   │   ├── formatters.ts
│   │   ├── validators.ts
│   │   └── helpers.ts
│   └── config.ts
├── services/                     # API services
│   ├── analysis.service.ts
│   ├── health.service.ts
│   └── report.service.ts
├── stores/                       # Global state
│   ├── analysis.store.ts
│   ├── ui.store.ts
│   └── auth.store.ts
├── types/                        # TypeScript types
│   ├── api/
│   ├── components/
│   └── store/
├── contexts/                     # React contexts
│   ├── analysis.context.tsx
│   └── theme.context.tsx
└── styles/                       # Global styles
    └── fonts.css
```

### 4.2 Component Hierarchy

```
Landing Page
├── HeroSection
│   ├── TypewriterText
│   ├── FeatureCards
│   └── CTAButton
├── Navbar
│   ├── Logo
│   ├── NavigationLinks
│   └── AuthButtons
└── Footer

Dashboard Layout
├── Sidebar
│   ├── NavigationMenu
│   ├── UserProfile
│   └── SettingsButton
├── Header
│   ├── SearchBar
│   ├── Notifications
│   └── UserMenu
└── MainContent
    └── Page Content

Analysis Page
├── UploadZone
│   ├── FileDropzone
│   ├── FilePreview
│   └── OptionsSelector
└── ProgressSection
    ├── StageIndicator
    ├── ProgressBar
    └── StreamingUpdates

Results Page
├── ExecutiveSummary
│   ├── TrustScore
│   └── VerdictBadge
├── FindingsTabs
│   ├── VideoFindings
│   ├── AudioFindings
│   └── TextFindings
├── EvidenceSection
│   ├── HeatmapViewer
│   ├── LocalizationTimeline
│   └── SpectrogramViewer
└── ActionButtons
    ├── DownloadReport
    └── ShareResults
```

---

## 5. Component Specifications

### 5.1 Base UI Components (Atoms)

#### 5.1.1 Button

```typescript
// components/ui/button/button.types.ts
interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  disabled?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  fullWidth?: boolean;
  children: React.ReactNode;
  onClick?: () => void;
}

// components/ui/button/button.tsx
import { cva, type VariantProps } from 'class-variance-authority';
import { forwardRef } from 'react';
import { Loader2 } from 'lucide-react';

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-lg font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        primary: 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-700 hover:to-purple-700 focus-visible:ring-indigo-500',
        secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 focus-visible:ring-gray-500',
        outline: 'border-2 border-gray-200 bg-transparent hover:bg-gray-50 focus-visible:ring-gray-500',
        ghost: 'bg-transparent hover:bg-gray-100 focus-visible:ring-gray-500',
        danger: 'bg-red-600 text-white hover:bg-red-700 focus-visible:ring-red-500',
      },
      size: {
        sm: 'h-8 px-3 text-sm',
        md: 'h-10 px-4 text-sm',
        lg: 'h-12 px-6 text-base',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  }
);

export const Button = forwardRef<HTMLButtonElement, ButtonProps & VariantProps<typeof buttonVariants>>(
  ({ className, variant, size, loading, leftIcon, rightIcon, children, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={buttonVariants({ variant, size, className })}
        disabled={loading || props.disabled}
        {...props}
      >
        {loading ? (
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        ) : leftIcon ? (
          <span className="mr-2">{leftIcon}</span>
        ) : null}
        {children}
        {rightIcon && !loading && <span className="ml-2">{rightIcon}</span>}
      </button>
    );
  }
);

Button.displayName = 'Button';
```

#### 5.1.2 Card

```typescript
// components/ui/card/card.types.ts
interface CardProps {
  variant?: 'default' | 'elevated' | 'outlined';
  padding?: 'none' | 'sm' | 'md' | 'lg';
  hoverable?: boolean;
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

// components/ui/card/card.tsx
import { forwardRef } from 'react';
import { motion } from 'framer-motion';

const cardVariants = {
  default: 'bg-white border border-gray-100',
  elevated: 'bg-white shadow-lg shadow-gray-200/50',
  outlined: 'bg-transparent border-2 border-gray-200',
};

export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ variant = 'default', padding = 'md', hoverable = false, children, className, onClick, ...props }, ref) => {
    const paddingClasses = {
      none: '',
      sm: 'p-3',
      md: 'p-5',
      lg: 'p-8',
    };

    const Component = hoverable ? motion.div : 'div';
    const hoverProps = hoverable ? {
      whileHover: { y: -4, boxShadow: '0 20px 40px rgba(0,0,0,0.1)' },
      transition: { duration: 0.2 },
    } : {};

    return (
      <Component
        ref={ref}
        className={`${cardVariants[variant]} ${paddingClasses[padding]} rounded-2xl ${className}`}
        onClick={onClick}
        {...hoverProps}
        {...props}
      >
        {children}
      </Component>
    );
  }
);

Card.displayName = 'Card';
```

#### 5.1.3 Progress Bar

```typescript
// components/ui/progress/progress.types.ts
interface ProgressProps {
  value: number;           // 0-100
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'success' | 'warning' | 'danger';
  showLabel?: boolean;
  animated?: boolean;
  striped?: boolean;
}

// components/ui/progress/progress.tsx
import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export const ProgressBar: React.FC<ProgressProps> = ({
  value,
  max = 100,
  size = 'md',
  variant = 'default',
  showLabel = false,
  animated = true,
  striped = false,
}) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    setDisplayValue(value);
  }, [value]);

  const percentage = (displayValue / max) * 100;

  const sizeClasses = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4',
  };

  const variantClasses = {
    default: 'from-indigo-500 to-purple-500',
    success: 'from-emerald-500 to-teal-500',
    warning: 'from-amber-500 to-orange-500',
    danger: 'from-red-500 to-rose-500',
  };

  return (
    <div className="w-full">
      <div
        className={`w-full bg-gray-100 rounded-full overflow-hidden ${sizeClasses[size]}`}
      >
        <motion.div
          className={`h-full bg-gradient-to-r ${variantClasses[variant]} rounded-full ${
            striped ? 'bg-[length:20px_20px]' : ''
          }`}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{
            duration: animated ? 0.5 : 0,
            ease: 'easeOut',
          }}
          style={{
            backgroundImage: striped
              ? 'linear-gradient(45deg,rgba(255,255,255,.15) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.15) 50%,rgba(255,255,255,.15) 75%,transparent 75%,transparent)'
              : undefined,
          }}
        />
      </div>
      {showLabel && (
        <div className="flex justify-between mt-1 text-sm text-gray-500">
          <span>Progress</span>
          <span>{Math.round(percentage)}%</span>
        </div>
      )}
    </div>
  );
};
```

### 5.2 Analysis Components (Organisms)

#### 5.2.1 File Upload Zone

```typescript
// components/analysis/upload/file-dropzone.types.ts
interface FileDropzoneProps {
  accept?: Record<string, string[]>;
  maxSize?: number;
  multiple?: boolean;
  onFilesSelected: (files: File[]) => void;
  onError?: (error: string) => void;
}

// components/analysis/upload/file-dropzone.tsx
import { useCallback, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, File, Video, Music, Image, FileText } from 'lucide-react';

const FILE_TYPE_ICONS = {
  video: Video,
  audio: Music,
  image: Image,
  text: FileText,
  default: File,
};

const ACCEPTED_TYPES = {
  video: ['video/mp4', 'video/avi', 'video/mov', 'video/webm'],
  audio: ['audio/wav', 'audio/mp3', 'audio/flac', 'audio/m4a'],
  image: ['image/jpeg', 'image/png', 'image/webp'],
  text: ['text/plain', 'application/pdf', 'text/markdown'],
};

export const FileDropzone: React.FC<FileDropzoneProps> = ({
  accept = ACCEPTED_TYPES,
  maxSize = 500 * 1024 * 1024, // 500MB
  multiple = false,
  onFilesSelected,
  onError,
}) => {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [dragCounter, setDragCounter] = useState(0);

  const getFileType = (file: File): string => {
    if (file.type.startsWith('video/')) return 'video';
    if (file.type.startsWith('audio/')) return 'audio';
    if (file.type.startsWith('image/')) return 'image';
    if (file.type.startsWith('text/') || file.type === 'application/pdf') return 'text';
    return 'default';
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  const validateFiles = useCallback(
    (fileList: FileList | File[]): File[] => {
      const validFiles: File[] = [];
      const errors: string[] = [];

      Array.from(fileList).forEach((file) => {
        if (file.size > maxSize) {
          errors.push(`${file.name} exceeds ${formatFileSize(maxSize)}`);
          return;
        }
        const isValidType = Object.values(accept).some((types) =>
          types.includes(file.type)
        );
        if (!isValidType) {
          errors.push(`${file.name} has unsupported type`);
          return;
        }
        validFiles.push(file);
      });

      if (errors.length > 0 && onError) {
        onError(errors.join(', '));
      }

      return validFiles;
    },
    [accept, maxSize, onError]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      setDragCounter(0);

      const droppedFiles = validateFiles(e.dataTransfer.files);
      if (droppedFiles.length > 0) {
        const newFiles = multiple ? [...files, ...droppedFiles] : droppedFiles.slice(0, 1);
        setFiles(newFiles);
        onFilesSelected(newFiles);
      }
    },
    [files, multiple, validateFiles, onFilesSelected]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = validateFiles(e.target.files || []);
      if (selectedFiles.length > 0) {
        const newFiles = multiple ? [...files, ...selectedFiles] : selectedFiles.slice(0, 1);
        setFiles(newFiles);
        onFilesSelected(newFiles);
      }
    },
    [files, multiple, validateFiles, onFilesSelected]
  );

  const removeFile = useCallback(
    (index: number) => {
      const newFiles = files.filter((_, i) => i !== index);
      setFiles(newFiles);
      onFilesSelected(newFiles);
    },
    [files, onFilesSelected]
  );

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragCounter((prev) => prev + 1);
    if (e.dataTransfer.dragEffect !== 'copy') return;
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragCounter((prev) => prev - 1);
    if (dragCounter <= 1) {
      setIsDragging(false);
    }
  }, [dragCounter]);

  return (
    <div className="space-y-4">
      <AnimatePresence mode="wait">
        <motion.div
          key={isDragging ? 'dragging' : 'idle'}
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.98 }}
          className={`relative border-2 border-dashed rounded-2xl p-8 transition-all duration-200 ${
            isDragging
              ? 'border-indigo-500 bg-indigo-50/50'
              : 'border-gray-200 hover:border-gray-300'
          }`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept={Object.values(accept).flat().join(',')}
            multiple={multiple}
            onChange={handleFileInput}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            aria-label="File upload"
          />

          <div className="flex flex-col items-center text-center">
            <motion.div
              animate={{
                y: isDragging ? -8 : 0,
                scale: isDragging ? 1.1 : 1,
              }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            >
              <div className="w-16 h-16 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-2xl flex items-center justify-center mb-4">
                <Upload className="w-8 h-8 text-indigo-600" />
              </div>
            </motion.div>

            <h3 className="text-lg font-semibold text-gray-900 mb-1">
              {isDragging ? 'Drop your files here' : 'Upload files for analysis'}
            </h3>
            <p className="text-sm text-gray-500 mb-4">
              Drag and drop or click to browse
            </p>

            <div className="flex flex-wrap justify-center gap-2 text-xs text-gray-400">
              <span className="px-2 py-1 bg-gray-100 rounded-lg">Video (MP4, AVI, MOV)</span>
              <span className="px-2 py-1 bg-gray-100 rounded-lg">Audio (WAV, MP3, FLAC)</span>
              <span className="px-2 py-1 bg-gray-100 rounded-lg">Image (PNG, JPG, WebP)</span>
              <span className="px-2 py-1 bg-gray-100 rounded-lg">Text (TXT, PDF, MD)</span>
            </div>

            <p className="text-xs text-gray-400 mt-4">
              Maximum file size: {formatFileSize(maxSize)}
            </p>
          </div>
        </motion.div>
      </AnimatePresence>

      <AnimatePresence mode="popLayout">
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-2"
          >
            {files.map((file, index) => {
              const fileType = getFileType(file);
              const Icon = FILE_TYPE_ICONS[fileType as keyof typeof FILE_TYPE_ICONS] || File;

              return (
                <motion.div
                  key={`${file.name}-${index}`}
                  layout
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="flex items-center gap-3 p-3 bg-white border border-gray-100 rounded-xl hover:border-gray-200 transition-colors"
                >
                  <div className="w-10 h-10 bg-gray-50 rounded-lg flex items-center justify-center">
                    <Icon className="w-5 h-5 text-gray-600" />
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {file.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {formatFileSize(file.size)}
                    </p>
                  </div>

                  <button
                    onClick={() => removeFile(index)}
                    className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"
                    aria-label={`Remove ${file.name}`}
                  >
                    <X className="w-4 h-4 text-gray-400" />
                  </button>
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
```

#### 5.2.2 Real-time Progress Tracker

```typescript
// components/analysis/progress/progress-tracker.types.ts
interface ProgressStage {
  id: string;
  label: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  progress: number;
  message?: string;
}

interface ProgressTrackerProps {
  stages: ProgressStage[];
  currentStage?: string;
  estimatedTimeRemaining?: number;
  onCancel?: () => void;
}

// components/analysis/progress/progress-tracker.tsx
import { motion, AnimatePresence } from 'framer-motion';
import { Check, Clock, AlertCircle, Activity, Layers, Video, Music, FileText, BarChart3 } from 'lucide-react';

const STAGE_ICONS: Record<string, React.ElementType> = {
  pending: Clock,
  extracting_frames: Layers,
  running_video_detection: Video,
  running_audio_detection: Music,
  running_text_detection: FileText,
  fusing_scores: BarChart3,
  generating_report: Activity,
  completed: Check,
  error: AlertCircle,
};

export const ProgressTracker: React.FC<ProgressTrackerProps> = ({
  stages,
  currentStage,
  estimatedTimeRemaining,
  onCancel,
}) => {
  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  return (
    <div className="bg-white rounded-2xl border border-gray-100 p-6 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Analysis Progress</h3>
        {estimatedTimeRemaining !== undefined && estimatedTimeRemaining > 0 && (
          <div className="flex items-center gap-1.5 text-sm text-gray-500">
            <Clock className="w-4 h-4" />
            <span>~{formatTime(estimatedTimeRemaining)} remaining</span>
          </div>
        )}
      </div>

      <div className="relative">
        {stages.map((stage, index) => {
          const Icon = STAGE_ICONS[stage.id] || Clock;
          const isCurrent = stage.id === currentStage;
          const isPending = stage.status === 'pending';
          const isCompleted = stage.status === 'completed';
          const isError = stage.status === 'error';

          return (
            <div key={stage.id} className="relative pb-8 last:pb-0">
              {index < stages.length - 1 && (
                <div className="absolute left-[19px] top-10 -bottom-2 w-0.5 bg-gray-100">
                  <motion.div
                    className="w-full bg-indigo-500"
                    initial={{ height: 0 }}
                    animate={{ height: isCompleted || isCurrent ? '100%' : '0%' }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              )}

              <div className="flex items-start gap-4">
                <motion.div
                  className={`relative z-10 w-10 h-10 rounded-xl flex items-center justify-center ${
                    isCompleted
                      ? 'bg-emerald-500'
                      : isError
                      ? 'bg-red-500'
                      : isCurrent
                      ? 'bg-indigo-500 shadow-lg shadow-indigo-200'
                      : 'bg-gray-100'
                  }`}
                  initial={false}
                  animate={{
                    scale: isCurrent ? [1, 1.1, 1] : 1,
                  }}
                  transition={{
                    duration: 1,
                    repeat: isCurrent ? Infinity : 0,
                  }}
                >
                  {isCompleted ? (
                    <Check className="w-5 h-5 text-white" />
                  ) : isError ? (
                    <AlertCircle className="w-5 h-5 text-white" />
                  ) : isCurrent ? (
                    <motion.div
                      className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    />
                  ) : (
                    <Icon className={`w-5 h-5 ${isPending ? 'text-gray-400' : 'text-gray-600'}`} />
                  )}
                </motion.div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <h4
                      className={`text-sm font-medium ${
                        isCompleted
                          ? 'text-emerald-700'
                          : isError
                          ? 'text-red-700'
                          : isCurrent
                          ? 'text-indigo-700'
                          : 'text-gray-500'
                      }`}
                    >
                      {stage.label}
                    </h4>
                    <span
                      className={`text-xs font-medium ${
                        isCompleted
                          ? 'text-emerald-600'
                          : isError
                          ? 'text-red-600'
                          : isCurrent
                          ? 'text-indigo-600'
                          : 'text-gray-400'
                      }`}
                    >
                      {isCompleted ? '100%' : isCurrent ? `${stage.progress}%` : '-'}
                    </span>
                  </div>

                  {(isCurrent || stage.message) && (
                    <motion.p
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="text-xs text-gray-500 mb-2"
                    >
                      {stage.message || 'Processing...'}
                    </motion.p>
                  )}

                  {isCurrent && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="w-full bg-gray-100 rounded-full h-1.5 overflow-hidden"
                    >
                      <motion.div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${stage.progress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </motion.div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {onCancel && (
        <div className="mt-6 pt-4 border-t border-gray-100">
          <button
            onClick={onCancel}
            className="w-full py-2.5 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-xl transition-colors"
          >
            Cancel Analysis
          </button>
        </div>
      )}
    </div>
  );
};
```

#### 5.2.3 Trust Score Display

```typescript
// components/analysis/results/trust-score.types.ts
interface TrustScoreProps {
  overallScore: number;          // 0-100
  confidence: number;             // 0-1
  confidenceLevel: 'high' | 'medium' | 'low';
  breakdown: {
    video: { weight: number; contribution: number; percentage: number };
    audio: { weight: number; contribution: number; percentage: number };
    text: { weight: number; contribution: number; percentage: number };
  };
  verdict: 'AUTHENTIC' | 'MANIPULATED' | 'UNCERTAIN';
}

// components/analysis/results/trust-score.tsx
import { motion } from 'framer-motion';
import { Shield, AlertTriangle, QuestionCircle, CheckCircle, XCircle } from 'lucide-react';

const VERDICT_CONFIG = {
  AUTHENTIC: {
    icon: CheckCircle,
    color: 'text-emerald-500',
    bgColor: 'bg-emerald-50',
    borderColor: 'border-emerald-200',
    gradient: 'from-emerald-500 to-teal-500',
    label: 'Authentic',
  },
  MANIPULATED: {
    icon: XCircle,
    color: 'text-red-500',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
    gradient: 'from-red-500 to-rose-500',
    label: 'Manipulated',
  },
  UNCERTAIN: {
    icon: AlertTriangle,
    color: 'text-amber-500',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-200',
    gradient: 'from-amber-500 to-orange-500',
    label: 'Uncertain',
  },
};

const CONFIDENCE_CONFIG = {
  high: { label: 'High Confidence', color: 'text-emerald-600', bg: 'bg-emerald-100' },
  medium: { label: 'Medium Confidence', color: 'text-amber-600', bg: 'bg-amber-100' },
  low: { label: 'Low Confidence', color: 'text-red-600', bg: 'bg-red-100' },
};

export const TrustScoreDisplay: React.FC<TrustScoreProps> = ({
  overallScore,
  confidence,
  confidenceLevel,
  breakdown,
  verdict,
}) => {
  const verdictConfig = VERDICT_CONFIG[verdict];
  const confidenceConfig = CONFIDENCE_CONFIG[confidenceLevel];

  const getScoreColor = (score: number): string => {
    if (score >= 80) return 'text-emerald-500';
    if (score >= 60) return 'text-amber-500';
    if (score >= 40) return 'text-orange-500';
    return 'text-red-500';
  };

  const getScoreGradient = (score: number): string => {
    if (score >= 80) return 'from-emerald-500 to-teal-500';
    if (score >= 60) return 'from-amber-500 to-orange-500';
    return 'from-red-500 to-rose-500';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-2xl border border-gray-100 p-6 shadow-lg"
    >
      <div className="flex items-start gap-6">
        {/* Score Circle */}
        <div className="relative">
          <svg className="w-32 h-32 transform -rotate-90">
            <circle
              cx="64"
              cy="64"
              r="56"
              fill="none"
              stroke="#f3f4f6"
              strokeWidth="12"
            />
            <motion.circle
              cx="64"
              cy="64"
              r="56"
              fill="none"
              stroke="url(#scoreGradient)"
              strokeWidth="12"
              strokeLinecap="round"
              initial={{ strokeDasharray: '351.86' }}
              animate={{ strokeDasharray: `${351.86 * (1 - overallScore / 100)} 351.86` }}
              transition={{ duration: 1, ease: 'easeOut' }}
            />
            <defs>
              <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor={verdictConfig.gradient.split(' ')[0].replace('from-', '')} />
                <stop offset="100%" stopColor={verdictConfig.gradient.split(' ')[1].replace('to-', '')} />
              </linearGradient>
            </defs>
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.span
              key={overallScore}
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className={`text-4xl font-bold ${getScoreColor(overallScore)}`}
            >
              {overallScore}
            </motion.span>
            <span className="text-xs text-gray-400">Trust Score</span>
          </div>
        </div>

        <div className="flex-1">
          {/* Verdict Badge */}
          <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${verdictConfig.bgColor} mb-4`}>
            {(() => {
              const Icon = verdictConfig.icon;
              return <Icon className={`w-5 h-5 ${verdictConfig.color}`} />;
            })()}
            <span className={`text-sm font-semibold ${verdictConfig.color}`}>
              {verdictConfig.label}
            </span>
          </div>

          <h3 className="text-xl font-semibold text-gray-900 mb-4">
            Analysis Result
          </h3>

          {/* Confidence */}
          <div className="flex items-center gap-2 mb-4">
            <span className="text-sm text-gray-500">Confidence:</span>
            <span className={`text-sm font-medium px-2 py-0.5 rounded ${confidenceConfig.bg} ${confidenceConfig.color}`}>
              {confidenceConfig.label}
            </span>
            <span className="text-sm text-gray-600">({Math.round(confidence * 100)}%)</span>
          </div>

          {/* Breakdown */}
          <div className="space-y-3">
            {Object.entries(breakdown).map(([modality, data]) => (
              <div key={modality} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="capitalize text-gray-600">{modality}</span>
                  <span className="text-gray-900 font-medium">{data.percentage}%</span>
                </div>
                <div className="w-full bg-gray-100 rounded-full h-2">
                  <motion.div
                    className={`h-full rounded-full bg-gradient-to-r ${
                      modality === 'video'
                        ? 'from-blue-500 to-indigo-500'
                        : modality === 'audio'
                        ? 'from-purple-500 to-pink-500'
                        : 'from-amber-500 to-orange-500'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${data.percentage}%` }}
                    transition={{ duration: 0.8, delay: 0.2 }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
};
```

---

## 6. State Management

### 6.1 Global Stores

```typescript
// stores/analysis.store.ts
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { devtools } from 'zustand/middleware';

interface Analysis {
  id: string;
  fileName: string;
  fileType: string;
  fileSize: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  currentStage?: string;
  verdict?: string;
  trustScore?: number;
  createdAt: string;
  result?: ForensicReport;
}

interface AnalysisState {
  analyses: Analysis[];
  currentAnalysis: Analysis | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  setCurrentAnalysis: (analysis: Analysis | null) => void;
  addAnalysis: (analysis: Analysis) => void;
  updateAnalysis: (id: string, updates: Partial<Analysis>) => void;
  removeAnalysis: (id: string) => void;
  setError: (error: string | null) => void;
  clearCurrentAnalysis: () => void;
}

export const useAnalysisStore = create<AnalysisState>()(
  devtools(
    subscribeWithSelector((set) => ({
      analyses: [],
      currentAnalysis: null,
      isLoading: false,
      error: null,

      setCurrentAnalysis: (analysis) => set({ currentAnalysis: analysis }),

      addAnalysis: (analysis) =>
        set((state) => ({
          analyses: [analysis, ...state.analyses],
        })),

      updateAnalysis: (id, updates) =>
        set((state) => ({
          analyses: state.analyses.map((a) =>
            a.id === id ? { ...a, ...updates } : a
          ),
          currentAnalysis:
            state.currentAnalysis?.id === id
              ? { ...state.currentAnalysis, ...updates }
              : state.currentAnalysis,
        })),

      removeAnalysis: (id) =>
        set((state) => ({
          analyses: state.analyses.filter((a) => a.id !== id),
          currentAnalysis:
            state.currentAnalysis?.id === id ? null : state.currentAnalysis,
        })),

      setError: (error) => set({ error }),

      clearCurrentAnalysis: () => set({ currentAnalysis: null }),
    })),
    { name: 'analysis-store' }
  )
);

// stores/ui.store.ts
interface UIState {
  sidebarOpen: boolean;
  theme: 'light' | 'dark' | 'system';
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'warning' | 'info';
    message: string;
    duration?: number;
  }>;

  toggleSidebar: () => void;
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  addNotification: (notification: Omit<UIState['notifications'][0], 'id'>) => void;
  removeNotification: (id: string) => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    subscribeWithSelector((set) => ({
      sidebarOpen: true,
      theme: 'light',
      notifications: [],

      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

      setTheme: (theme) => set({ theme }),

      addNotification: (notification) =>
        set((state) => ({
          notifications: [
            ...state.notifications,
            { ...notification, id: crypto.randomUUID() },
          ],
        })),

      removeNotification: (id) =>
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
        })),
    })),
    { name: 'ui-store' }
  )
);
```

### 6.2 API Query Keys

```typescript
// lib/api/query-keys.ts
export const queryKeys = {
  analyses: {
    all: ['analyses'] as const,
    lists: () => [...queryKeys.analyses.all, 'list'] as const,
    list: (filters: { status?: string; fileType?: string; limit?: number; offset?: number }) =>
      [...queryKeys.analyses.lists(), filters] as const,
    details: () => [...queryKeys.analyses.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.analyses.details(), id] as const,
    results: () => [...queryKeys.analyses.all, 'result'] as const,
    result: (id: string) => [...queryKeys.analyses.results(), id] as const,
  },
  health: {
    all: ['health'] as const,
    basic: () => [...queryKeys.health.all, 'basic'] as const,
    detailed: () => [...queryKeys.health.all, 'detailed'] as const,
    ready: () => [...queryKeys.health.all, 'ready'] as const,
    config: () => [...queryKeys.health.all, 'config'] as const,
  },
};
```

---

## 7. UI/UX Design System

### 7.1 Color Palette

```typescript
// styles/colors.ts
export const COLORS = {
  // Primary gradients
  primary: {
    light: 'from-indigo-400 to-purple-400',
    DEFAULT: 'from-indigo-600 to-purple-600',
    dark: 'from-indigo-700 to-purple-700',
  },

  // Semantic colors
  success: {
    50: '#f0fdf4',
    100: '#dcfce7',
    500: '#22c55e',
    600: '#16a34a',
    700: '#15803d',
  },
  warning: {
    50: '#fffbeb',
    100: '#fef3c7',
    500: '#f59e0b',
    600: '#d97706',
    700: '#b45309',
  },
  danger: {
    50: '#fef2f2',
    100: '#fee2e2',
    500: '#ef4444',
    600: '#dc2626',
    700: '#b91c1c',
  },

  // Neutrals
  gray: {
    50: '#f9fafb',
    100: '#f3f4f6',
    200: '#e5e7eb',
    300: '#d1d5db',
    400: '#9ca3af',
    500: '#6b7280',
    600: '#4b5563',
    700: '#374151',
    800: '#1f2937',
    900: '#111827',
  },
};

export type ColorScale = keyof typeof COLORS.success;
```

### 7.2 Typography

```typescript
// styles/typography.ts
export const TYPOGRAPHY = {
  fontFamily: {
    sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
    mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
  },

  fontSize: {
    xs: ['0.75rem', { lineHeight: '1rem' }],
    sm: ['0.875rem', { lineHeight: '1.25rem' }],
    base: ['1rem', { lineHeight: '1.5rem' }],
    lg: ['1.125rem', { lineHeight: '1.75rem' }],
    xl: ['1.25rem', { lineHeight: '1.75rem' }],
    '2xl': ['1.5rem', { lineHeight: '2rem' }],
    '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
    '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
    '5xl': ['3rem', { lineHeight: '1' }],
  },

  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },

  letterSpacing: {
    tight: '-0.025em',
    normal: '0',
    wide: '0.025em',
  },
};
```

### 7.3 Spacing & Layout

```typescript
// styles/spacing.ts
export const SPACING = {
  // Tailwind-like spacing scale
  0: '0px',
  1: '0.25rem',
  2: '0.5rem',
  3: '0.75rem',
  4: '1rem',
  5: '1.25rem',
  6: '1.5rem',
  8: '2rem',
  10: '2.5rem',
  12: '3rem',
  16: '4rem',
  20: '5rem',
  24: '6rem',
  32: '8rem',

  // Container widths
  'screen-sm': '640px',
  'screen-md': '768px',
  'screen-lg': '1024px',
  'screen-xl': '1280px',
  'screen-2xl': '1536px',
};
```

### 7.4 Animation Tokens

```typescript
// styles/animations.ts
export const ANIMATIONS = {
  // Transitions
  transition: {
    fast: '150ms ease',
    normal: '200ms ease',
    slow: '300ms ease',
  },

  // Keyframes
  keyframes: {
    fadeIn: {
      '0%': { opacity: '0' },
      '100%': { opacity: '1' },
    },
    slideUp: {
      '0%': { opacity: '0', transform: 'translateY(10px)' },
      '100%': { opacity: '1', transform: 'translateY(0)' },
    },
    pulse: {
      '0%, 100%': { opacity: '1' },
      '50%': { opacity: '0.5' },
    },
    spin: {
      '0%': { transform: 'rotate(0deg)' },
      '100%': { transform: 'rotate(360deg)' },
    },
  },

  // Spring configurations
  spring: {
    gentle: { type: 'spring', stiffness: 200, damping: 20 },
    bouncy: { type: 'spring', stiffness: 300, damping: 25 },
    snappy: { type: 'spring', stiffness: 400, damping: 30 },
  },
};
```

---

## 8. Feature Implementation

### 8.1 Landing Page

```typescript
// app/page.tsx
import { HeroSection } from '@/components/landing/hero-section';
import { FeaturesSection } from '@/components/landing/features-section';
import { CTASection } from '@/components/landing/cta-section';

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-white to-gray-50">
      <HeroSection />
      <FeaturesSection />
      <CTASection />
    </main>
  );
}
```

### 8.2 Dashboard Layout

```typescript
// app/(dashboard)/layout.tsx
import { Sidebar } from '@/components/shared/sidebar';
import { Header } from '@/components/shared/header';
import { useUIStore } from '@/stores/ui.store';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const sidebarOpen = useUIStore((state) => state.sidebarOpen);

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar />
      <div
        className={`transition-all duration-300 ${
          sidebarOpen ? 'lg:ml-64' : 'lg:ml-20'
        }`}
      >
        <Header />
        <main className="p-6">{children}</main>
      </div>
    </div>
  );
}
```

### 8.3 Analysis Flow

```typescript
// app/(dashboard)/analyze/page.tsx
'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileDropzone } from '@/components/analysis/upload/file-dropzone';
import { ProgressTracker } from '@/components/analysis/progress/progress-tracker';
import { TrustScoreDisplay } from '@/components/analysis/results/trust-score';
import { AnalysisService } from '@/services/analysis.service';
import { useAnalysisStore } from '@/stores/analysis.store';
import { useWebSocket } from '@/hooks/useWebSocket';
import { ArrowRight, Sparkles } from 'lucide-react';

type AnalysisStep = 'upload' | 'processing' | 'results';

export default function AnalyzePage() {
  const [step, setStep] = useState<AnalysisStep>('upload');
  const [requestId, setRequestId] = useState<string | null>(null);
  const [progressStages, setProgressStages] = useState<ProgressStage[]>([]);
  const [result, setResult] = useState<ForensicReport | null>(null);

  const addAnalysis = useAnalysisStore((state) => state.addAnalysis);
  const updateAnalysis = useAnalysisStore((state) => state.updateAnalysis);

  const handleFilesSelected = async (files: File[]) => {
    try {
      const response = await AnalysisService.submitAnalysis(files[0]);
      setRequestId(response.request_id);
      setStep('processing');

      // Initialize progress stages
      setProgressStages([
        { id: 'extracting_frames', label: 'Extracting Frames', status: 'pending', progress: 0 },
        { id: 'running_video_detection', label: 'Running Video Detection', status: 'pending', progress: 0 },
        { id: 'running_audio_detection', label: 'Running Audio Detection', status: 'pending', progress: 0 },
        { id: 'fusing_scores', label: 'Fusing Scores', status: 'pending', progress: 0 },
        { id: 'generating_report', label: 'Generating Report', status: 'pending', progress: 0 },
      ]);

      addAnalysis({
        id: response.request_id,
        fileName: files[0].name,
        fileType: files[0].type.split('/')[0],
        fileSize: files[0].size,
        status: 'pending',
        progress: 0,
        createdAt: new Date().toISOString(),
      });
    } catch (error) {
      console.error('Analysis submission failed:', error);
    }
  };

  // WebSocket progress updates
  useWebSocket(
    requestId ? `/api/v1/analyze/${requestId}/stream` : null,
    (update) => {
      setProgressStages((prev) =>
        prev.map((stage) =>
          stage.id === update.stage
            ? { ...stage, status: update.progress === 100 ? 'completed' : 'in_progress', progress: update.progress, message: update.message }
            : stage
        )
      );

      updateAnalysis(requestId!, {
        progress: update.progress,
        currentStage: update.stage,
        status: update.progress === 100 ? 'completed' : 'processing',
      });

      if (update.progress === 100 && update.stage === 'generating_report') {
        // Fetch results
        AnalysisService.getResult(requestId!).then(setResult).then(() => setStep('results'));
      }
    }
  );

  return (
    <div className="max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Analyze Content</h1>
        <p className="text-gray-500">
          Upload video, audio, or text content for deepfake detection analysis
        </p>
      </motion.div>

      <AnimatePresence mode="wait">
        {step === 'upload' && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
          >
            <FileDropzone onFilesSelected={handleFilesSelected} />
          </motion.div>
        )}

        {step === 'processing' && (
          <motion.div
            key="processing"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
          >
            <ProgressTracker
              stages={progressStages}
              currentStage={progressStages.find((s) => s.status === 'in_progress')?.id}
              onCancel={() => setStep('upload')}
            />
          </motion.div>
        )}

        {step === 'results' && result && (
          <motion.div
            key="results"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="space-y-6"
          >
            <TrustScoreDisplay
              overallScore={result.trust_score.overall_score}
              confidence={result.trust_score.confidence}
              confidenceLevel={result.trust_score.confidence_level}
              breakdown={result.trust_score.breakdown}
              verdict={result.executive_summary.verdict as any}
            />

            <div className="flex justify-end gap-4">
              <button
                onClick={() => {
                  setStep('upload');
                  setRequestId(null);
                  setResult(null);
                }}
                className="px-6 py-3 text-gray-600 hover:text-gray-900 transition-colors"
              >
                Analyze Another
              </button>
              <button className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:from-indigo-700 hover:to-purple-700 transition-all flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                Download Full Report
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
```

---

## 9. Performance Optimization

### 9.1 Code Splitting

```typescript
// lib/utils/lazy.ts
import dynamic from 'next/dynamic';

export const lazy = {
  HeatmapViewer: dynamic(
    () => import('@/components/analysis/results/heatmap-viewer'),
    { loading: () => <div className="h-64 bg-gray-100 animate-pulse rounded-xl" /> }
  ),
  SpectrogramViewer: dynamic(
    () => import('@/components/analysis/results/spectrogram-viewer'),
    { loading: () => <div className="h-64 bg-gray-100 animate-pulse rounded-xl" /> }
  ),
  ChartsPanel: dynamic(
    () => import('@/components/analysis/results/charts-panel'),
    { loading: () => <div className="h-80 bg-gray-100 animate-pulse rounded-xl" /> }
  ),
};
```

### 9.2 Image Optimization

```typescript
// components/ui/image/optimized-image.types.ts
interface OptimizedImageProps {
  src: string;
  alt: string;
  width: number;
  height: number;
  priority?: boolean;
  placeholder?: 'blur' | 'empty';
  blurDataURL?: string;
  sizes?: string;
}

// Using Next.js Image component with optimization
import Image from 'next/image';

export const OptimizedImage: React.FC<OptimizedImageProps> = ({
  src,
  alt,
  width,
  height,
  priority = false,
  placeholder = 'empty',
  blurDataURL,
  sizes = '(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw',
}) => {
  return (
    <Image
      src={src}
      alt={alt}
      width={width}
      height={height}
      priority={priority}
      placeholder={placeholder}
      blurDataURL={blurDataURL}
      sizes={sizes}
      className="object-cover rounded-xl"
      quality={85}
    />
  );
};
```

### 9.3 Bundle Optimization

```typescript
// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable React strict mode
  reactStrictMode: true,

  // Image optimization
  images: {
    domains: ['localhost'],
    formats: ['image/avif', 'image/webp'],
  },

  // Bundle analyzer
  ...(process.env.ANALYZE === 'true' && {
    // eslint: { ignoreDuringBuilds: true },
  }),

  // Experimental features for performance
  experimental: {
    optimizePackageImports: ['lucide-react', 'framer-motion', 'recharts'],
  },

  // Transpile packages for compatibility
  transpilePackages: ['lucide-react'],

  // Webpack configuration
  webpack: (config, { isServer }) => {
    // Optimize chunk splitting
    if (!isServer) {
      config.optimization.splitChunks = {
        chunks: 'all',
        minSize: 20000,
        maxSize: 244000,
        minChunks: 1,
        maxAsyncRequests: 30,
        maxInitialRequests: 30,
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            priority: 10,
          },
          react: {
            test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
            name: 'react',
            priority: 20,
          },
          ui: {
            test: /[\\/]node_modules[\\/](lucide-react|framer-motion)[\\/]/,
            name: 'ui',
            priority: 20,
          },
          charts: {
            test: /[\\/]node_modules[\\/](recharts)[\\/]/,
            name: 'charts',
            priority: 20,
          },
        },
      };
    }

    return config;
  },
};

module.exports = nextConfig;
```

---

## 10. Security Implementation

### 10.1 XSS Prevention

```typescript
// lib/security/sanitization.ts
import DOMPurify from 'isomorphic-dompurify';

export const sanitizeHtml = (html: string): string => {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
    ALLOWED_ATTR: ['href', 'target', 'rel'],
  });
};

export const sanitizeFileName = (fileName: string): string => {
  return fileName.replace(/[^a-zA-Z0-9._-]/g, '_');
};
```

### 10.2 API Security

```typescript
// lib/api/security.ts
import axios, { AxiosInstance } from 'axios';
import { API_CONFIG } from '@/config/api';

const createSecureClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_CONFIG.baseUrl,
    timeout: API_CONFIG.timeout.request,
    headers: {
      'Content-Type': 'application/json',
    },
    withCredentials: true,
  });

  // Request interceptor for security headers
  client.interceptors.request.use(
    (config) => {
      // Add CSRF token if available
      const csrfToken = document.cookie
        .split('; ')
        .find((row) => row.startsWith('csrf_token='))
        ?.split('=')[1];

      if (csrfToken) {
        config.headers['X-CSRF-Token'] = csrfToken;
      }

      return config;
    },
    (error) => Promise.reject(error)
  );

  // Response interceptor for error handling
  client.interceptors.response.use(
    (response) => response,
    (error) => {
      // Handle 401/403 responses
      if (error.response?.status === 401) {
        // Redirect to login
        window.location.href = '/login';
      }
      return Promise.reject(error);
    }
  );

  return client;
};

export const apiClient = createSecureClient();
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

```typescript
// tests/components/button.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from '@/components/ui/button/button';
import { describe, it, expect, vi } from 'vitest';

describe('Button', () => {
  it('renders children correctly', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button')).toHaveTextContent('Click me');
  });

  it('calls onClick when clicked', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click me</Button>);
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('shows loading state', () => {
    render(<Button loading>Click me</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
    expect(screen.getByRole('button')).toContainHTML('animate-spin');
  });

  it('applies variant classes', () => {
    render(<Button variant="danger">Delete</Button>);
    expect(screen.getByRole('button')).toHaveClass('bg-red-600');
  });
});
```

### 11.2 Integration Tests

```typescript
// tests/features/analysis.test.ts
import { test, expect } from '@playwright/test';

test.describe('Analysis Flow', () => {
  test('completes analysis workflow', async ({ page }) => {
    await page.goto('/analyze');

    // Upload file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles('tests/fixtures/sample-video.mp4');

    // Verify upload
    await expect(page.locator('text=sample-video.mp4')).toBeVisible();

    // Submit analysis
    await page.click('button:has-text("Analyze")');

    // Wait for progress
    await expect(page.locator('text=Extracting Frames')).toBeVisible();

    // Wait for completion
    await page.waitForSelector('text=Analysis Result', { timeout: 30000 });

    // Verify result
    await expect(page.locator('text=Trust Score')).toBeVisible();
  });
});
```

---

## 12. Deployment Configuration

### 12.1 Environment Variables

```bash
# .env.example
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NODE_ENV=production

# Optional: Analytics
NEXT_PUBLIC_GA_ID=
NEXT_PUBLIC_HOTJAR_ID=
```

### 12.2 Docker Configuration

```dockerfile
# Dockerfile
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy source
COPY . .

# Build application
RUN npm run build

# Production image
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy necessary files
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
```

### 12.3 CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Run type check
        run: npm run typecheck

      - name: Run tests
        run: npm run test:ci

      - name: Build application
        run: npm run build

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          vercel-args: '--prod'
```

---

## Appendix A: Component Checklist

### A.1 Base Components (Atoms)
- [ ] Button (variants: primary, secondary, outline, ghost, danger)
- [ ] Card (variants: default, elevated, outlined)
- [ ] Input (text, number, email, password)
- [ ] Select (single, multi)
- [ ] Modal (confirm, form, info)
- [ ] Progress (bar, spinner)
- [ ] Toast (success, error, warning, info)
- [ ] Badge (status, count)
- [ ] Avatar (image, initials, icon)
- [ ] Tooltip
- [ ] Dropdown
- [ ] Tabs
- [ ] Table (sortable, paginated)
- [ ] Pagination
- [ ] Empty State
- [ ] Loading State
- [ ] Error State

### A.2 Feature Components (Organisms)
- [ ] FileDropzone (drag & drop, preview, validation)
- [ ] ProgressTracker (stages, streaming updates)
- [ ] TrustScoreDisplay (score circle, breakdown chart)
- [ ] HeatmapViewer (zoom, pan, overlay)
- [ ] TimelineViewer (timestamps, regions)
- [ ] ChartsPanel (line, bar, pie charts)
- [ ] EvidenceViewer (heatmaps, spectrograms, metadata)
- [ ] ResultsSummary (verdict, key findings)
- [ ] SidebarNavigation
- [ ] Header (search, notifications, user menu)
- [ ] AnalysisHistory (list, filters, pagination)
- [ ] SettingsPanel

### A.3 Page Templates
- [ ] Landing Page
- [ ] Dashboard Layout
- [ ] Analyze Page (upload, progress, results)
- [ ] History Page
- [ ] Results Detail Page
- [ ] Settings Page
- [ ] Login/Register Pages
- [ ] Error Pages (404, 500)

---

## Appendix B: API Endpoint Quick Reference

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/api/v1/analyze` | POST | `FormData` | `{ request_id, status, websocket_url }` |
| `/api/v1/analyze/{id}/status` | GET | - | `{ request_id, status, progress, stage }` |
| `/api/v1/analyze/{id}/result` | GET | - | `ForensicReport` |
| `/api/v1/analyses` | GET | `?status=&file_type=&limit=&offset=` | `{ analyses: [], total }` |
| `/api/v1/analyses/{id}` | GET | - | `AnalysisDetails` |
| `/api/v1/analyze/{id}/stream` | WS | - | `ProgressUpdate` stream |
| `/api/v1/health` | GET | - | `{ status, version }` |
| `/api/v1/health/detailed` | GET | - | `{ status, components }` |
| `/api/v1/config` | GET | - | `{ max_file_size, supported_types }` |

---

## Appendix C: Performance Budgets

| Metric | Target | Threshold |
|--------|--------|-----------|
| First Contentful Paint | < 1.0s | < 1.5s |
| Largest Contentful Paint | < 2.5s | < 3.0s |
| Time to Interactive | < 3.0s | < 4.0s |
| Cumulative Layout Shift | < 0.1 | < 0.2 |
| First Input Delay | < 100ms | < 200ms |
| Bundle Size (Initial) | < 150KB | < 200KB |
| JavaScript (Total) | < 300KB | < 500KB |
| CSS (Total) | < 50KB | < 100KB |

---

## Appendix D: Accessibility Requirements

- WCAG 2.1 AA compliance
- Keyboard navigation for all interactive elements
- ARIA labels for custom components
- Screen reader compatibility
- Color contrast ratio ≥ 4.5:1
- Focus indicators visible
- Skip to main content link
- Form labels and error messages
- Alternative text for images
- Captcha alternatives for forms

---

**Document Version:** 1.0.0  
**Last Updated:** January 22, 2026  
**Next Review:** February 2026

This document serves as the authoritative guide for Argus Prime frontend implementation. All team members must adhere to the specifications outlined herein to ensure consistency, quality, and maintainability of the codebase.
