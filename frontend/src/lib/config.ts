export const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || '/api',
  wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001',
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

export const FILE_LIMITS = {
  maxSize: 500 * 1024 * 1024,
  acceptedTypes: {
    video: ['video/mp4', 'video/avi', 'video/mov', 'video/webm'],
    audio: ['audio/wav', 'audio/mp3', 'audio/flac', 'audio/m4a'],
    image: ['image/jpeg', 'image/png', 'image/webp'],
    text: ['text/plain', 'application/pdf', 'text/markdown'],
  },
};

export const QUERY_KEYS = {
  analyses: {
    all: ['analyses'] as const,
    lists: () => [...QUERY_KEYS.analyses.all, 'list'] as const,
    list: (filters: { status?: string; fileType?: string; limit?: number; offset?: number }) =>
      [...QUERY_KEYS.analyses.lists(), filters] as const,
    details: () => [...QUERY_KEYS.analyses.all, 'detail'] as const,
    detail: (id: string) => [...QUERY_KEYS.analyses.details(), id] as const,
    results: () => [...QUERY_KEYS.analyses.all, 'result'] as const,
    result: (id: string) => [...QUERY_KEYS.analyses.results(), id] as const,
  },
  health: {
    all: ['health'] as const,
    basic: () => [...QUERY_KEYS.health.all, 'basic'] as const,
    detailed: () => [...QUERY_KEYS.health.all, 'detailed'] as const,
    ready: () => [...QUERY_KEYS.health.all, 'ready'] as const,
    config: () => [...QUERY_KEYS.health.all, 'config'] as const,
  },
} as const;
