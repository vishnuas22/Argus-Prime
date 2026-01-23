// API Types for Argus Prime Frontend
// These types match the backend Pydantic schemas in src/api/schemas.py

// ============================================
// Enums
// ============================================

export type Verdict = 'AUTHENTIC' | 'MANIPULATED' | 'UNCERTAIN';

export type FileType = 'video' | 'audio' | 'image' | 'text';

export type AnalysisStatus = 'pending' | 'processing' | 'completed' | 'failed';

export type ConfidenceLevel = 'high' | 'medium' | 'low';

// ============================================
// Request Types
// ============================================

export interface VideoOptions {
  fps?: number;
  extractFaces?: boolean;
  generateHeatmaps?: boolean;
  rppgAnalysis?: boolean;
}

export interface AudioOptions {
  sampleRate?: number;
  segmentDuration?: number;
  applyVad?: boolean;
}

export interface TextOptions {
  language?: string;
  includePerplexity?: boolean;
  includeBurstiness?: boolean;
}

export interface AnalysisOptions {
  video?: VideoOptions;
  audio?: AudioOptions;
  text?: TextOptions;
}

export interface AnalysisSubmitRequest {
  fileId: string;
  analysisOptions?: AnalysisOptions;
}

export interface FileUploadMetadata {
  fileName: string;
  contentType: string;
  fileSize: number;
  md5Hash?: string;
}

// ============================================
// Response Types
// ============================================

export interface TrustScore {
  overallScore: number;
  confidence: number;
  confidenceLevel: ConfidenceLevel;
  breakdown: {
    video: number;
    audio: number;
    text: number;
  };
  verdict: Verdict;
}

export interface FrameAnalysis {
  frameNumber: number;
  timestamp: number;
  faceBbox?: number[];
  manipulationScore: number;
  heatmapPath?: string;
}

export interface VideoResult {
  framesAnalyzed: number;
  facesDetected: number;
  overallScore: number;
  perFrameScores: FrameAnalysis[];
  heatmapPath?: string;
  branchScores: {
    global: number;
    texture: number;
    spectral: number;
  };
  rppgResult?: Record<string, unknown>;
}

export interface SegmentAnalysis {
  startTime: number;
  endTime: number;
  manipulationScore: number;
  vocoderClass?: string;
  spectrogramPath?: string;
}

export interface AudioResult {
  durationSeconds: number;
  segmentsAnalyzed: number;
  overallScore: number;
  perSegmentScores: SegmentAnalysis[];
  vocoderDetected?: string;
  artifactEvidence: Record<string, number>;
}

export interface SentenceAnalysis {
  sentence: string;
  startChar: number;
  endChar: number;
  manipulationScore: number;
  perplexity: number;
  burstiness: number;
}

export interface TextResult {
  wordCount: number;
  languageDetected: string;
  overallScore: number;
  perSentenceScores: SentenceAnalysis[];
  perplexityScores: Record<string, number>;
  burstinessMetrics: Record<string, number>;
}

export interface LocalizationEntry {
  start: number;
  end: number;
  confidence: number;
}

export interface SpatialRegion {
  polygon: number[][];
  confidence: number;
}

export interface Localization {
  timestamps: LocalizationEntry[];
  spatialRegions: SpatialRegion[];
  heatmapOverlayPath?: string;
}

export interface Evidence {
  heatmaps: string[];
  spectrograms: string[];
  localization: Localization;
  c2paMetadata?: Record<string, unknown>;
  adversarialCheck?: Record<string, unknown>;
}

export interface ExecutiveSummary {
  verdict: Verdict;
  trustScore: number;
  confidence: number;
}

export interface ForensicReport {
  reportId: string;
  requestId: string;
  createdAt: string;
  fileInfo: {
    fileName: string;
    fileType: string;
    fileSize: number;
    contentType: string;
  };
  executiveSummary: ExecutiveSummary;
  trustScore: TrustScore;
  findings: {
    video?: VideoResult;
    audio?: AudioResult;
    text?: TextResult;
  };
  evidence: Evidence;
  recommendations: string[];
  rawOutput?: Record<string, unknown>;
}

export interface AnalysisStatusResponse {
  requestId: string;
  status: AnalysisStatus;
  progress: number;
  currentStage?: string;
  estimatedTimeRemaining?: number;
  errorMessage?: string;
}

export interface AnalysisSubmitResponse {
  requestId: string;
  status: 'pending';
  message: string;
  websocketUrl: string;
}

export interface ProgressUpdate {
  requestId: string;
  stage: string;
  progress: number;
  message: string;
  timestamp: string;
}

// ============================================
// List/Pagination Types
// ============================================

export interface PaginationParams {
  status?: string;
  fileType?: string;
  limit?: number;
  offset?: number;
}

export interface PaginatedAnalyses {
  items: AnalysisSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface AnalysisSummary {
  id: string;
  fileName: string;
  fileType: FileType;
  fileSize: number;
  status: AnalysisStatus;
  verdict?: Verdict;
  trustScore?: number;
  createdAt: string;
  completedAt?: string;
}

// ============================================
// Health Check Types
// ============================================

export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  version: string;
}

export interface DetailedHealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  components: {
    database: ComponentHealth;
    cache: ComponentHealth;
    models: ComponentHealth;
    storage: ComponentHealth;
  };
}

export interface ComponentHealth {
  status: 'healthy' | 'unhealthy';
  latencyMs: number;
  error?: string;
}

export interface ReadinessResponse {
  ready: boolean;
  checks: Record<string, boolean>;
}

export interface ConfigResponse {
  maxFileSize: number;
  allowedFileTypes: string[];
  defaultAnalysisOptions: AnalysisOptions;
}

// ============================================
// WebSocket Types
// ============================================

export interface WebSocketMessage {
  type: 'progress' | 'status' | 'result' | 'error';
  payload: ProgressUpdate | AnalysisStatusResponse | ForensicReport | { message: string };
}

export interface WebSocketConfig {
  url: string;
  reconnectInterval: number;
  maxReconnectAttempts: number;
}

// ============================================
// Analysis Flow Types
// ============================================

export interface AnalysisFile {
  id: string;
  file: File;
  metadata: FileUploadMetadata;
  progress: number;
  status: AnalysisStatus;
  error?: string;
  resultId?: string;
}

export interface AnalysisStage {
  id: string;
  label: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  progress: number;
  message?: string;
}

export const ANALYSIS_STAGES: AnalysisStage[] = [
  { id: 'pending', label: 'Pending', status: 'pending', progress: 0 },
  { id: 'extracting', label: 'Extracting', status: 'pending', progress: 0 },
  { id: 'video_detection', label: 'Video Detection', status: 'pending', progress: 0 },
  { id: 'audio_detection', label: 'Audio Detection', status: 'pending', progress: 0 },
  { id: 'text_detection', label: 'Text Detection', status: 'pending', progress: 0 },
  { id: 'fusing_scores', label: 'Fusing Scores', status: 'pending', progress: 0 },
  { id: 'generating_report', label: 'Generating Report', status: 'pending', progress: 0 },
  { id: 'completed', label: 'Completed', status: 'pending', progress: 0 },
];

// ============================================
// UI State Types
// ============================================

export interface ToastNotification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message?: string;
  duration?: number;
}

export interface ModalState {
  isOpen: boolean;
  modalType: string | null;
  modalData: unknown;
}

export interface SidebarState {
  isCollapsed: boolean;
}

export interface ThemeState {
  theme: 'light' | 'dark' | 'system';
}

// ============================================
// Chart/Data Visualization Types
// ============================================

export interface ChartDataPoint {
  x: number | string;
  y: number;
  label?: string;
}

export interface TimeSeriesData {
  timestamp: number;
  value: number;
}

export interface ScoreDistribution {
  range: string;
  count: number;
  percentage: number;
}

export interface ModalityContribution {
  modality: 'video' | 'audio' | 'text';
  weight: number;
  contribution: number;
  percentage: number;
}

// ============================================
// Report Download Types
// ============================================

export interface ReportExportOptions {
  format: 'json' | 'pdf' | 'csv';
  includeEvidence: boolean;
  includeRawOutput: boolean;
}

export interface DownloadProgress {
  status: 'idle' | 'preparing' | 'downloading' | 'completed' | 'error';
  progress: number;
  downloadUrl?: string;
  error?: string;
}
