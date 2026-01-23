import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { API_CONFIG } from '@/lib/config';
import type {
  AnalysisSubmitRequest,
  AnalysisSubmitResponse,
  AnalysisStatusResponse,
  ForensicReport,
  PaginatedAnalyses,
  HealthCheckResponse,
  DetailedHealthResponse,
  ReadinessResponse,
  ConfigResponse,
  AnalysisOptions,
  ReportExportOptions,
} from '@/types';

const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_CONFIG.baseUrl,
    timeout: API_CONFIG.timeout.request,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
      return Promise.reject(error);
    }
  );

  return client;
};

export const apiClient = createApiClient();

// ============================================
// Analysis API
// ============================================

export const analysisApi = {
  async submit(request: AnalysisSubmitRequest): Promise<AnalysisSubmitResponse> {
    const response = await apiClient.post<AnalysisSubmitResponse>('/api/v1/analyze', request);
    return response.data;
  },

  async getStatus(requestId: string): Promise<AnalysisStatusResponse> {
    const response = await apiClient.get<AnalysisStatusResponse>(`/api/v1/analyze/${requestId}/status`);
    return response.data;
  },

  async getResult(requestId: string): Promise<ForensicReport> {
    const response = await apiClient.get<ForensicReport>(`/api/v1/analyze/${requestId}/result`);
    return response.data;
  },

  async listAnalyses(params?: {
    status?: string;
    fileType?: string;
    limit?: number;
    offset?: number;
  }): Promise<PaginatedAnalyses> {
    const response = await apiClient.get<PaginatedAnalyses>('/api/v1/analyses', { params });
    return response.data;
  },

  async getDetails(requestId: string): Promise<ForensicReport> {
    const response = await apiClient.get<ForensicReport>(`/api/v1/analyses/${requestId}`);
    return response.data;
  },

  async cancel(requestId: string): Promise<void> {
    await apiClient.post(`/api/v1/analyze/${requestId}/cancel`);
  },

  async delete(requestId: string): Promise<void> {
    await apiClient.delete(`/api/v1/analyses/${requestId}`);
  },
};

// ============================================
// Health API
// ============================================

export const healthApi = {
  async basic(): Promise<HealthCheckResponse> {
    const response = await apiClient.get<HealthCheckResponse>('/api/v1/health');
    return response.data;
  },

  async detailed(): Promise<DetailedHealthResponse> {
    const response = await apiClient.get<DetailedHealthResponse>('/api/v1/health/detailed');
    return response.data;
  },

  async ready(): Promise<ReadinessResponse> {
    const response = await apiClient.get<ReadinessResponse>('/api/v1/health/ready');
    return response.data;
  },

  async live(): Promise<HealthCheckResponse> {
    const response = await apiClient.get<HealthCheckResponse>('/api/v1/health/live');
    return response.data;
  },

  async config(): Promise<ConfigResponse> {
    const response = await apiClient.get<ConfigResponse>('/api/v1/config');
    return response.data;
  },
};

// ============================================
// Report API
// ============================================

export const reportApi = {
  async downloadReport(
    requestId: string,
    options: ReportExportOptions
  ): Promise<Blob> {
    const response = await apiClient.get(`/api/v1/analyses/${requestId}/export`, {
      params: options,
      responseType: 'blob',
    });
    return response.data;
  },

  async getShareableLink(requestId: string): Promise<{ shareUrl: string; expiresAt: string }> {
    const response = await apiClient.post(`/api/v1/analyses/${requestId}/share`);
    return response.data;
  },
};

// ============================================
// Utility Functions
// ============================================

export const createAnalysisSocket = (requestId: string): WebSocket => {
  const wsUrl = API_CONFIG.wsUrl.replace('http', 'ws');
  return new WebSocket(`${wsUrl}/api/v1/analyze/${requestId}/stream`);
};

export const handleApiError = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ detail?: string }>;
    if (axiosError.response?.data?.detail) {
      return axiosError.response.data.detail;
    }
    if (axiosError.message) {
      return axiosError.message;
    }
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
};

// ============================================
// Convenience Exports
// ============================================

export async function analyzeFile(
  formData: FormData,
  onProgress?: (progress: number) => void
): Promise<AnalysisSubmitResponse> {
  const response = await apiClient.post<AnalysisSubmitResponse>('/api/v1/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: API_CONFIG.timeout.upload,
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(Math.min(progress, 50));
      }
    },
  });
  return response.data;
}

export async function getAnalysisStatus(id: string): Promise<AnalysisStatusResponse> {
  const response = await apiClient.get<AnalysisStatusResponse>(`/api/v1/analyze/${id}/status`);
  return response.data;
}

export async function getAnalysisResult(id: string): Promise<ForensicReport> {
  const response = await apiClient.get<ForensicReport>(`/api/v1/analyze/${id}/result`);
  return response.data;
}

export async function getAnalyses(params?: {
  status?: string;
  fileType?: string;
  limit?: number;
  offset?: number;
}): Promise<PaginatedAnalyses> {
  const response = await apiClient.get<PaginatedAnalyses>('/api/v1/analyses', { params });
  return response.data;
}

// Named exports for convenience
export type { AxiosError };
