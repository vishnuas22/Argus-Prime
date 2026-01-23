"use client";

import { useState, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { analyzeFile, getAnalysisResult, getAnalysisStatus, getAnalyses } from "@/lib/api";
import { useAnalysisStore } from "@/stores/analysis.store";
import { useToast } from "@/components/ui/toast";

export function useAnalysis() {
  const queryClient = useQueryClient();
  const { addToast } = useToast();
  const { setCurrentAnalysis, addToHistory } = useAnalysisStore();

  const [uploadProgress, setUploadProgress] = useState(0);

  const analysesQuery = useQuery({
    queryKey: ["analyses"],
    queryFn: () => getAnalyses(),
    staleTime: 30000,
  });

  const analysisMutation = useMutation({
    mutationFn: (formData: FormData) => analyzeFile(formData, setUploadProgress),
    onSuccess: (data) => {
      addToast("success", "Analysis Complete", `Analysis started with request ID: ${data.requestId}`);
      queryClient.invalidateQueries({ queryKey: ["analyses"] });
    },
    onError: (error: Error) => {
      addToast("error", "Analysis Failed", error.message || "An error occurred during analysis");
    },
  });

  const getAnalysisResultFn = useCallback(async (id: string) => {
    try {
      const result = await getAnalysisResult(id);
      return result;
    } catch (error) {
      console.error("Failed to get analysis result:", error);
      throw error;
    }
  }, []);

  const getAnalysisStatusFn = useCallback(async (id: string) => {
    try {
      const status = await getAnalysisStatus(id);
      return status;
    } catch (error) {
      console.error("Failed to get analysis status:", error);
      throw error;
    }
  }, []);

  return {
    analyses: analysesQuery.data || [],
    isLoading: analysesQuery.isLoading,
    isError: analysesQuery.isError,
    analyze: analysisMutation.mutate,
    isAnalyzing: analysisMutation.isPending,
    uploadProgress,
    getAnalysisResult: getAnalysisResultFn,
    getAnalysisStatus: getAnalysisStatusFn,
    refetch: analysesQuery.refetch,
  };
}
