"use client";

import React, { useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  FileText,
  Calendar,
  Clock,
  Download,
  Share2,
  AlertTriangle,
  CheckCircle,
  Info,
  Loader2,
} from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { analysisApi, reportApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabList, Tab, TabPanel } from "@/components/ui/tabs";
import { TrustScoreDisplay } from "@/components/analysis/results/trust-score-display";
import { ShareLink } from "@/components/analysis/results/share-link";
import { ErrorState } from "@/components/ui/error-state";
import { useToast } from "@/components/ui/toast";

export default function ResultsPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;
  const queryClient = useQueryClient();
  const { addToast } = useToast();

  const { data: result, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["analysis", id],
    queryFn: () => analysisApi.getResult(id),
    retry: 3,
    staleTime: 30000,
  });

  const exportMutation = useMutation({
    mutationFn: (format: "json" | "pdf" | "csv") =>
      reportApi.downloadReport(id, { format, includeEvidence: true, includeRawOutput: false }),
    onSuccess: (blob, format) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `analysis-report-${id}.${format === "json" ? "json" : format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      addToast("success", "Export Complete", `Report downloaded as ${format.toUpperCase()}`);
    },
    onError: () => {
      addToast("error", "Export Failed", "Could not download report");
    },
  });

  const handleExport = (format: "json" | "pdf" | "csv") => {
    exportMutation.mutate(format);
  };

  const getVerdictBadge = (verdict: string) => {
    switch (verdict) {
      case "AUTHENTIC":
        return <Badge variant="success">Authentic</Badge>;
      case "MANIPULATED":
        return <Badge variant="danger">Manipulated</Badge>;
      default:
        return <Badge variant="warning">Uncertain</Badge>;
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 70) return "text-emerald-600 dark:text-emerald-400";
    if (score >= 40) return "text-amber-600 dark:text-amber-400";
    return "text-red-600 dark:text-red-400";
  };

  if (isLoading) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <div className="flex items-center gap-4">
            <Link href="/history">
              <Button variant="ghost" size="sm">
                <ArrowLeft size={18} />
                Back
              </Button>
            </Link>
            <div className="flex-1">
              <div className="h-8 w-64 bg-slate-200 dark:bg-slate-700 rounded animate-pulse" />
              <div className="flex items-center gap-4 mt-2">
                <div className="h-4 w-32 bg-slate-200 dark:bg-slate-700 rounded animate-pulse" />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <Card className="p-6 sticky top-24">
                <div className="h-64 bg-slate-100 dark:bg-slate-800 rounded-xl animate-pulse" />
              </Card>
            </div>
            <div className="lg:col-span-2 space-y-6">
              <Card className="p-6">
                <div className="h-48 bg-slate-100 dark:bg-slate-800 rounded-xl animate-pulse" />
              </Card>
              <Card className="p-6">
                <div className="h-32 bg-slate-100 dark:bg-slate-800 rounded-xl animate-pulse" />
              </Card>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
          <ErrorState
            title="Failed to Load Analysis"
            message={error instanceof Error ? error.message : "An unknown error occurred"}
            onRetry={() => refetch()}
          />
      </div>
    );
  }

  if (!result) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
          <ErrorState
            title="Analysis Not Found"
            message="The requested analysis could not be found"
            onGoHome={() => router.push("/history")}
          />
      </div>
    );
  }

  const trustScore = result.trustScore;
  const executiveSummary = result.executiveSummary;
  const findings = result.findings || {};
  const evidence = result.evidence || {};
  const recommendations = result.recommendations || [];

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        <div className="flex items-center gap-4">
          <Link href="/history">
            <Button variant="ghost" size="sm">
              <ArrowLeft size={18} />
              Back
            </Button>
          </Link>
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
              {result.fileInfo?.fileName || "Analysis Result"}
            </h1>
            <div className="flex items-center gap-4 mt-1 text-sm text-slate-500 dark:text-slate-400">
              <span className="flex items-center gap-1">
                <Calendar size={14} />
                {new Date(result.createdAt).toLocaleDateString()}
              </span>
              <span className="flex items-center gap-1">
                <Clock size={14} />
                {new Date(result.createdAt).toLocaleTimeString()}
              </span>
              {executiveSummary?.verdict && getVerdictBadge(executiveSummary.verdict)}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <ShareLink analysisId={id} />
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleExport("json")}
              disabled={exportMutation.isPending}
            >
              {exportMutation.isPending ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Download size={16} />
              )}
              Export
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <Card className="p-6 sticky top-24">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
                Trust Score
              </h2>
                <TrustScoreDisplay
                  score={trustScore?.overallScore || 0}
                breakdown={
                  trustScore?.breakdown
                    ? {
                        authenticity: trustScore.breakdown.video || 50,
                        credibility: trustScore.breakdown.audio || 50,
                        manipulation: 100 - (trustScore.breakdown.video || 50),
                        sentiment: trustScore.breakdown.text || 50,
                      }
                    : undefined
                }
              />
              {trustScore?.confidence && (
                <div className="mt-4 text-center">
                  <Badge variant="info">
                    {Math.round(trustScore.confidence * 100)}% Confidence
                  </Badge>
                </div>
              )}
            </Card>

            <Card className="p-6">
              <h3 className="text-sm font-semibold text-slate-900 dark:text-white mb-4">
                File Information
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Type</span>
                  <span className="text-slate-900 dark:text-white capitalize">
                    {result.fileInfo?.fileType || "Unknown"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500 dark:text-slate-400">Size</span>
                  <span className="text-slate-900 dark:text-white">
                      {result.fileInfo?.fileSize
                        ? `${(result.fileInfo.fileSize / 1024 / 1024).toFixed(2)} MB`
                      : "Unknown"}
                  </span>
                </div>
              </div>
            </Card>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <Card className="p-6">
              <Tabs defaultValue="summary">
                <TabList>
                  <Tab value="summary">Summary</Tab>
                  <Tab value="findings">Findings</Tab>
                  <Tab value="evidence">Evidence</Tab>
                  <Tab value="recommendations">Recommendations</Tab>
                </TabList>

                <TabPanel value="summary" className="mt-4">
                  <div className="space-y-4">
                    {executiveSummary?.verdict && (
                      <div className="flex items-center gap-2">
                        {executiveSummary.verdict === "AUTHENTIC" ? (
                          <CheckCircle className="text-emerald-500" size={20} />
                        ) : executiveSummary.verdict === "MANIPULATED" ? (
                          <AlertTriangle className="text-red-500" size={20} />
                        ) : (
                          <Info className="text-amber-500" size={20} />
                        )}
                        <span className="font-medium text-slate-900 dark:text-white">
                          Verdict: {executiveSummary.verdict}
                        </span>
                      </div>
                    )}
                    <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                      {result.reportId ? "Analysis completed. Review detailed findings below." : "No summary available."}
                    </p>
                  </div>
                </TabPanel>

                <TabPanel value="findings" className="mt-4">
                  <div className="space-y-6">
                    {findings.video && (
                      <div>
                        <h4 className="font-medium text-slate-900 dark:text-white mb-3">
                          Video Analysis
                        </h4>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                            <div className="text-sm text-slate-500 dark:text-slate-400">
                              Frames Analyzed
                            </div>
                            <div className="text-lg font-semibold text-slate-900 dark:text-white">
                              {findings.video?.framesAnalyzed || 0}
                            </div>
                          </div>
                          <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                            <div className="text-sm text-slate-500 dark:text-slate-400">
                              Faces Detected
                            </div>
                            <div className="text-lg font-semibold text-slate-900 dark:text-white">
                              {findings.video?.facesDetected || 0}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {findings.audio && (
                      <div>
                        <h4 className="font-medium text-slate-900 dark:text-white mb-3">
                          Audio Analysis
                        </h4>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                            <div className="text-sm text-slate-500 dark:text-slate-400">
                              Duration
                            </div>
                            <div className="text-lg font-semibold text-slate-900 dark:text-white">
                              {findings.audio?.durationSeconds
                                ? `${findings.audio.durationSeconds.toFixed(1)}s`
                                : "N/A"}
                            </div>
                          </div>
                          <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                            <div className="text-sm text-slate-500 dark:text-slate-400">
                              Vocoder Detected
                            </div>
                            <div className="text-lg font-semibold text-slate-900 dark:text-white">
                              {findings.audio?.vocoderDetected || "None"}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {findings.text && (
                      <div>
                        <h4 className="font-medium text-slate-900 dark:text-white mb-3">
                          Text Analysis
                        </h4>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                            <div className="text-sm text-slate-500 dark:text-slate-400">
                              Word Count
                            </div>
                            <div className="text-lg font-semibold text-slate-900 dark:text-white">
                              {findings.text?.wordCount || 0}
                            </div>
                          </div>
                          <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                            <div className="text-sm text-slate-500 dark:text-slate-400">
                              Language
                            </div>
                            <div className="text-lg font-semibold text-slate-900 dark:text-white">
                              {findings.text?.languageDetected || "Unknown"}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </TabPanel>

                <TabPanel value="evidence" className="mt-4">
                  <div className="space-y-4">
                    {evidence.localization?.timestamps?.length > 0 ? (
                      <div>
                        <h4 className="font-medium text-slate-900 dark:text-white mb-3">
                          Manipulation Timestamps
                        </h4>
                        <div className="space-y-2">
                          {evidence.localization.timestamps.map(
                            (ts: { start: number; end: number; confidence: number }, idx: number) => (
                              <div
                                key={idx}
                                className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg"
                              >
                                <span className="text-slate-600 dark:text-slate-300">
                                  {ts.start.toFixed(2)}s - {ts.end.toFixed(2)}s
                                </span>
                                <Badge variant="warning">
                                  {Math.round(ts.confidence * 100)}% confidence
                                </Badge>
                              </div>
                            )
                          )}
                        </div>
                      </div>
                    ) : (
                      <p className="text-slate-500 dark:text-slate-400">
                        No manipulation timestamps detected.
                      </p>
                    )}
                  </div>
                </TabPanel>

                <TabPanel value="recommendations" className="mt-4">
                  {recommendations.length > 0 ? (
                    <ul className="space-y-3">
                      {recommendations.map((rec: string, idx: number) => (
                        <li key={idx} className="flex items-start gap-3">
                          <Info size={18} className="text-blue-500 mt-0.5 flex-shrink-0" />
                          <span className="text-slate-600 dark:text-slate-300">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-slate-500 dark:text-slate-400">
                      No specific recommendations for this analysis.
                    </p>
                  )}
                </TabPanel>
              </Tabs>
            </Card>

            <Card className="p-6">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Score Breakdown
              </h2>
              <div className="grid grid-cols-2 gap-4">
                {[
                  {
                    label: "Trust Score",
                    score: trustScore?.overallScore || 0,
                  },
                  {
                    label: "Confidence",
                    score: trustScore?.confidence
                      ? trustScore.confidence * 100
                      : 0,
                  },
                ].map((item) => (
                  <div
                    key={item.label}
                    className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg"
                  >
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-slate-500 dark:text-slate-400">
                        {item.label}
                      </span>
                      <span className={`font-semibold ${getScoreColor(item.score)}`}>
                        {Math.round(item.score)}
                      </span>
                    </div>
                    <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          getScoreColor(item.score).replace("text-", "bg-")
                        }`}
                        style={{ width: `${item.score}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
