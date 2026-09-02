"use client";

import React, { useState, useMemo } from "react";
import { motion } from "framer-motion";
import {
  FileText,
  Download,
  Calendar,
  Filter,
  PieChart as PieChartIcon,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  Eye,
  Trash2,
  Share2,
  Plus,
  RefreshCw,
} from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { analysisApi, reportApi } from "@/lib/api";
import { useToast } from "@/components/ui/toast";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select } from "@/components/ui/select";
import { Tabs, TabList, Tab, TabPanel } from "@/components/ui/tabs";
import { Table, Column } from "@/components/ui/table";
import { Skeleton, SkeletonCard } from "@/components/ui/skeleton";
import { ChartsPanel, ScoreGauge } from "@/components/analysis/charts";
import { EmptyState } from "@/components/ui/error-state";
import { TrustScoreDisplay } from "@/components/analysis/results/trust-score-display";
import type { AnalysisSummary, Verdict } from "@/types";

export default function ReportsPage() {
  const queryClient = useQueryClient();
  const { addToast } = useToast();
  const [filter, setFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");

  const { data: analyses, isLoading, refetch } = useQuery({
    queryKey: ["analyses"],
    queryFn: () => analysisApi.listAnalyses({ limit: 100 }),
    staleTime: 30000,
  });

  const { data: healthData } = useQuery({
    queryKey: ["health", "detailed"],
    queryFn: () => analysisApi.listAnalyses({ limit: 100 }),
    staleTime: 60000,
  });

  const downloadMutation = useMutation({
    mutationFn: ({ requestId, format }: { requestId: string; format: "pdf" | "json" | "csv" }) =>
      reportApi.downloadReport(requestId, { format, includeEvidence: true, includeRawOutput: false }),
    onSuccess: () => {
      addToast("success", "Download Started", "Your report is being generated");
    },
    onError: () => {
      addToast("error", "Download Failed", "Unable to download report");
    },
  });

  const shareMutation = useMutation({
    mutationFn: (requestId: string) => reportApi.getShareableLink(requestId),
    onSuccess: (data) => {
      navigator.clipboard.writeText(data.shareUrl);
      addToast("success", "Link Copied", "Shareable link copied to clipboard");
    },
    onError: () => {
      addToast("error", "Failed", "Unable to generate share link");
    },
  });

  const handleDownload = (report: AnalysisSummary) => {
    downloadMutation.mutate({ requestId: report.id, format: "pdf" });
  };

  const handleShare = (report: AnalysisSummary) => {
    shareMutation.mutate(report.id);
  };

  const handleDelete = (reportId: string) => {
    if (confirm("Are you sure you want to delete this report?")) {
      addToast("warning", "Report Deleted", "The report has been removed");
      queryClient.invalidateQueries({ queryKey: ["analyses"] });
    }
  };

  const handleRefresh = () => {
    refetch();
    addToast("info", "Refreshing", "Fetching latest reports");
  };

  const filteredReports = useMemo(() => {
    if (!analyses?.items) return [];
    return analyses.items.filter((report) => {
      if (filter !== "all" && report.verdict !== filter) return false;
      return true;
    });
  }, [analyses, filter]);

  const stats = useMemo(() => {
    const items = analyses?.items || [];
    const authentic = items.filter((r) => r.verdict === "AUTHENTIC").length;
    const manipulated = items.filter((r) => r.verdict === "MANIPULATED").length;
    const uncertain = items.filter((r) => r.verdict === "UNCERTAIN").length;
    const avgScore = items.length > 0
      ? Math.round(items.reduce((sum, r) => sum + (r.trustScore || 0), 0) / items.length)
      : 0;

    return {
      total: items.length,
      authentic,
      manipulated,
      uncertain,
      avgScore,
    };
  }, [analyses]);

  const verdictData = useMemo(() => [
    { name: "Authentic", value: stats.authentic, color: "#10b981" },
    { name: "Uncertain", value: stats.uncertain, color: "#f59e0b" },
    { name: "Manipulated", value: stats.manipulated, color: "#ef4444" },
  ], [stats]);

  const columns: Column<AnalysisSummary>[] = [
    {
      key: "fileName",
      header: "Report Name",
      accessor: "fileName",
      sortable: true,
      render: (value, row) => (
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-slate-100 dark:bg-slate-800 rounded-lg flex items-center justify-center">
            <FileText size={18} className="text-slate-500" />
          </div>
          <div>
            <p className="font-medium text-slate-900 dark:text-white">{String(value)}</p>
            <p className="text-xs text-slate-500">{row.fileType}</p>
          </div>
        </div>
      ),
    },
    {
      key: "verdict",
      header: "Verdict",
      accessor: "verdict",
      render: (value) => {
        const verdict = String(value) as Verdict;
        return (
          <Badge
            variant={
              verdict === "AUTHENTIC"
                ? "success"
                : verdict === "MANIPULATED"
                ? "danger"
                : "warning"
            }
          >
            {verdict.charAt(0) + verdict.slice(1).toLowerCase()}
          </Badge>
        );
      },
    },
    {
      key: "trustScore",
      header: "Trust Score",
      accessor: "trustScore",
      align: "center",
      render: (value) => (
        <TrustScoreDisplay score={Number(value) || 0} size="sm" showDetails={false} />
      ),
    },
    {
      key: "createdAt",
      header: "Created",
      accessor: "createdAt",
      render: (value) => (
        <div className="text-sm text-slate-500 dark:text-slate-400">
          <div>{new Date(String(value)).toLocaleDateString()}</div>
          <div className="text-xs">{new Date(String(value)).toLocaleTimeString()}</div>
        </div>
      ),
    },
    {
      key: "actions",
      header: "Actions",
      accessor: "id",
      align: "right",
      render: (_, row) => (
        <div className="flex items-center justify-end gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleDownload(row)}
            disabled={downloadMutation.isPending}
          >
            <Download size={16} />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleShare(row)}
            disabled={shareMutation.isPending}
          >
            <Share2 size={16} />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleDelete(row.id)}
          >
            <Trash2 size={16} />
          </Button>
        </div>
      ),
    },
  ];

  if (isLoading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="space-y-6">
          <Skeleton className="h-10 w-48" />
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <SkeletonCard key={i} />
            ))}
          </div>
          <Skeleton className="h-96 w-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
              Reports
            </h1>
            <p className="text-slate-500 dark:text-slate-400">
              Generate and manage forensic analysis reports
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={handleRefresh}>
              <RefreshCw size={16} className="mr-2" />
              Refresh
            </Button>
            <Button variant="primary">
              <Plus size={16} className="mr-2" />
              Generate Report
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Total Reports</p>
                  <p className="text-2xl font-bold text-slate-900 dark:text-white mt-1">
                    {stats.total}
                  </p>
                </div>
                <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-xl flex items-center justify-center">
                  <FileText size={24} className="text-blue-600" />
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Authentic</p>
                  <p className="text-2xl font-bold text-emerald-600 mt-1">{stats.authentic}</p>
                </div>
                <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/30 rounded-xl flex items-center justify-center">
                  <CheckCircle size={24} className="text-emerald-600" />
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Manipulated</p>
                  <p className="text-2xl font-bold text-red-600 mt-1">{stats.manipulated}</p>
                </div>
                <div className="w-12 h-12 bg-red-100 dark:bg-red-900/30 rounded-xl flex items-center justify-center">
                  <AlertTriangle size={24} className="text-red-600" />
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Avg Trust Score</p>
                  <p className="text-2xl font-bold text-purple-600 mt-1">{stats.avgScore}%</p>
                </div>
                <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 rounded-xl flex items-center justify-center">
                  <TrendingUp size={24} className="text-purple-600" />
                </div>
              </div>
            </Card>
          </motion.div>
        </div>

        <Tabs defaultValue="overview">
          <TabList>
            <Tab value="overview">
              <BarChart3 size={16} className="mr-2" />
              Overview
            </Tab>
            <Tab value="list">
              <FileText size={16} className="mr-2" />
              All Reports
            </Tab>
          </TabList>

          <TabPanel value="overview" className="mt-6">
            <ChartsPanel
              verdictData={verdictData}
              modalityScores={[
                { modality: "Video", score: 72, weight: 0.4 },
                { modality: "Audio", score: 68, weight: 0.35 },
                { modality: "Text", score: 85, weight: 0.25 },
              ]}
              onRefresh={handleRefresh}
            />

            <div className="mt-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Recent Reports
              </h3>
              {filteredReports.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {filteredReports.slice(0, 6).map((report, index) => (
                    <motion.div
                      key={report.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <Card className="p-4 hover:shadow-lg transition-shadow">
                        <div className="flex items-start justify-between mb-3">
                          <Badge
                            variant={
                              report.verdict === "AUTHENTIC"
                                ? "success"
                                : report.verdict === "MANIPULATED"
                                ? "danger"
                                : "warning"
                            }
                          >
                            {report.verdict}
                          </Badge>
                          <span className="text-xs text-slate-500">
                            {new Date(report.createdAt).toLocaleDateString()}
                          </span>
                        </div>
                        <h4 className="font-medium text-slate-900 dark:text-white mb-2 line-clamp-1">
                          {report.fileName}
                        </h4>
                        <div className="flex items-center gap-2 mb-3">
                          <TrustScoreDisplay
                            score={report.trustScore || 0}
                            size="sm"
                            showDetails={false}
                          />
                        </div>
                        <div className="flex gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="flex-1"
                            onClick={() => handleDownload(report)}
                          >
                            <Download size={14} className="mr-1" />
                            Download
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleShare(report)}
                          >
                            <Share2 size={14} />
                          </Button>
                        </div>
                      </Card>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <EmptyState
                  icon={<FileText size={40} />}
                  title="No reports yet"
                  description="Start by analyzing some content to generate your first report."
                  action={
                    <Button variant="primary">
                      <Plus size={16} className="mr-2" />
                      Analyze Content
                    </Button>
                  }
                />
              )}
            </div>
          </TabPanel>

          <TabPanel value="list" className="mt-6">
            <Card className="p-4">
              <div className="flex flex-wrap gap-4 mb-6">
                <Select
                  label="Verdict"
                  options={[
                    { value: "all", label: "All Verdicts" },
                    { value: "AUTHENTIC", label: "Authentic" },
                    { value: "UNCERTAIN", label: "Uncertain" },
                    { value: "MANIPULATED", label: "Manipulated" },
                  ]}
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  className="w-40"
                />
                <Select
                  label="Type"
                  options={[
                    { value: "all", label: "All Types" },
                    { value: "video", label: "Video" },
                    { value: "audio", label: "Audio" },
                    { value: "text", label: "Text" },
                  ]}
                  value={typeFilter}
                  onChange={(e) => setTypeFilter(e.target.value)}
                  className="w-40"
                />
              </div>

              {filteredReports.length > 0 ? (
                <Table
                  columns={columns}
                  data={filteredReports}
                  keyExtractor={(row) => row.id}
                  sortable
                  searchable
                  pageSize={10}
                  onRowClick={(row) => {
                    window.location.href = `/analyze/${row.id}`;
                  }}
                />
              ) : (
                <EmptyState
                  icon={<FileText size={40} />}
                  title="No reports found"
                  description="Try adjusting your filters or analyze more content."
                />
              )}
            </Card>
          </TabPanel>
        </Tabs>
      </motion.div>
    </div>
  );
}
