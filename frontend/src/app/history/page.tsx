"use client";

import React, { useState, useCallback } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  FileText,
  Calendar,
  Clock,
  Filter,
  RefreshCw,
  Trash2,
  Download,
  Search,
  ChevronDown,
  Video,
  Music,
  Type,
  Image,
} from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { analysisApi } from "@/lib/api";
import { useToast } from "@/components/ui/toast";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Dropdown, DropdownMenu, DropdownMenuItem, DropdownMenuSeparator } from "@/components/ui/dropdown";
import { PaginationWithInfo } from "@/components/ui/pagination";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState, ErrorState } from "@/components/ui/error-state";
import { TrustScoreDisplay } from "@/components/analysis/results/trust-score-display";
import type { AnalysisStatus, FileType, AnalysisSummary, PaginatedAnalyses } from "@/types";

const FILE_TYPE_ICONS: Record<FileType, React.ElementType> = {
  video: Video,
  audio: Music,
  text: Type,
  image: Image,
};

const STATUS_FILTERS = [
  { value: "all", label: "All Status" },
  { value: "completed", label: "Completed" },
  { value: "processing", label: "Processing" },
  { value: "pending", label: "Pending" },
  { value: "failed", label: "Failed" },
];

const FILE_TYPE_FILTERS = [
  { value: "all", label: "All Types" },
  { value: "video", label: "Video" },
  { value: "audio", label: "Audio" },
  { value: "text", label: "Text" },
  { value: "image", label: "Image" },
];

const ITEMS_PER_PAGE = 10;

export default function HistoryPage() {
  const queryClient = useQueryClient();
  const { addToast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [fileTypeFilter, setFileTypeFilter] = useState("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());

  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ["analyses", statusFilter, fileTypeFilter, currentPage],
    queryFn: async () => {
      const result = await analysisApi.listAnalyses({
        status: statusFilter === "all" ? undefined : statusFilter as AnalysisStatus,
        fileType: fileTypeFilter === "all" ? undefined : fileTypeFilter as FileType,
        limit: ITEMS_PER_PAGE,
        offset: (currentPage - 1) * ITEMS_PER_PAGE,
      });
      return result;
    },
    staleTime: 30000,
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => analysisApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["analyses"] });
      addToast("success", "Deleted", "Analysis has been removed");
    },
    onError: (error: Error) => {
      addToast("error", "Error", error.message || "Failed to delete analysis");
    },
  });

  const handleSearch = useCallback((query: string) => {
    setSearchQuery(query);
  }, []);

  const handleSelectAll = () => {
    if (selectedItems.size === (data?.items.length || 0)) {
      setSelectedItems(new Set());
    } else {
      setSelectedItems(new Set(data?.items.map((item) => item.id) || []));
    }
  };

  const handleSelectItem = (id: string) => {
    const newSelected = new Set(selectedItems);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedItems(newSelected);
  };

  const handleDelete = (id: string) => {
    if (confirm("Are you sure you want to delete this analysis?")) {
      deleteMutation.mutate(id);
    }
  };

  const handleBulkDelete = () => {
    if (selectedItems.size === 0) return;
    if (confirm(`Are you sure you want to delete ${selectedItems.size} analyses?`)) {
      selectedItems.forEach((id) => deleteMutation.mutate(id));
      setSelectedItems(new Set());
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getStatusBadge = (status: AnalysisStatus) => {
    const variants: Record<AnalysisStatus, "success" | "warning" | "danger" | "default"> = {
      completed: "success",
      processing: "warning",
      pending: "default",
      failed: "danger",
    };
    const labels: Record<AnalysisStatus, string> = {
      completed: "Completed",
      processing: "Processing",
      pending: "Pending",
      failed: "Failed",
    };
    return <Badge variant={variants[status]}>{labels[status]}</Badge>;
  };

  const filteredData = data?.items.filter((item) =>
    item.fileName.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (error) {
    return (
      <ErrorState
        title="Failed to load history"
        message="Unable to fetch analysis history. Please try again."
        onRetry={() => refetch()}
      />
    );
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
              Analysis History
            </h1>
            <p className="text-slate-500 dark:text-slate-400">
              View and manage your past analysis results
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={() => refetch()}
              leftIcon={<RefreshCw size={16} className={isFetching ? "animate-spin" : ""} />}
              disabled={isFetching}
            >
              Refresh
            </Button>
            <Link href="/analyze">
              <Button variant="primary">New Analysis</Button>
            </Link>
          </div>
        </div>

        <Card className="p-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <Input
                placeholder="Search by file name..."
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                leftIcon={<Search size={16} />}
              />
            </div>
            <div className="flex gap-2">
              <div className="relative group">
                <DropdownButton
                  variant="default"
                  onClick={() => setStatusFilter(statusFilter === "all" ? "completed" : statusFilter === "completed" ? "pending" : "all")}
                >
                  <Filter size={16} />
                  {STATUS_FILTERS.find((f) => f.value === statusFilter)?.label}
                  <ChevronDown size={14} />
                </DropdownButton>
                <div className="absolute right-0 mt-1 w-40 bg-white dark:bg-slate-800 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                  {STATUS_FILTERS.map((filter) => (
                    <button
                      key={filter.value}
                      onClick={() => setStatusFilter(filter.value)}
                      className={`w-full text-left px-4 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-700 first:rounded-t-lg last:rounded-b-lg ${
                        statusFilter === filter.value
                          ? "text-indigo-600 dark:text-indigo-400 font-medium"
                          : "text-slate-700 dark:text-slate-300"
                      }`}
                    >
                      {filter.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="relative group">
                <DropdownButton
                  variant="default"
                >
                  <FileText size={16} />
                  {FILE_TYPE_FILTERS.find((f) => f.value === fileTypeFilter)?.label}
                  <ChevronDown size={14} />
                </DropdownButton>
                <div className="absolute right-0 mt-1 w-40 bg-white dark:bg-slate-800 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                  {FILE_TYPE_FILTERS.map((filter) => (
                    <button
                      key={filter.value}
                      onClick={() => setFileTypeFilter(filter.value)}
                      className={`w-full text-left px-4 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-700 first:rounded-t-lg last:rounded-b-lg ${
                        fileTypeFilter === filter.value
                          ? "text-indigo-600 dark:text-indigo-400 font-medium"
                          : "text-slate-700 dark:text-slate-300"
                      }`}
                    >
                      {filter.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </Card>

        {selectedItems.size > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-4 p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-xl"
          >
            <span className="text-sm font-medium text-indigo-700 dark:text-indigo-300">
              {selectedItems.size} item(s) selected
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleBulkDelete}
              leftIcon={<Trash2 size={14} />}
              className="text-red-600 hover:text-red-700"
            >
              Delete Selected
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSelectedItems(new Set())}
            >
              Clear Selection
            </Button>
          </motion.div>
        )}

        {isLoading ? (
          <div className="space-y-4">
            {Array.from({ length: 5 }).map((_, index) => (
              <Skeleton key={index} className="h-24 w-full rounded-xl" />
            ))}
          </div>
        ) : filteredData && filteredData.length > 0 ? (
          <>
            <div className="space-y-4">
              <div className="flex items-center gap-4 px-4 text-sm text-slate-500 dark:text-slate-400">
                <div className="w-10">
                  <input
                    type="checkbox"
                    checked={selectedItems.size === filteredData.length}
                    onChange={handleSelectAll}
                    className="w-4 h-4 rounded border-slate-300 dark:border-slate-600"
                  />
                </div>
                <div className="flex-1">File Name</div>
                <div className="w-24">Type</div>
                <div className="w-24">Status</div>
                <div className="w-24">Score</div>
                <div className="w-32">Date</div>
                <div className="w-20">Actions</div>
              </div>
              {filteredData.map((analysis, index) => {
                const FileIcon = FILE_TYPE_ICONS[analysis.fileType] || FileText;
                return (
                  <motion.div
                    key={analysis.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <Link href={`/analyze/${analysis.id}`}>
                      <Card className="p-4 cursor-pointer hover:shadow-lg transition-all group">
                        <div className="flex items-center gap-4">
                          <div className="w-10">
                            <input
                              type="checkbox"
                              checked={selectedItems.has(analysis.id)}
                              onChange={(e) => {
                                e.stopPropagation();
                                handleSelectItem(analysis.id);
                              }}
                              className="w-4 h-4 rounded border-slate-300 dark:border-slate-600"
                            />
                          </div>
                          <div
                            className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                              analysis.fileType === "video"
                                ? "bg-blue-100 dark:bg-blue-900/30"
                                : analysis.fileType === "audio"
                                ? "bg-purple-100 dark:bg-purple-900/30"
                                : analysis.fileType === "image"
                                ? "bg-amber-100 dark:bg-amber-900/30"
                                : "bg-slate-100 dark:bg-slate-800"
                            }`}
                          >
                            <FileIcon
                              size={20}
                              className={
                                analysis.fileType === "video"
                                  ? "text-blue-600 dark:text-blue-400"
                                  : analysis.fileType === "audio"
                                  ? "text-purple-600 dark:text-purple-400"
                                  : analysis.fileType === "image"
                                  ? "text-amber-600 dark:text-amber-400"
                                  : "text-slate-600 dark:text-slate-400"
                              }
                            />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-slate-900 dark:text-white truncate group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                              {analysis.fileName}
                            </p>
                            <p className="text-sm text-slate-500 dark:text-slate-400">
                              {formatFileSize(analysis.fileSize)}
                            </p>
                          </div>
                          <div className="w-24">
                            <Badge variant="default" className="capitalize">
                              {analysis.fileType}
                            </Badge>
                          </div>
                          <div className="w-24">{getStatusBadge(analysis.status)}</div>
                          <div className="w-24">
                            {analysis.trustScore !== undefined && (
                              <TrustScoreDisplay
                                score={analysis.trustScore}
                                size="sm"
                                showDetails={false}
                              />
                            )}
                          </div>
                          <div className="w-32 text-sm text-slate-500 dark:text-slate-400">
                            <div className="flex items-center gap-1">
                              <Calendar size={14} />
                              {new Date(analysis.createdAt).toLocaleDateString()}
                            </div>
                            <div className="flex items-center gap-1 mt-0.5">
                              <Clock size={12} />
                              {new Date(analysis.createdAt).toLocaleTimeString()}
                            </div>
                          </div>
                          <div className="w-20 flex items-center gap-1">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDelete(analysis.id);
                              }}
                              className="p-2 rounded-lg text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors opacity-0 group-hover:opacity-100"
                              title="Delete"
                            >
                              <Trash2 size={16} />
                            </button>
                            {analysis.status === "completed" && (
                              <button
                                onClick={(e) => e.stopPropagation()}
                                className="p-2 rounded-lg text-slate-400 hover:text-indigo-500 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 transition-colors opacity-0 group-hover:opacity-100"
                                title="Download"
                              >
                                <Download size={16} />
                              </button>
                            )}
                          </div>
                        </div>
                      </Card>
                    </Link>
                  </motion.div>
                );
              })}
            </div>

            <PaginationWithInfo
              currentPage={currentPage}
              totalPages={Math.ceil((data?.total || 0) / ITEMS_PER_PAGE)}
              totalItems={data?.total || 0}
              pageSize={ITEMS_PER_PAGE}
              onPageChange={setCurrentPage}
            />
          </>
        ) : (
          <EmptyState
            icon={<FileText size={40} />}
            title="No analyses yet"
            description="Upload your first file to start analyzing content for deepfakes."
            action={
              <Link href="/analyze">
                <Button variant="primary">Start Analysis</Button>
              </Link>
            }
          />
        )}
      </motion.div>
    </div>
  );
}

function DropdownButton({
  children,
  variant,
  onClick,
}: {
  children: React.ReactNode;
  variant: "default" | "primary";
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        inline-flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors
        ${
          variant === "default"
            ? "bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600"
            : ""
        }
      `}
    >
      {children}
    </button>
  );
}
