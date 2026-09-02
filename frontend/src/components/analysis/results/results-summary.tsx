"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  Shield,
  AlertTriangle,
  HelpCircle,
  CheckCircle,
  XCircle,
  FileText,
  Video,
  Music,
  Type,
  Download,
  Share2,
  RefreshCw,
  Lightbulb,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TrustScoreDisplay } from "@/components/analysis/results/trust-score-display";
import type { Verdict, ForensicReport } from "@/types";

interface ResultsSummaryProps {
  report: ForensicReport;
  onDownload?: () => void;
  onShare?: () => void;
  onAnalyzeAnother?: () => void;
  className?: string;
}

const VERDICT_CONFIG = {
  AUTHENTIC: {
    icon: CheckCircle,
    color: "text-emerald-500",
    bgColor: "bg-emerald-50 dark:bg-emerald-900/20",
    borderColor: "border-emerald-200 dark:border-emerald-800",
    label: "Authentic",
    description: "No manipulation indicators detected in this content.",
  },
  MANIPULATED: {
    icon: XCircle,
    color: "text-red-500",
    bgColor: "bg-red-50 dark:bg-red-900/20",
    borderColor: "border-red-200 dark:border-red-800",
    label: "Manipulated",
    description: "Multiple manipulation indicators were detected in this content.",
  },
  UNCERTAIN: {
    icon: AlertTriangle,
    color: "text-amber-500",
    bgColor: "bg-amber-50 dark:bg-amber-900/20",
    borderColor: "border-amber-200 dark:border-amber-800",
    label: "Uncertain",
    description: "The analysis returned mixed results. Manual review recommended.",
  },
};

const MODALITY_ICONS = {
  video: Video,
  audio: Music,
  text: Type,
};

export function ResultsSummary({
  report,
  onDownload,
  onShare,
  onAnalyzeAnother,
  className = "",
}: ResultsSummaryProps) {
  const verdictConfig = VERDICT_CONFIG[report.trustScore.verdict];
  const VerdictIcon = verdictConfig.icon;

  const hasVideo = report.findings.video;
  const hasAudio = report.findings.audio;
  const hasText = report.findings.text;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`space-y-6 ${className}`}
    >
      <Card className={`p-6 border-2 ${verdictConfig.borderColor} ${verdictConfig.bgColor}`}>
        <div className="flex flex-col md:flex-row md:items-center gap-6">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 15 }}
            className={`w-20 h-20 rounded-full ${verdictConfig.bgColor} flex items-center justify-center`}
          >
            <VerdictIcon size={40} className={verdictConfig.color} />
          </motion.div>

          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
                {verdictConfig.label}
              </h2>
              <Badge
                variant={
                  report.trustScore.verdict === "AUTHENTIC"
                    ? "success"
                    : report.trustScore.verdict === "MANIPULATED"
                    ? "danger"
                    : "warning"
                }
              >
                {report.trustScore.verdict}
              </Badge>
            </div>
            <p className="text-slate-600 dark:text-slate-400 mb-4">
              {verdictConfig.description}
            </p>
            <div className="flex flex-wrap gap-4 text-sm text-slate-500 dark:text-slate-400">
              <span className="flex items-center gap-1">
                <FileText size={14} />
                {report.fileInfo.fileName}
              </span>
              <span>
                {(report.fileInfo.fileSize / 1024 / 1024).toFixed(2)} MB
              </span>
              <span className="capitalize">{report.fileInfo.fileType}</span>
            </div>
          </div>

          <div className="hidden md:block">
            <TrustScoreDisplay
              score={report.trustScore.overallScore}
              size="lg"
              showDetails={false}
            />
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="p-4">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                <Video size={20} className="text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <h3 className="font-medium text-slate-900 dark:text-white">Video</h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  {hasVideo ? `${report.findings.video?.framesAnalyzed || 0} frames` : "Not analyzed"}
                </p>
              </div>
            </div>
            {hasVideo && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-500 dark:text-slate-400">Score</span>
                  <span className="font-medium text-slate-900 dark:text-white">
                    {((report.findings.video?.overallScore || 0) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(report.findings.video?.overallScore || 0) * 100}%` }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                    className={`h-full rounded-full ${
                      (report.findings.video?.overallScore || 0) >= 0.7
                        ? "bg-red-500"
                        : (report.findings.video?.overallScore || 0) >= 0.4
                        ? "bg-amber-500"
                        : "bg-emerald-500"
                    }`}
                  />
                </div>
              </div>
            )}
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="p-4">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                <Music size={20} className="text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <h3 className="font-medium text-slate-900 dark:text-white">Audio</h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  {hasAudio ? `${report.findings.audio?.segmentsAnalyzed || 0} segments` : "Not analyzed"}
                </p>
              </div>
            </div>
            {hasAudio && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-500 dark:text-slate-400">Score</span>
                  <span className="font-medium text-slate-900 dark:text-white">
                    {((report.findings.audio?.overallScore || 0) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(report.findings.audio?.overallScore || 0) * 100}%` }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                    className={`h-full rounded-full ${
                      (report.findings.audio?.overallScore || 0) >= 0.7
                        ? "bg-red-500"
                        : (report.findings.audio?.overallScore || 0) >= 0.4
                        ? "bg-amber-500"
                        : "bg-emerald-500"
                    }`}
                  />
                </div>
              </div>
            )}
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="p-4">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-lg bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
                <Type size={20} className="text-amber-600 dark:text-amber-400" />
              </div>
              <div>
                <h3 className="font-medium text-slate-900 dark:text-white">Text</h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  {hasText ? `${report.findings.text?.wordCount || 0} words` : "Not analyzed"}
                </p>
              </div>
            </div>
            {hasText && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-500 dark:text-slate-400">Score</span>
                  <span className="font-medium text-slate-900 dark:text-white">
                    {((report.findings.text?.overallScore || 0) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(report.findings.text?.overallScore || 0) * 100}%` }}
                    transition={{ duration: 0.5, delay: 0.4 }}
                    className={`h-full rounded-full ${
                      (report.findings.text?.overallScore || 0) >= 0.7
                        ? "bg-red-500"
                        : (report.findings.text?.overallScore || 0) >= 0.4
                        ? "bg-amber-500"
                        : "bg-emerald-500"
                    }`}
                  />
                </div>
              </div>
            )}
          </Card>
        </motion.div>
      </div>

      {report.recommendations && report.recommendations.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card className="p-6">
            <div className="flex items-center gap-2 mb-4">
              <Lightbulb size={20} className="text-amber-500" />
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                Recommendations
              </h3>
            </div>
            <ul className="space-y-2">
              {report.recommendations.map((recommendation, index) => (
                <li
                  key={index}
                  className="flex items-start gap-2 text-slate-600 dark:text-slate-400"
                >
                  <span className="mt-1 w-1.5 h-1.5 rounded-full bg-indigo-500 flex-shrink-0" />
                  {recommendation}
                </li>
              ))}
            </ul>
          </Card>
        </motion.div>
      )}

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="flex flex-wrap gap-3"
      >
        {onDownload && (
          <Button variant="primary" onClick={onDownload} leftIcon={<Download size={16} />}>
            Download Report
          </Button>
        )}
        {onShare && (
          <Button variant="outline" onClick={onShare} leftIcon={<Share2 size={16} />}>
            Share Results
          </Button>
        )}
        {onAnalyzeAnother && (
          <Button variant="ghost" onClick={onAnalyzeAnother} leftIcon={<RefreshCw size={16} />}>
            Analyze Another
          </Button>
        )}
      </motion.div>
    </motion.div>
  );
}
