"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  FileText,
  Brain,
  Shield,
  AlertTriangle,
  CheckCircle,
  Loader2,
} from "lucide-react";

interface Stage {
  id: string;
  label: string;
  icon: React.ReactNode;
  description: string;
}

interface ProgressTrackerProps {
  currentStage: number;
  stages: Stage[];
  progress: number;
  currentFile?: string;
  error?: string;
}

const defaultStages: Stage[] = [
  {
    id: "upload",
    label: "Uploading",
    icon: <FileText size={18} />,
    description: "Processing file upload",
  },
  {
    id: "preprocessing",
    label: "Preprocessing",
    icon: <Loader2 size={18} className="animate-spin" />,
    description: "Preparing document for analysis",
  },
  {
    id: "extraction",
    label: "Extracting",
    icon: <Brain size={18} />,
    description: "Extracting text and entities",
  },
  {
    id: "analysis",
    label: "Analyzing",
    icon: <Brain size={18} />,
    description: "Running AI analysis",
  },
  {
    id: "scoring",
    label: "Scoring",
    icon: <Shield size={18} />,
    description: "Calculating trust scores",
  },
  {
    id: "complete",
    label: "Complete",
    icon: <CheckCircle size={18} />,
    description: "Analysis finished",
  },
];

export function ProgressTracker({
  currentStage,
  stages = defaultStages,
  progress,
  currentFile,
  error,
}: ProgressTrackerProps) {
  return (
    <div className="w-full max-w-2xl mx-auto">
      {currentFile && (
        <div className="mb-6 flex items-center gap-3 px-4 py-3 bg-slate-50 dark:bg-slate-800/50 rounded-lg">
          <FileText size={18} className="text-slate-400" />
          <span className="text-sm text-slate-600 dark:text-slate-300 truncate">
            {currentFile}
          </span>
        </div>
      )}

      <div className="relative">
        <div className="absolute top-5 left-5 right-5 h-0.5 bg-slate-200 dark:bg-slate-700" />
        <motion.div
          className="absolute top-5 left-5 h-0.5 bg-indigo-500"
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(0, (currentStage / (stages.length - 1)) * 100)}%` }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
        />
        <div className="relative flex justify-between">
          {stages.map((stage, index) => {
            const isCompleted = index < currentStage;
            const isCurrent = index === currentStage;
            const isPending = index > currentStage;

            return (
              <div key={stage.id} className="flex flex-col items-center group">
                <motion.div
                  initial={false}
                  animate={{
                    scale: isCurrent ? 1.1 : 1,
                    backgroundColor: isCompleted || isCurrent
                      ? "rgb(99, 102, 241)"
                      : "rgb(255, 255, 255)",
                    borderColor: isCompleted || isCurrent
                      ? "rgb(99, 102, 241)"
                      : "rgb(226, 232, 240)",
                  }}
                  className={`
                    w-10 h-10 rounded-full border-2 flex items-center justify-center
                    transition-colors duration-300 z-10
                    ${isPending ? "bg-white dark:bg-slate-800" : ""}
                  `}
                >
                  {isCompleted ? (
                    <CheckCircle size={18} className="text-white" />
                  ) : isCurrent ? (
                    <span className="text-white">{stage.icon}</span>
                  ) : (
                    <span className="text-slate-300 dark:text-slate-600">{stage.icon}</span>
                  )}
                </motion.div>
                <div className="mt-3 flex flex-col items-center">
                  <span
                    className={`
                      text-sm font-medium transition-colors duration-300
                      ${isCurrent
                        ? "text-indigo-600 dark:text-indigo-400"
                        : isCompleted
                        ? "text-slate-900 dark:text-white"
                        : "text-slate-400"
                      }
                    `}
                  >
                    {stage.label}
                  </span>
                  <span className="text-xs text-slate-400 mt-0.5 max-w-[100px] text-center">
                    {stage.description}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-8">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-slate-500">Progress</span>
          <span className="font-medium text-slate-900 dark:text-white">
            {Math.round(progress)}%
          </span>
        </div>
        <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-start gap-3"
        >
          <AlertTriangle size={18} className="text-red-500 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-800 dark:text-red-200">
              Analysis Error
            </p>
            <p className="text-sm text-red-600 dark:text-red-300 mt-1">{error}</p>
          </div>
        </motion.div>
      )}
    </div>
  );
}
