"use client";

import React from "react";
import { motion } from "framer-motion";
import { Shield, AlertTriangle, CheckCircle, Info } from "lucide-react";

interface TrustScoreDisplayProps {
  score: number;
  maxScore?: number;
  size?: "sm" | "md" | "lg";
  showDetails?: boolean;
  breakdown?: {
    authenticity: number;
    credibility: number;
    manipulation: number;
    sentiment: number;
  };
}

const sizeConfig = {
  sm: { svg: 80, stroke: 6, text: "text-lg" },
  md: { svg: 140, stroke: 8, text: "text-3xl" },
  lg: { svg: 200, stroke: 10, text: "text-5xl" },
};

const getScoreColor = (score: number) => {
  if (score >= 80) return { color: "#10b981", bg: "bg-emerald-50 dark:bg-emerald-900/20" };
  if (score >= 60) return { color: "#f59e0b", bg: "bg-amber-50 dark:bg-amber-900/20" };
  if (score >= 40) return { color: "#f97316", bg: "bg-orange-50 dark:bg-orange-900/20" };
  return { color: "#ef4444", bg: "bg-red-50 dark:bg-red-900/20" };
};

const getScoreLabel = (score: number) => {
  if (score >= 80) return { label: "High Trust", icon: CheckCircle };
  if (score >= 60) return { label: "Moderate Trust", icon: Info };
  if (score >= 40) return { label: "Low Trust", icon: AlertTriangle };
  return { label: "Untrustworthy", icon: AlertTriangle };
};

export function TrustScoreDisplay({
  score,
  maxScore = 100,
  size = "md",
  showDetails = true,
  breakdown,
}: TrustScoreDisplayProps) {
  const config = sizeConfig[size];
  const percentage = (score / maxScore) * 100;
  const colorConfig = getScoreColor(percentage);
  const scoreLabel = getScoreLabel(percentage);
  const ScoreIcon = scoreLabel.icon;

  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative">
        <svg
          width={config.svg}
          height={config.svg}
          viewBox="0 0 100 100"
          className="transform -rotate-90"
        >
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="currentColor"
            strokeWidth={config.stroke}
            className="text-slate-100 dark:text-slate-700"
          />
          <motion.circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke={colorConfig.color}
            strokeWidth={config.stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1.5, ease: "easeOut" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5, duration: 0.3 }}
            className={`${config.text} font-bold text-slate-900 dark:text-white`}
          >
            {Math.round(score)}
          </motion.span>
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.7 }}
            className="text-xs text-slate-400"
          >
            /{maxScore}
          </motion.span>
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className={`mt-4 flex items-center gap-2 px-4 py-2 rounded-full ${colorConfig.bg}`}
      >
        <ScoreIcon size={16} style={{ color: colorConfig.color }} />
        <span className="text-sm font-medium" style={{ color: colorConfig.color }}>
          {scoreLabel.label}
        </span>
      </motion.div>

      {showDetails && breakdown && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-6 grid grid-cols-2 gap-3 w-full"
        >
          <ScoreBreakdownItem label="Authenticity" score={breakdown.authenticity} />
          <ScoreBreakdownItem label="Credibility" score={breakdown.credibility} />
          <ScoreBreakdownItem label="Manipulation" score={breakdown.manipulation} />
          <ScoreBreakdownItem label="Sentiment" score={breakdown.sentiment} />
        </motion.div>
      )}
    </div>
  );
}

interface ScoreBreakdownItemProps {
  label: string;
  score: number;
}

function ScoreBreakdownItem({ label, score }: ScoreBreakdownItemProps) {
  const percentage = (score / 100) * 100;
  const colorConfig = getScoreColor(percentage);

  return (
    <div className="p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-100 dark:border-slate-700">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs text-slate-500 dark:text-slate-400">{label}</span>
        <span className="text-sm font-medium text-slate-900 dark:text-white">{score}</span>
      </div>
      <div className="h-1.5 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="h-full rounded-full"
          style={{ backgroundColor: colorConfig.color }}
        />
      </div>
    </div>
  );
}
