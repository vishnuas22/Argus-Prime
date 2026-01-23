"use client";

import React from 'react';
import { motion } from 'framer-motion';
import { FileText, Type, Globe, BarChart2 } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { TextResult, SentenceAnalysis } from '@/types';

interface TextFindingsProps {
  findings: TextResult;
}

export function TextFindings({ findings }: TextFindingsProps) {
  const getScoreColor = (score: number) => {
    if (score < 0.4) return 'text-emerald-500 bg-emerald-50 dark:bg-emerald-900/20';
    if (score < 0.6) return 'text-amber-500 bg-amber-50 dark:bg-amber-900/20';
    return 'text-red-500 bg-red-50 dark:bg-red-900/20';
  };

  const formatScore = (score: number) => (score * 100).toFixed(1);

  const getManipulationLevel = (score: number): 'low' | 'medium' | 'high' => {
    if (score < 0.4) return 'low';
    if (score < 0.6) return 'medium';
    return 'high';
  };

  const highlightSentence = (sentence: string, score: number): React.ReactNode => {
    const level = getManipulationLevel(score);
    const bgColors = {
      low: 'bg-emerald-100 dark:bg-emerald-900/30',
      medium: 'bg-amber-100 dark:bg-amber-900/30',
      high: 'bg-red-100 dark:bg-red-900/30',
    };

    return (
      <span className={`${bgColors[level]} px-1 rounded`}>
        {sentence}
      </span>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
              <FileText size={20} className="text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">Word Count</p>
              <p className="text-lg font-semibold text-slate-900 dark:text-white">
                {findings.wordCount}
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
              <Globe size={20} className="text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">Language</p>
              <p className="text-lg font-semibold text-slate-900 dark:text-white uppercase">
                {findings.languageDetected}
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-amber-100 dark:bg-amber-900/30 rounded-lg flex items-center justify-center">
              <Type size={20} className="text-amber-600 dark:text-amber-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">Overall Score</p>
              <p className={`text-lg font-semibold ${getScoreColor(findings.overallScore)}`}>
                {formatScore(findings.overallScore)}%
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg flex items-center justify-center">
              <BarChart2 size={20} className="text-emerald-600 dark:text-emerald-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">Sentences</p>
              <p className="text-lg font-semibold text-slate-900 dark:text-white">
                {findings.perSentenceScores.length}
              </p>
            </div>
          </div>
        </Card>
      </div>

      {Object.keys(findings.perplexityScores).length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Perplexity Scores
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(findings.perplexityScores).map(([model, score]) => (
              <div key={model} className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                <p className="text-xs text-slate-500 dark:text-slate-400 capitalize">
                  {model.replace(/([A-Z])/g, ' $1').trim()}
                </p>
                <p className="text-lg font-semibold text-slate-900 dark:text-white">
                  {score.toFixed(2)}
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {Object.keys(findings.burstinessMetrics).length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Burstiness Metrics
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(findings.burstinessMetrics).map(([metric, value]) => (
              <div key={metric} className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                <p className="text-xs text-slate-500 dark:text-slate-400 capitalize">
                  {metric.replace(/([A-Z])/g, ' $1').trim()}
                </p>
                <p className="text-lg font-semibold text-slate-900 dark:text-white">
                  {typeof value === 'number' ? value.toFixed(4) : String(value)}
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {findings.perSentenceScores.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Sentence Analysis
          </h3>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {findings.perSentenceScores.slice(0, 20).map((sentence: SentenceAnalysis, index: number) => (
              <motion.div
                key={`${sentence.startChar}-${sentence.endChar}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.03 }}
                className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg"
              >
                <div className="flex items-start justify-between gap-4">
                  <p className="text-sm text-slate-700 dark:text-slate-300 flex-1">
                    {highlightSentence(sentence.sentence, sentence.manipulationScore)}
                  </p>
                  <div className="flex flex-col items-end gap-1">
                    <Badge className={getScoreColor(sentence.manipulationScore)}>
                      {formatScore(sentence.manipulationScore)}%
                    </Badge>
                    <span className="text-xs text-slate-500 dark:text-slate-400">
                      PPL: {sentence.perplexity.toFixed(1)} | BUR: {sentence.burstiness.toFixed(2)}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
          {findings.perSentenceScores.length > 20 && (
            <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 text-center">
              Showing 20 of {findings.perSentenceScores.length} sentences
            </p>
          )}
        </Card>
      )}
    </motion.div>
  );
}
