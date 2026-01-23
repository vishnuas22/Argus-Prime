"use client";

import React from 'react';
import { motion } from 'framer-motion';
import { Music, Mic, Radio, AudioWaveform } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { AudioResult, SegmentAnalysis } from '@/types';

interface AudioFindingsProps {
  findings: AudioResult;
}

export function AudioFindings({ findings }: AudioFindingsProps) {
  const getScoreColor = (score: number) => {
    if (score < 0.4) return 'text-emerald-500 bg-emerald-50 dark:bg-emerald-900/20';
    if (score < 0.6) return 'text-amber-500 bg-amber-50 dark:bg-amber-900/20';
    return 'text-red-500 bg-red-50 dark:bg-red-900/20';
  };

  const formatScore = (score: number) => (score * 100).toFixed(1);
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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
              <Music size={20} className="text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">Duration</p>
              <p className="text-lg font-semibold text-slate-900 dark:text-white">
                {formatTime(findings.durationSeconds)}
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
              <AudioWaveform size={20} className="text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">Segments</p>
              <p className="text-lg font-semibold text-slate-900 dark:text-white">
                {findings.segmentsAnalyzed}
              </p>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-amber-100 dark:bg-amber-900/30 rounded-lg flex items-center justify-center">
              <Radio size={20} className="text-amber-600 dark:text-amber-400" />
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
              <Mic size={20} className="text-emerald-600 dark:text-emerald-400" />
            </div>
            <div>
              <p className="text-xs text-slate-500 dark:text-slate-400">Vocoder</p>
              <p className="text-lg font-semibold text-slate-900 dark:text-white">
                {findings.vocoderDetected || 'None'}
              </p>
            </div>
          </div>
        </Card>
      </div>

      {Object.entries(findings.artifactEvidence).length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Artifact Evidence
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(findings.artifactEvidence).map(([artifact, score]) => (
              <div key={artifact} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium text-slate-600 dark:text-slate-300 capitalize">
                    {artifact.replace(/([A-Z])/g, ' $1').trim()}
                  </span>
                  <span className={`text-sm font-semibold ${getScoreColor(score)}`}>
                    {formatScore(score)}%
                  </span>
                </div>
                <div className="h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${getScoreColor(score)}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${score * 100}%` }}
                    transition={{ duration: 0.5, ease: 'easeOut' }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {findings.perSegmentScores.length > 0 && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Per-Segment Analysis
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <th className="text-left py-2 px-3 font-medium text-slate-500 dark:text-slate-400">
                    Start
                  </th>
                  <th className="text-left py-2 px-3 font-medium text-slate-500 dark:text-slate-400">
                    End
                  </th>
                  <th className="text-left py-2 px-3 font-medium text-slate-500 dark:text-slate-400">
                    Vocoder
                  </th>
                  <th className="text-left py-2 px-3 font-medium text-slate-500 dark:text-slate-400">
                    Score
                  </th>
                </tr>
              </thead>
              <tbody>
                {findings.perSegmentScores.slice(0, 10).map((segment: SegmentAnalysis, index: number) => (
                  <motion.tr
                    key={`${segment.startTime}-${segment.endTime}`}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50"
                  >
                    <td className="py-2 px-3 text-slate-900 dark:text-white">
                      {formatTime(segment.startTime)}
                    </td>
                    <td className="py-2 px-3 text-slate-500 dark:text-slate-400">
                      {formatTime(segment.endTime)}
                    </td>
                    <td className="py-2 px-3 text-slate-500 dark:text-slate-400">
                      {segment.vocoderClass ? (
                        <Badge variant="outline">{segment.vocoderClass}</Badge>
                      ) : (
                        <span className="text-slate-400">-</span>
                      )}
                    </td>
                    <td className="py-2 px-3">
                      <Badge className={getScoreColor(segment.manipulationScore)}>
                        {formatScore(segment.manipulationScore)}%
                      </Badge>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
            {findings.perSegmentScores.length > 10 && (
              <p className="mt-4 text-sm text-slate-500 dark:text-slate-400 text-center">
                Showing 10 of {findings.perSegmentScores.length} segments
              </p>
            )}
          </div>
        </Card>
      )}
    </motion.div>
  );
}
