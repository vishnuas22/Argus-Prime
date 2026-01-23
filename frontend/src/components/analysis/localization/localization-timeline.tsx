"use client";

import React from 'react';
import { motion } from 'framer-motion';
import { Clock, AlertTriangle, MapPin } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { Localization, LocalizationEntry, SpatialRegion } from '@/types';

interface LocalizationTimelineProps {
  localization: Localization;
  duration?: number;
  title?: string;
}

export function LocalizationTimeline({ localization, duration, title = 'Manipulation Timeline' }: LocalizationTimelineProps) {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-red-500';
    if (confidence >= 0.6) return 'bg-amber-500';
    return 'bg-yellow-500';
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
  };

  return (
    <Card className="p-6">
      <div className="flex items-center gap-2 mb-4">
        <Clock size={20} className="text-indigo-600 dark:text-indigo-400" />
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
          {title}
        </h3>
      </div>

      {localization.timestamps.length === 0 && localization.spatialRegions.length === 0 ? (
        <div className="text-center py-8">
          <MapPin size={40} className="mx-auto text-slate-300 dark:text-slate-600 mb-3" />
          <p className="text-slate-500 dark:text-slate-400">
            No manipulation regions detected
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {duration && (
            <div className="relative mb-6">
              <div className="absolute top-0 left-0 right-0 h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-indigo-500 to-purple-500"
                  initial={{ width: 0 }}
                  animate={{ width: '100%' }}
                  transition={{ duration: 1 }}
                />
              </div>
              <div className="flex justify-between text-xs text-slate-400 mt-2">
                <span>0:00</span>
                <span>{formatTime(duration)}</span>
              </div>
            </div>
          )}

          {localization.timestamps.map((entry: LocalizationEntry, index: number) => (
            <motion.div
              key={`timestamp-${index}`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-start gap-4 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg"
            >
              <div className="flex flex-col items-center">
                <div className={`w-3 h-3 rounded-full ${getConfidenceColor(entry.confidence)}`} />
                {index < localization.timestamps.length - 1 && (
                  <div className="w-0.5 h-full bg-slate-200 dark:bg-slate-700 mt-1" />
                )}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Clock size={14} className="text-slate-400" />
                    <span className="text-sm font-mono text-slate-700 dark:text-slate-300">
                      {formatTime(entry.start)} - {formatTime(entry.end)}
                    </span>
                  </div>
                  <Badge
                    className={
                      entry.confidence >= 0.8
                        ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                        : entry.confidence >= 0.6
                        ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
                        : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                    }
                  >
                    {Math.round(entry.confidence * 100)}% confidence
                  </Badge>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Duration: {(entry.end - entry.start).toFixed(2)}s
                </p>
              </div>
            </motion.div>
          ))}

          {localization.spatialRegions.length > 0 && (
            <div className="mt-6">
              <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
                <MapPin size={14} />
                Spatial Regions ({localization.spatialRegions.length})
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {localization.spatialRegions.map((region: SpatialRegion, index: number) => (
                  <motion.div
                    key={`spatial-${index}`}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="p-3 border border-slate-200 dark:border-slate-700 rounded-lg"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-slate-500 dark:text-slate-400">
                        Region {index + 1}
                      </span>
                      <Badge variant="outline">
                        {Math.round(region.confidence * 100)}%
                      </Badge>
                    </div>
                    <div className="text-xs font-mono text-slate-600 dark:text-slate-400">
                      {region.polygon.length} points
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
