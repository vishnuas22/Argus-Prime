"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ZoomIn, ZoomOut, RotateCcw, Image as ImageIcon, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Select } from '@/components/ui/select';

interface HeatmapViewerProps {
  heatmaps: string[];
  title?: string;
  onDownload?: (heatmapUrl: string) => void;
}

export function HeatmapViewer({ heatmaps, title = 'Attention Heatmaps', onDownload }: HeatmapViewerProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [opacity, setOpacity] = useState(0.7);

  if (heatmaps.length === 0) {
    return (
      <Card className="p-6 text-center">
        <ImageIcon size={48} className="mx-auto text-slate-300 dark:text-slate-600 mb-4" />
        <p className="text-slate-500 dark:text-slate-400">
          No heatmaps available for this analysis
        </p>
      </Card>
    );
  }

  const currentHeatmap = heatmaps[selectedIndex];

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 3));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.25, 0.5));
  const handleReset = () => {
    setZoom(1);
    setOpacity(0.7);
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
          {title}
        </h3>
        <div className="flex items-center gap-2">
          {heatmaps.length > 1 && (
            <Select
              options={heatmaps.map((_, index) => ({
                value: String(index),
                label: `Heatmap ${index + 1}`,
              }))}
              value={String(selectedIndex)}
              onChange={(value) => setSelectedIndex(Number(value))}
              className="w-36"
            />
          )}
        </div>
      </div>

      <div className="relative bg-slate-100 dark:bg-slate-900 rounded-lg overflow-hidden">
        <div
          className="relative overflow-auto max-h-[500px] flex items-center justify-center"
          style={{ minHeight: '300px' }}
        >
          <motion.div
            className="relative"
            style={{
              transform: `scale(${zoom})`,
              transition: 'transform 0.2s ease-out',
            }}
          >
            <img
              src={currentHeatmap}
              alt={`Heatmap ${selectedIndex + 1}`}
              className="max-w-full h-auto rounded-lg"
              style={{ opacity }}
              onError={(e) => {
                (e.target as HTMLImageElement).src = '/placeholder-heatmap.png';
              }}
            />
            <motion.div
              className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent pointer-events-none rounded-lg"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            />
          </motion.div>
        </div>

        <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-lg p-2">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={handleZoomOut}>
              <ZoomOut size={16} />
            </Button>
            <span className="text-sm text-slate-600 dark:text-slate-300 w-16 text-center">
              {Math.round(zoom * 100)}%
            </span>
            <Button variant="ghost" size="sm" onClick={handleZoomIn}>
              <ZoomIn size={16} />
            </Button>
            <Button variant="ghost" size="sm" onClick={handleReset}>
              <RotateCcw size={16} />
            </Button>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-500 dark:text-slate-400">Opacity:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="w-24"
            />
          </div>

          {onDownload && (
            <Button variant="ghost" size="sm" onClick={() => onDownload(currentHeatmap)}>
              <Download size={16} />
            </Button>
          )}
        </div>
      </div>

      {heatmaps.length > 1 && (
        <div className="mt-4 flex gap-2 overflow-x-auto pb-2">
          {heatmaps.map((heatmap, index) => (
            <button
              key={index}
              onClick={() => setSelectedIndex(index)}
              className={`flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden border-2 transition-colors ${
                index === selectedIndex
                  ? 'border-indigo-500'
                  : 'border-transparent hover:border-slate-300'
              }`}
            >
              <img
                src={heatmap}
                alt={`Thumbnail ${index + 1}`}
                className="w-full h-full object-cover"
              />
            </button>
          ))}
        </div>
      )}
    </Card>
  );
}
