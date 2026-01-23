"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  Video,
  Music,
  FileText,
  Settings,
  ChevronDown,
  Check,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { VideoOptions, AudioOptions, TextOptions, AnalysisOptions } from "@/types";

interface AnalysisOptionsSelectorProps {
  options: AnalysisOptions;
  onChange: (options: AnalysisOptions) => void;
  fileType: "video" | "audio" | "image" | "text";
  className?: string;
}

export function AnalysisOptionsSelector({
  options,
  onChange,
  fileType,
  className = "",
}: AnalysisOptionsSelectorProps) {
  const updateVideoOptions = (updates: Partial<VideoOptions>) => {
    onChange({
      ...options,
      video: { ...options.video, ...updates },
    });
  };

  const updateAudioOptions = (updates: Partial<AudioOptions>) => {
    onChange({
      ...options,
      audio: { ...options.audio, ...updates },
    });
  };

  const updateTextOptions = (updates: Partial<TextOptions>) => {
    onChange({
      ...options,
      text: { ...options.text, ...updates },
    });
  };

  const resetOptions = () => {
    onChange({
      video: {
        fps: 5,
        extractFaces: true,
        generateHeatmaps: true,
        rppgAnalysis: true,
      },
      audio: {
        sampleRate: 16000,
        segmentDuration: 2,
        applyVad: true,
      },
      text: {
        language: undefined,
        includePerplexity: true,
        includeBurstiness: true,
      },
    });
  };

  return (
    <Card className={`p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Settings size={20} className="text-indigo-600" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
            Analysis Options
          </h3>
        </div>
        <Button variant="ghost" size="sm" onClick={resetOptions}>
          Reset to Defaults
        </Button>
      </div>

      <div className="space-y-6">
        {(fileType === "video" || fileType === "image") && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-slate-700 dark:text-slate-300">
              <Video size={18} />
              <h4 className="font-medium">Video Analysis</h4>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pl-8">
              <div>
                <label className="block text-sm text-slate-500 dark:text-slate-400 mb-1">
                  Frames per Second
                </label>
                <Select
                  options={[
                    { value: "1", label: "1 FPS" },
                    { value: "5", label: "5 FPS (Default)" },
                    { value: "10", label: "10 FPS" },
                    { value: "15", label: "15 FPS" },
                    { value: "30", label: "30 FPS" },
                  ]}
                  value={String(options.video?.fps || 5)}
                  onChange={(value) => updateVideoOptions({ fps: Number(value) })}
                />
              </div>

              <div className="space-y-3">
                <label className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      Extract Faces
                    </p>
                    <p className="text-xs text-slate-500">Detect and analyze faces</p>
                  </div>
                  <Switch
                    checked={options.video?.extractFaces ?? true}
                    onChange={(checked) => updateVideoOptions({ extractFaces: checked })}
                  />
                </label>

                <label className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      Generate Heatmaps
                    </p>
                    <p className="text-xs text-slate-500">Create attention heatmaps</p>
                  </div>
                  <Switch
                    checked={options.video?.generateHeatmaps ?? true}
                    onChange={(checked) => updateVideoOptions({ generateHeatmaps: checked })}
                  />
                </label>

                <label className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      rPPG Analysis
                    </p>
                    <p className="text-xs text-slate-500">Remote photoplethysmography</p>
                  </div>
                  <Switch
                    checked={options.video?.rppgAnalysis ?? true}
                    onChange={(checked) => updateVideoOptions({ rppgAnalysis: checked })}
                  />
                </label>
              </div>
            </div>
          </div>
        )}

        {fileType === "audio" && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-slate-700 dark:text-slate-300">
              <Music size={18} />
              <h4 className="font-medium">Audio Analysis</h4>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pl-8">
              <div>
                <label className="block text-sm text-slate-500 dark:text-slate-400 mb-1">
                  Sample Rate
                </label>
                <Select
                  options={[
                    { value: "8000", label: "8 kHz" },
                    { value: "16000", label: "16 kHz (Default)" },
                    { value: "22050", label: "22.05 kHz" },
                    { value: "44100", label: "44.1 kHz" },
                    { value: "48000", label: "48 kHz" },
                  ]}
                  value={String(options.audio?.sampleRate || 16000)}
                  onChange={(value) => updateAudioOptions({ sampleRate: Number(value) })}
                />
              </div>

              <div>
                <label className="block text-sm text-slate-500 dark:text-slate-400 mb-1">
                  Segment Duration (seconds)
                </label>
                <Select
                  options={[
                    { value: "1", label: "1 second" },
                    { value: "2", label: "2 seconds (Default)" },
                    { value: "3", label: "3 seconds" },
                    { value: "5", label: "5 seconds" },
                    { value: "10", label: "10 seconds" },
                  ]}
                  value={String(options.audio?.segmentDuration || 2)}
                  onChange={(value) => updateAudioOptions({ segmentDuration: Number(value) })}
                />
              </div>

              <div className="md:col-span-2">
                <label className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      Voice Activity Detection
                    </p>
                    <p className="text-xs text-slate-500">Remove silent segments</p>
                  </div>
                  <Switch
                    checked={options.audio?.applyVad ?? true}
                    onChange={(checked) => updateAudioOptions({ applyVad: checked })}
                  />
                </label>
              </div>
            </div>
          </div>
        )}

        {fileType === "text" && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-slate-700 dark:text-slate-300">
              <FileText size={18} />
              <h4 className="font-medium">Text Analysis</h4>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pl-8">
              <div>
                <label className="block text-sm text-slate-500 dark:text-slate-400 mb-1">
                  Language (Auto-detect if empty)
                </label>
                <Input
                  placeholder="e.g., en, es, fr"
                  value={options.text?.language || ""}
                  onChange={(e) => updateTextOptions({ language: e.target.value || undefined })}
                />
              </div>

              <div className="space-y-3">
                <label className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      Perplexity Analysis
                    </p>
                    <p className="text-xs text-slate-500">Measure text randomness</p>
                  </div>
                  <Switch
                    checked={options.text?.includePerplexity ?? true}
                    onChange={(checked) => updateTextOptions({ includePerplexity: checked })}
                  />
                </label>

                <label className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                      Burstiness Analysis
                    </p>
                    <p className="text-xs text-slate-500">Detect writing patterns</p>
                  </div>
                  <Switch
                    checked={options.text?.includeBurstiness ?? true}
                    onChange={(checked) => updateTextOptions({ includeBurstiness: checked })}
                  />
                </label>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
