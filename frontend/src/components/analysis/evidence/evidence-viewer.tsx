"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Layers,
  Image,
  Music,
  MapPin,
  Info,
  Download,
  ZoomIn,
  ZoomOut,
  Maximize2,
  X,
} from "lucide-react";
import { Tabs, TabList, Tab, TabPanel } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { HeatmapViewer } from "@/components/analysis/heatmap";

interface EvidenceViewerProps {
  evidence: {
    heatmaps?: string[];
    spectrograms?: string[];
    localization?: {
      timestamps: Array<{ start: number; end: number; confidence: number }>;
      spatialRegions: Array<{ polygon: number[][]; confidence: number }>;
    };
    c2paMetadata?: Record<string, unknown>;
    adversarialCheck?: Record<string, unknown>;
  };
  className?: string;
}

export function EvidenceViewer({ evidence, className = "" }: EvidenceViewerProps) {
  const [activeTab, setActiveTab] = useState("heatmaps");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);

  const hasHeatmaps = evidence.heatmaps && evidence.heatmaps.length > 0;
  const hasSpectrograms = evidence.spectrograms && evidence.spectrograms.length > 0;
  const hasLocalization = evidence.localization && (
    evidence.localization.timestamps.length > 0 ||
    evidence.localization.spatialRegions.length > 0
  );
  const hasMetadata = evidence.c2paMetadata || evidence.adversarialCheck;

  if (!hasHeatmaps && !hasSpectrograms && !hasLocalization && !hasMetadata) {
    return (
      <Card className="p-6 text-center text-slate-500 dark:text-slate-400">
        No evidence available for this analysis.
      </Card>
    );
  }

  const tabs: Array<{ id: string; label: string; icon: React.ReactNode }> = [];

  if (hasHeatmaps) {
    tabs.push({ id: "heatmaps", label: "Heatmaps", icon: <Layers size={16} /> });
  }
  if (hasSpectrograms) {
    tabs.push({ id: "spectrograms", label: "Spectrograms", icon: <Image size={16} /> });
  }
  if (hasLocalization) {
    tabs.push({ id: "localization", label: "Localization", icon: <MapPin size={16} /> });
  }
  if (hasMetadata) {
    tabs.push({ id: "metadata", label: "Metadata", icon: <Info size={16} /> });
  }

  const defaultTab = tabs[0]?.id || "heatmaps";

  return (
    <div className={className}>
      <Tabs defaultValue={defaultTab}>
        <TabList className="mb-4">
          {tabs.map((tab) => (
            <Tab key={tab.id} value={tab.id} leftIcon={tab.icon}>
              {tab.label}
            </Tab>
          ))}
        </TabList>

        <AnimatePresence mode="wait">
          {(activeTab === "heatmaps" || defaultTab === "heatmaps") && hasHeatmaps && (
            <TabPanel value="heatmaps">
              <motion.div
                key="heatmaps"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <HeatmapGrid heatmaps={evidence.heatmaps || []} onSelect={setSelectedImage} />
              </motion.div>
            </TabPanel>
          )}

          {activeTab === "spectrograms" && hasSpectrograms && (
            <TabPanel value="spectrograms">
              <motion.div
                key="spectrograms"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <SpectrogramGrid spectrograms={evidence.spectrograms || []} onSelect={setSelectedImage} />
              </motion.div>
            </TabPanel>
          )}

          {activeTab === "localization" && hasLocalization && (
            <TabPanel value="localization">
              <motion.div
                key="localization"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <LocalizationPanel
                  timestamps={evidence.localization?.timestamps || []}
                  spatialRegions={evidence.localization?.spatialRegions || []}
                />
              </motion.div>
            </TabPanel>
          )}

          {activeTab === "metadata" && hasMetadata && (
            <TabPanel value="metadata">
              <motion.div
                key="metadata"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <MetadataPanel
                  c2paMetadata={evidence.c2paMetadata}
                  adversarialCheck={evidence.adversarialCheck}
                />
              </motion.div>
            </TabPanel>
          )}
        </AnimatePresence>
      </Tabs>

      {selectedImage && (
        <ImageModal
          src={selectedImage}
          onClose={() => setSelectedImage(null)}
          zoom={zoom}
          onZoomChange={setZoom}
        />
      )}
    </div>
  );
}

interface HeatmapGridProps {
  heatmaps: string[];
  onSelect: (src: string) => void;
}

function HeatmapGrid({ heatmaps, onSelect }: HeatmapGridProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      {heatmaps.map((heatmap, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: index * 0.1 }}
          className="relative aspect-video rounded-lg overflow-hidden cursor-pointer group"
          onClick={() => onSelect(heatmap)}
        >
          <img
            src={heatmap}
            alt={`Heatmap ${index + 1}`}
            className="w-full h-full object-cover transition-transform group-hover:scale-105"
          />
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
            <ZoomIn className="text-white opacity-0 group-hover:opacity-100 transition-opacity" size={24} />
          </div>
          <div className="absolute bottom-2 left-2 right-2 flex justify-between items-center">
            <Badge variant="default" className="bg-black/50 backdrop-blur-sm">
              Frame {index + 1}
            </Badge>
            <Badge
              variant={
                index < 3
                  ? "danger"
                  : index < 6
                  ? "warning"
                  : "success"
              }
              className="bg-black/50 backdrop-blur-sm"
            >
              {index < 3 ? "High Risk" : index < 6 ? "Medium Risk" : "Low Risk"}
            </Badge>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

interface SpectrogramGridProps {
  spectrograms: string[];
  onSelect: (src: string) => void;
}

function SpectrogramGrid({ spectrograms, onSelect }: SpectrogramGridProps) {
  return (
    <div className="space-y-4">
      {spectrograms.map((spectrogram, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="relative rounded-lg overflow-hidden cursor-pointer group"
          onClick={() => onSelect(spectrogram)}
        >
          <img
            src={spectrogram}
            alt={`Spectrogram ${index + 1}`}
            className="w-full h-auto transition-transform group-hover:scale-[1.02]"
          />
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
            <ZoomIn className="text-white opacity-0 group-hover:opacity-100 transition-opacity" size={24} />
          </div>
          <div className="absolute top-2 left-2">
            <Badge variant="default" className="bg-black/50 backdrop-blur-sm">
              Segment {index + 1}
            </Badge>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

interface LocalizationPanelProps {
  timestamps: Array<{ start: number; end: number; confidence: number }>;
  spatialRegions: Array<{ polygon: number[][]; confidence: number }>;
}

function LocalizationPanel({ timestamps, spatialRegions }: LocalizationPanelProps) {
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}.${ms.toString().padStart(3, "0")}`;
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return "text-red-600 dark:text-red-400";
    if (confidence >= 0.5) return "text-amber-600 dark:text-amber-400";
    return "text-emerald-600 dark:text-emerald-400";
  };

  return (
    <div className="space-y-6">
      {timestamps.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
            Temporal Regions
          </h4>
          <div className="space-y-2">
            {timestamps.map((region, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="flex items-center gap-4 p-3 bg-slate-50 dark:bg-slate-800 rounded-lg"
              >
                <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400 font-mono text-sm">
                  <span>{formatTime(region.start)}</span>
                  <span>-</span>
                  <span>{formatTime(region.end)}</span>
                </div>
                <div className="flex-1 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${region.confidence * 100}%` }}
                    transition={{ duration: 0.5, delay: 0.2 + index * 0.05 }}
                    className={`h-full rounded-full ${
                      region.confidence >= 0.8
                        ? "bg-red-500"
                        : region.confidence >= 0.5
                        ? "bg-amber-500"
                        : "bg-emerald-500"
                    }`}
                  />
                </div>
                <span className={`text-sm font-medium ${getConfidenceColor(region.confidence)}`}>
                  {Math.round(region.confidence * 100)}%
                </span>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {spatialRegions.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
            Spatial Regions ({spatialRegions.length})
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {spatialRegions.map((region, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                className="aspect-square bg-slate-100 dark:bg-slate-800 rounded-lg flex items-center justify-center relative overflow-hidden"
              >
                <div
                  className="absolute inset-0 border-2 border-red-500/50"
                  style={{
                    clipPath: "polygon(0 0, 100% 0, 100% 25%, 75% 25%, 75% 50%, 100% 50%, 100% 100%, 0 100%, 0 75%, 25% 75%, 25% 50%, 0 50%)",
                  }}
                />
                <Badge
                  variant={
                    region.confidence >= 0.8
                      ? "danger"
                      : region.confidence >= 0.5
                      ? "warning"
                      : "success"
                  }
                >
                  {Math.round(region.confidence * 100)}% confidence
                </Badge>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface MetadataPanelProps {
  c2paMetadata?: Record<string, unknown>;
  adversarialCheck?: Record<string, unknown>;
}

function MetadataPanel({ c2paMetadata, adversarialCheck }: MetadataPanelProps) {
  if (!c2paMetadata && !adversarialCheck) {
    return (
      <div className="text-center text-slate-500 dark:text-slate-400 py-8">
        No metadata available
      </div>
    );
  }

  const formatValue = (value: unknown): string => {
    if (typeof value === "object") {
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  };

  return (
    <div className="space-y-6">
      {c2paMetadata && (
        <div>
          <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
            <Badge variant="info">C2PA</Badge>
            Provenance Metadata
          </h4>
          <Card className="p-4 bg-slate-50 dark:bg-slate-800/50">
            <pre className="text-xs text-slate-600 dark:text-slate-400 overflow-auto max-h-64">
              {formatValue(c2paMetadata)}
            </pre>
          </Card>
        </div>
      )}

      {adversarialCheck && (
        <div>
          <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3 flex items-center gap-2">
            <Badge variant="warning">Adversarial</Badge>
            Defense Analysis
          </h4>
          <Card className="p-4 bg-slate-50 dark:bg-slate-800/50">
            <pre className="text-xs text-slate-600 dark:text-slate-400 overflow-auto max-h-64">
              {formatValue(adversarialCheck)}
            </pre>
          </Card>
        </div>
      )}
    </div>
  );
}

interface ImageModalProps {
  src: string;
  onClose: () => void;
  zoom: number;
  onZoomChange: (zoom: number) => void;
}

function ImageModal({ src, onClose, zoom, onZoomChange }: ImageModalProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
      onClick={onClose}
    >
      <div className="absolute top-4 right-4 flex items-center gap-2 z-10">
        <Button
          variant="ghost"
          size="sm"
          onClick={(e) => {
            e.stopPropagation();
            onZoomChange(Math.max(0.5, zoom - 0.25));
          }}
          leftIcon={<ZoomOut size={16} />}
          className="text-white hover:bg-white/10"
        >
          Zoom Out
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={(e) => {
            e.stopPropagation();
            onZoomChange(Math.min(3, zoom + 0.25));
          }}
          leftIcon={<ZoomIn size={16} />}
          className="text-white hover:bg-white/10"
        >
          Zoom In
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClose}
          className="text-white hover:bg-white/10"
        >
          <X size={20} />
        </Button>
      </div>

      <motion.div
        initial={{ scale: 0.9 }}
        animate={{ scale: zoom }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className="max-w-[90vw] max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <img src={src} alt="Evidence" className="max-w-full max-h-[85vh] object-contain" />
      </motion.div>

      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-white/60 text-sm">
        {(zoom * 100).toFixed(0)}% - Press ESC to close
      </div>
    </motion.div>
  );
}
