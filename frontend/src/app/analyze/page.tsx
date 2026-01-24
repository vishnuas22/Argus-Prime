"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileVideo,
  FileAudio,
  FileText,
  Image,
  X,
  CheckCircle,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { useDropzone } from "react-dropzone";
import { useQueryClient } from "@tanstack/react-query";
import { analysisApi } from "@/lib/api";
import { useToast } from "@/components/ui/toast";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { FILE_LIMITS } from "@/lib/config";

const FILE_TYPE_ICONS = {
  video: FileVideo,
  audio: FileAudio,
  text: FileText,
  image: Image,
};

const STAGES = [
  { label: "Upload", description: "File uploaded successfully" },
  { label: "Extracting Frames", description: "Processing video frames" },
  { label: "Video Analysis", description: "Running video detection models" },
  { label: "Audio Analysis", description: "Running audio detection models" },
  { label: "Fusing Scores", description: "Combining multi-modal results" },
  { label: "Complete", description: "Analysis finished" },
];

export default function AnalyzePage() {
  const [step, setStep] = useState<"upload" | "processing">("upload");
  const [currentStage, setCurrentStage] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [requestId, setRequestId] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const { addToast } = useToast();
  const queryClient = useQueryClient();
  const pollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const stageMapping: Record<string, number> = {
    pending: 0,
    extracting_frames: 1,
    running_video_detection: 2,
    running_audio_detection: 2,
    fusing_scores: 3,
    generating_report: 4,
    completed: 5,
    failed: 6,
  };

  const pollProgress = useCallback(async (id: string) => {
    let attempts = 0;
    const maxAttempts = 600;

    const poll = async () => {
      try {
        const status = await analysisApi.getStatus(id);
        
        const newStage = stageMapping[status.status] ?? 0;
        const newProgress = status.progress ?? 0;

        setCurrentStage(newStage);
        setProgress(newProgress);

        if (status.status === "completed") {
          setIsAnalyzing(false);
          setStep("upload");
          queryClient.invalidateQueries({ queryKey: ["analyses"] });
          
          const result = await analysisApi.getResult(id);
          setAnalysisResult(result);
          addToast("success", "Analysis Complete", "Your results are ready");
          setSelectedFile(null);
          setRequestId(null);
          return;
        }

        if (status.status === "failed") {
          setIsAnalyzing(false);
          setStep("upload");
          setCurrentStage(0);
          setProgress(0);
          setSelectedFile(null);
          setRequestId(null);
          addToast("error", "Analysis Failed", status.errorMessage || "An error occurred");
          return;
        }

        attempts++;
        if (attempts < maxAttempts) {
          pollTimeoutRef.current = setTimeout(poll, 1500);
        } else {
          setIsAnalyzing(false);
          setStep("upload");
          addToast("error", "Timeout", "Analysis timed out. Please try again.");
        }
      } catch {
        attempts++;
        if (attempts < maxAttempts) {
          pollTimeoutRef.current = setTimeout(poll, 2000);
        } else {
          setIsAnalyzing(false);
          setStep("upload");
          addToast("error", "Connection Error", "Lost connection to server");
        }
      }
    };

    poll();
  }, [addToast, queryClient]);

  const handleUpload = useCallback(async (file: File) => {
    setSelectedFile(file);
    setStep("processing");
    setCurrentStage(1);
    setProgress(5);
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await analysisApi.submit({ file: formData } as any);
      const id = response.requestId || (response as any).request_id;
      setRequestId(id);
      
      setProgress(10);
      addToast("success", "Analysis Started", `Submitted ${file.name} for analysis`);
      
      pollProgress(id);
    } catch (error) {
      setIsAnalyzing(false);
      setStep("upload");
      setCurrentStage(0);
      setProgress(0);
      setSelectedFile(null);
      addToast("error", "Submission Failed", error instanceof Error ? error.message : "Could not submit file");
    }
  }, [addToast, pollProgress]);

  const onDrop = useCallback((files: File[]) => {
    if (files.length > 0) {
      const file = files[0];
      
      const fileType = Object.entries(FILE_LIMITS.acceptedTypes).find(([_, types]) =>
        types.includes(file.type)
      )?.[0];

      if (!fileType) {
        addToast("error", "Invalid File", "Please upload a supported file type");
        return;
      }

      if (file.size > FILE_LIMITS.maxSize) {
        addToast("error", "File Too Large", `Maximum file size is ${FILE_LIMITS.maxSize / (1024 * 1024)}MB`);
        return;
      }

      handleUpload(file);
    }
  }, [addToast, handleUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "video/*": [".mp4", ".avi", ".mov", ".webm"],
      "audio/*": [".wav", ".mp3", ".flac", ".m4a"],
      "image/*": [".jpg", ".jpeg", ".png", ".webp"],
      "text/*": [".txt", ".md", ".pdf"],
    },
    maxFiles: 1,
    multiple: false,
    disabled: isAnalyzing,
  });

  useEffect(() => {
    return () => {
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
    };
  }, []);

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Analyze Content
          </h1>
          <p className="text-slate-500 dark:text-slate-400">
            Upload video, audio, or text content for deepfake detection analysis
          </p>
        </div>

        {analysisResult && (
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-4 mb-4">
                <CheckCircle className="w-8 h-8 text-green-500" />
                <div>
                  <h3 className="text-lg font-semibold">Analysis Complete</h3>
                  <p className="text-slate-500">
                    Verdict: <span className={analysisResult.verdict === "manipulated" ? "text-red-500 font-medium" : "text-green-500 font-medium"}>{analysisResult.verdict}</span>
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <p className="text-2xl font-bold">{Math.round(analysisResult.overall_score * 100)}%</p>
                  <p className="text-sm text-slate-500">Manipulation Score</p>
                </div>
                <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <p className="text-2xl font-bold">{Math.round(analysisResult.confidence * 100)}%</p>
                  <p className="text-sm text-slate-500">Confidence</p>
                </div>
                <div className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <p className="text-2xl font-bold">{analysisResult.frames_analyzed || 0}</p>
                  <p className="text-sm text-slate-500">Frames Analyzed</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        <AnimatePresence mode="wait">
          {step === "upload" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-4"
            >
              <div
                {...getRootProps()}
                className={`
                  border-2 border-dashed rounded-xl p-12 text-center cursor-pointer
                  transition-all duration-200
                  ${isDragActive
                    ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20"
                    : "border-slate-200 dark:border-slate-700 hover:border-indigo-400"
                  }
                  ${isAnalyzing ? "opacity-50 cursor-not-allowed" : ""}
                `}
              >
                <input {...getInputProps()} />
                <div className="flex flex-col items-center gap-4">
                  <div className={`
                    p-4 rounded-full
                    ${isDragActive ? "bg-indigo-100 dark:bg-indigo-900/40" : "bg-slate-100 dark:bg-slate-800"}
                  `}>
                    <Upload className={`
                      w-8 h-8
                      ${isDragActive ? "text-indigo-600" : "text-slate-400"}
                    `} />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-slate-700 dark:text-slate-300">
                      {isDragActive
                        ? "Drop your file here"
                        : "Drag & drop a file, or click to browse"
                      }
                    </p>
                    <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                      Supports MP4, AVI, MOV, WAV, MP3, FLAC, TXT, PDF, PNG, JPG
                    </p>
                  </div>
                </div>
              </div>

              {selectedFile && step === "upload" && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex items-center gap-4 p-4 bg-slate-50 dark:bg-slate-800 rounded-xl"
                >
                  <div className="p-2 bg-indigo-100 dark:bg-indigo-900/40 rounded-lg">
                    <FileVideo className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-slate-900 dark:text-white truncate">
                      {selectedFile.name}
                    </p>
                    <p className="text-sm text-slate-500">
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedFile(null);
                    }}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </motion.div>
              )}
            </motion.div>
          )}

          {step === "processing" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-6"
            >
              <Card>
                <CardContent className="p-8">
                  <div className="flex flex-col items-center gap-6 text-center">
                    <div className="relative">
                      <div className="w-20 h-20 rounded-full bg-indigo-100 dark:bg-indigo-900/40 flex items-center justify-center">
                        {currentStage === STAGES.length - 1 ? (
                          <CheckCircle className="w-10 h-10 text-green-500" />
                        ) : (
                          <Loader2 className="w-10 h-10 text-indigo-600 animate-spin" />
                        )}
                      </div>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                        {STAGES[currentStage]?.label || "Processing..."}
                      </h3>
                      <p className="text-slate-500 dark:text-slate-400 mt-1">
                        {STAGES[currentStage]?.description || "Please wait"}
                      </p>
                    </div>

                    <div className="w-full max-w-md">
                      <Progress value={progress} className="h-2" />
                      <p className="text-sm text-slate-500 mt-2 text-center">
                        {progress}% complete
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {STAGES.map((stage, index) => (
                  <div
                    key={stage.label}
                    className={`
                      p-3 rounded-lg text-center text-sm
                      ${index < currentStage
                        ? "bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-400"
                        : index === currentStage
                        ? "bg-indigo-100 dark:bg-indigo-900/20 text-indigo-700 dark:text-indigo-400"
                        : "bg-slate-100 dark:bg-slate-800 text-slate-500"
                      }
                    `}
                  >
                    {stage.label}
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
