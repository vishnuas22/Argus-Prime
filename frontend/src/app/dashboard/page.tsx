"use client";

import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  FileVideo,
  FileAudio,
  FileText,
  Image,
  Clock,
  CheckCircle,
  AlertCircle,
  BarChart3,
  Shield,
  Zap,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { analysisApi, healthApi } from "@/lib/api";
import { useToast } from "@/components/ui/toast";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

export default function DashboardPage() {
  const { addToast } = useToast();

  const { data: analysesData, isLoading } = useQuery({
    queryKey: ["analyses"],
    queryFn: () => analysisApi.listAnalyses({ limit: 5 }),
  });

  const { data: healthData } = useQuery({
    queryKey: ["health"],
    queryFn: () => healthApi.basic(),
    refetchInterval: 30000,
  });

  const stats = {
    total: analysesData?.total || 0,
    pending: analysesData?.items?.filter((a: any) => a.status === "pending").length || 0,
    completed: analysesData?.items?.filter((a: any) => a.status === "completed").length || 0,
    failed: analysesData?.items?.filter((a: any) => a.status === "failed").length || 0,
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
              Dashboard
            </h1>
            <p className="text-slate-500 dark:text-slate-400 mt-1">
              Monitor your deepfake detection activity
            </p>
          </div>
          <Link href="/analyze">
            <Button className="bg-indigo-600 hover:bg-indigo-700">
              <Zap className="w-4 h-4 mr-2" />
              New Analysis
            </Button>
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="bg-gradient-to-br from-indigo-500 to-purple-600 text-white border-0">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-indigo-100">Total Analyses</p>
                  <p className="text-4xl font-bold mt-2">
                    {isLoading ? <Skeleton className="h-10 w-20 bg-white/20" /> : stats.total}
                  </p>
                </div>
                <BarChart3 className="w-12 h-12 text-indigo-200" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-500">Pending</p>
                  <p className="text-4xl font-bold mt-2 text-yellow-600">
                    {isLoading ? <Skeleton className="h-10 w-12" /> : stats.pending}
                  </p>
                </div>
                <Clock className="w-12 h-12 text-yellow-200" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-500">Completed</p>
                  <p className="text-4xl font-bold mt-2 text-green-600">
                    {isLoading ? <Skeleton className="h-10 w-12" /> : stats.completed}
                  </p>
                </div>
                <CheckCircle className="w-12 h-12 text-green-200" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-slate-500">Failed</p>
                  <p className="text-4xl font-bold mt-2 text-red-600">
                    {isLoading ? <Skeleton className="h-10 w-12" /> : stats.failed}
                  </p>
                </div>
                <AlertCircle className="w-12 h-12 text-red-200" />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardContent className="p-6">
              <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
                Quick Actions
              </h2>
              <div className="grid grid-cols-2 gap-4">
                <Link href="/analyze">
                  <Button variant="outline" className="w-full h-24 flex flex-col gap-2">
                    <FileVideo className="w-6 h-6" />
                    Analyze Video
                  </Button>
                </Link>
                <Button variant="outline" className="w-full h-24 flex flex-col gap-2">
                  <FileAudio className="w-6 h-6" />
                  Analyze Audio
                </Button>
                <Button variant="outline" className="w-full h-24 flex flex-col gap-2">
                  <Image className="w-6 h-6" />
                  Analyze Image
                </Button>
                <Button variant="outline" className="w-full h-24 flex flex-col gap-2">
                  <FileText className="w-6 h-6" />
                  Analyze Text
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
                Recent Activity
              </h2>
              {isLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : analysesData?.items?.length === 0 ? (
                <div className="text-center py-8">
                  <Shield className="w-12 h-12 text-slate-300 mx-auto mb-3" />
                  <p className="text-slate-500">No analyses yet</p>
                  <Link href="/analyze">
                    <Button className="mt-4">Start Your First Analysis</Button>
                  </Link>
                </div>
              ) : (
                <div className="space-y-3">
                  {analysesData?.items?.slice(0, 5).map((analysis: any) => (
                    <div
                      key={analysis.id}
                      className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        {analysis.file_type === "video" && (
                          <FileVideo className="w-5 h-5 text-indigo-500" />
                        )}
                        {analysis.file_type === "audio" && (
                          <FileAudio className="w-5 h-5 text-green-500" />
                        )}
                        {analysis.file_type === "image" && (
                          <Image className="w-5 h-5 text-blue-500" />
                        )}
                        {analysis.file_type === "text" && (
                          <FileText className="w-5 h-5 text-purple-500" />
                        )}
                        <div>
                          <p className="font-medium text-sm">{analysis.file_name}</p>
                          <p className="text-xs text-slate-500">
                            {new Date(analysis.created_at).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <Badge
                        variant={
                          analysis.status === "completed"
                            ? "default"
                            : analysis.status === "failed"
                            ? "destructive"
                            : "secondary"
                        }
                      >
                        {analysis.status}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </motion.div>
    </div>
  );
}
