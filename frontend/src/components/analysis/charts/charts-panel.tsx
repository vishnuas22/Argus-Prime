"use client";

import React, { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
  ComposedChart,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  PieChart as PieChartIcon,
  RefreshCw,
  Download,
  Maximize2,
  X,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabList, Tab, TabPanel } from "@/components/ui/tabs";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";

interface DataPoint {
  name: string;
  value: number;
  [key: string]: string | number;
}

interface ChartData {
  labels: string[];
  datasets: Array<{
    name: string;
    data: number[];
    color?: string;
  }>;
}

interface VerdictData {
  name: string;
  value: number;
  color: string;
}

interface TimeSeriesPoint {
  timestamp: number;
  value: number;
  label?: string;
}

interface ChartsPanelProps {
  title?: string;
  description?: string;
  timeSeriesData?: TimeSeriesPoint[];
  verdictData?: VerdictData[];
  scoreDistribution?: DataPoint[];
  modalityScores?: Array<{
    modality: string;
    score: number;
    weight: number;
  }>;
  trendData?: Array<{
    date: string;
    authentic: number;
    manipulated: number;
    uncertain: number;
  }>;
  loading?: boolean;
  onExport?: () => void;
  onRefresh?: () => void;
  className?: string;
}

const COLORS = {
  authentic: "#10b981",
  uncertain: "#f59e0b",
  manipulated: "#ef4444",
  primary: "#6366f1",
  secondary: "#8b5cf6",
  tertiary: "#ec4899",
  quaternary: "#14b8a6",
};

const CHART_COLORS = ["#6366f1", "#8b5cf6", "#ec4899", "#14b8a6", "#f59e0b", "#10b981"];

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ color: string; name: string; value: number }>;
  label?: string;
}) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white dark:bg-slate-800 p-3 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700">
        <p className="text-sm font-medium text-slate-900 dark:text-white mb-2">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            {entry.name}: {typeof entry.value === "number" ? entry.value.toFixed(2) : entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export function ChartsPanel({
  title = "Analytics Dashboard",
  description,
  timeSeriesData,
  verdictData,
  scoreDistribution,
  modalityScores,
  trendData,
  loading = false,
  onExport,
  onRefresh,
  className = "",
}: ChartsPanelProps) {
  const [activeTab, setActiveTab] = useState("overview");
  const [timeRange, setTimeRange] = useState("7d");
  const [selectedChart, setSelectedChart] = useState<string | null>(null);

  const processedTrendData = useMemo(() => {
    if (trendData) return trendData;
    if (!timeSeriesData) return [];

    const days = timeRange === "7d" ? 7 : timeRange === "30d" ? 30 : 90;
    const dailyData: { [key: string]: { authentic: number; manipulated: number; uncertain: number } } = {};

    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split("T")[0];
      dailyData[dateStr] = { authentic: Math.random() * 50 + 20, manipulated: Math.random() * 30 + 10, uncertain: Math.random() * 20 + 5 };
    }

    return Object.entries(dailyData)
      .map(([date, counts]) => ({ date, ...counts }))
      .reverse();
  }, [trendData, timeSeriesData, timeRange]);

  const processedVerdictData = useMemo(() => {
    if (verdictData) return verdictData;
    return [
      { name: "Authentic", value: 57, color: COLORS.authentic },
      { name: "Uncertain", value: 18, color: COLORS.uncertain },
      { name: "Manipulated", value: 25, color: COLORS.manipulated },
    ];
  }, [verdictData]);

  const processedDistribution = useMemo(() => {
    if (scoreDistribution) return scoreDistribution;
    return [
      { name: "0-20", value: 15 },
      { name: "21-40", value: 25 },
      { name: "41-60", value: 35 },
      { name: "61-80", value: 18 },
      { name: "81-100", value: 7 },
    ];
  }, [scoreDistribution]);

  const processedModalityScores = useMemo(() => {
    if (modalityScores) return modalityScores;
    return [
      { modality: "Video", score: 72, weight: 0.4 },
      { modality: "Audio", score: 68, weight: 0.35 },
      { modality: "Text", score: 85, weight: 0.25 },
    ];
  }, [modalityScores]);

  if (loading) {
    return (
      <Card className={`p-6 ${className}`}>
        <div className="space-y-4">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-64 w-full" />
          <div className="grid grid-cols-2 gap-4">
            <Skeleton className="h-48 w-full" />
            <Skeleton className="h-48 w-full" />
          </div>
        </div>
      </Card>
    );
  }

  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-white">{title}</h2>
          {description && (
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">{description}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Select
            options={[
              { value: "7d", label: "Last 7 days" },
              { value: "30d", label: "Last 30 days" },
              { value: "90d", label: "Last 90 days" },
            ]}
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="w-36"
          />
          {onRefresh && (
            <Button variant="outline" size="sm" onClick={onRefresh}>
              <RefreshCw size={14} className="mr-1" />
              Refresh
            </Button>
          )}
          {onExport && (
            <Button variant="outline" size="sm" onClick={onExport}>
              <Download size={14} className="mr-1" />
              Export
            </Button>
          )}
        </div>
      </div>

      <Tabs defaultValue="overview">
        <TabList className="mb-6">
          <Tab value="overview">
            <BarChart3 size={16} className="mr-2" />
            Overview
          </Tab>
          <Tab value="trends">
            <TrendingUp size={16} className="mr-2" />
            Trends
          </Tab>
          <Tab value="distribution">
            <Activity size={16} className="mr-2" />
            Distribution
          </Tab>
        </TabList>

        <TabPanel value="overview">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Verdict Distribution
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={processedVerdictData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {processedVerdictData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                      formatter={(value, entry) => (
                        <span className="text-sm text-slate-600 dark:text-slate-300">
                          {value} ({entry.payload?.value}%)
                        </span>
                      )}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Modality Scores
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <RadialBarChart
                    cx="50%"
                    cy="50%"
                    innerRadius={20}
                    outerRadius={120}
                    data={processedModalityScores}
                    startAngle={180}
                    endAngle={0}
                  >
                    <RadialBar
                      label={{ fill: "#666", position: "insideStart" }}
                      background
                      dataKey="score"
                    >
                      {processedModalityScores.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                      ))}
                    </RadialBar>
                    <Legend
                      iconSize={10}
                      layout="vertical"
                      verticalAlign="middle"
                      align="right"
                      formatter={(value, entry: any) => (
                        <span className="text-sm text-slate-600 dark:text-slate-300">
                          {entry.payload?.modality} ({entry.payload?.score}%)
                        </span>
                      )}
                    />
                    <Tooltip />
                  </RadialBarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card className="p-6 lg:col-span-2">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Score Breakdown
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={processedModalityScores} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12, fill: "#6b7280" }} />
                    <YAxis
                      type="category"
                      dataKey="modality"
                      tick={{ fontSize: 12, fill: "#6b7280" }}
                      width={80}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="score" fill={COLORS.primary} radius={[0, 4, 4, 0]} />
                    <Bar dataKey="weight" fill={COLORS.secondary} radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>
        </TabPanel>

        <TabPanel value="trends">
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
              Analysis Trends
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={processedTrendData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 12, fill: "#6b7280" }}
                    tickFormatter={(value) => new Date(value).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                  />
                  <YAxis tick={{ fontSize: 12, fill: "#6b7280" }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="authentic"
                    fill={COLORS.authentic}
                    stroke={COLORS.authentic}
                    fillOpacity={0.3}
                    name="Authentic"
                  />
                  <Area
                    type="monotone"
                    dataKey="uncertain"
                    fill={COLORS.uncertain}
                    stroke={COLORS.uncertain}
                    fillOpacity={0.3}
                    name="Uncertain"
                  />
                  <Area
                    type="monotone"
                    dataKey="manipulated"
                    fill={COLORS.manipulated}
                    stroke={COLORS.manipulated}
                    fillOpacity={0.3}
                    name="Manipulated"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </TabPanel>

        <TabPanel value="distribution">
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
              Trust Score Distribution
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={processedDistribution}>
                  <defs>
                    <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={COLORS.primary} stopOpacity={0.8} />
                      <stop offset="95%" stopColor={COLORS.primary} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: "#6b7280" }} />
                  <YAxis tick={{ fontSize: 12, fill: "#6b7280" }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke={COLORS.primary}
                    fillOpacity={1}
                    fill="url(#colorValue)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </TabPanel>
      </Tabs>
    </div>
  );
}

interface MiniChartProps {
  data: Array<{ x: number | string; y: number }>;
  type?: "line" | "bar" | "area";
  color?: string;
  height?: number;
  className?: string;
}

export function MiniChart({
  data,
  type = "line",
  color = COLORS.primary,
  height = 40,
  className = "",
}: MiniChartProps) {
  const ChartComponent = type === "bar" ? BarChart : type === "area" ? AreaChart : LineChart;

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={height}>
        <ChartComponent data={data}>
          {type === "line" && (
            <Line
              type="monotone"
              dataKey="y"
              stroke={color}
              strokeWidth={2}
              dot={false}
            />
          )}
          {type === "bar" && (
            <Bar dataKey="y" fill={color} radius={[2, 2, 0, 0]} />
          )}
          {type === "area" && (
            <Area
              type="monotone"
              dataKey="y"
              stroke={color}
              fill={color}
              fillOpacity={0.3}
            />
          )}
        </ChartComponent>
      </ResponsiveContainer>
    </div>
  );
}

interface ScoreGaugeProps {
  score: number;
  label?: string;
  size?: "sm" | "md" | "lg";
  showTrend?: boolean;
  trend?: "up" | "down" | "stable";
}

export function ScoreGauge({
  score,
  label = "Trust Score",
  size = "md",
  showTrend = false,
  trend,
}: ScoreGaugeProps) {
  const sizeClasses = {
    sm: "w-16 h-16",
    md: "w-24 h-24",
    lg: "w-32 h-32",
  };

  const fontClasses = {
    sm: "text-lg",
    md: "text-2xl",
    lg: "text-3xl",
  };

  const getScoreColor = (score: number) => {
    if (score >= 70) return COLORS.authentic;
    if (score >= 40) return COLORS.uncertain;
    return COLORS.manipulated;
  };

  const circumference = 2 * Math.PI * 28;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className="relative inline-flex flex-col items-center">
      <svg className={`${sizeClasses[size]} transform -rotate-90`}>
        <circle
          cx="50%"
          cy="50%"
          r="28"
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="6"
        />
        <motion.circle
          cx="50%"
          cy="50%"
          r="28"
          fill="none"
          stroke={getScoreColor(score)}
          strokeWidth="6"
          strokeLinecap="round"
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1, ease: "easeOut" }}
          strokeDasharray={circumference}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`${fontClasses[size]} font-bold text-slate-900 dark:text-white`}>
          {score}
        </span>
        {showTrend && trend && (
          <span className="text-xs">
            {trend === "up" && <TrendingUp size={12} className="text-emerald-500" />}
            {trend === "down" && <TrendingDown size={12} className="text-red-500" />}
          </span>
        )}
      </div>
      {label && (
        <span className="text-xs text-slate-500 dark:text-slate-400 mt-1">{label}</span>
      )}
    </div>
  );
}
