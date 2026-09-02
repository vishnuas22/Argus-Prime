"use client";

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import type { TimeSeriesData } from '@/types';

interface FrameScoreChartProps {
  data: TimeSeriesData[];
  title?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  color?: string;
}

export function FrameScoreChart({
  data,
  title = 'Manipulation Score Over Time',
  xAxisLabel = 'Time (s)',
  yAxisLabel = 'Score',
  color = '#6366f1',
}: FrameScoreChartProps) {
  const formattedData = data.map((point) => ({
    ...point,
    time: new Date(point.timestamp).toLocaleTimeString(),
  }));

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
      {title && (
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
          {title}
        </h3>
      )}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={formattedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 12, fill: '#6b7280' }}
              label={{
                value: xAxisLabel,
                position: 'bottom',
                offset: 0,
                style: { fontSize: 12, fill: '#6b7280' },
              }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#6b7280' }}
              domain={[0, 1]}
              label={{
                value: yAxisLabel,
                angle: -90,
                position: 'insideLeft',
                style: { fontSize: 12, fill: '#6b7280' },
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
              }}
              labelStyle={{ color: '#111827', fontWeight: 600 }}
              itemStyle={{ color: '#4b5563' }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="value"
              name="Manipulation Score"
              stroke={color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6, strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
