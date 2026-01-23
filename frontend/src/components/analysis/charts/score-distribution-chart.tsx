"use client";

import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from 'recharts';
import type { ScoreDistribution } from '@/types';

interface ScoreDistributionChartProps {
  data: ScoreDistribution[];
  title?: string;
}

const COLORS = {
  high: '#22c55e',
  medium: '#f59e0b',
  low: '#ef4444',
};

export function ScoreDistributionChart({ data, title = 'Score Distribution' }: ScoreDistributionChartProps) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
      {title && (
        <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
          {title}
        </h3>
      )}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={2}
              dataKey="count"
              nameKey="range"
              label={({ range, percentage }) => `${range}: ${percentage}%`}
              labelLine={false}
            >
              {data.map((entry, index) => {
                const key = entry.range.toLowerCase().includes('high')
                  ? 'high'
                  : entry.range.toLowerCase().includes('low')
                  ? 'low'
                  : 'medium';
                return <Cell key={`cell-${index}`} fill={COLORS[key]} />;
              })}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
              }}
              formatter={(value: number, name: string) => [
                `${value} analyses`,
                name === 'range' ? 'Score Range' : name,
              ]}
            />
            <Legend
              verticalAlign="bottom"
              height={36}
              formatter={(value) => (
                <span className="text-sm text-slate-600 dark:text-slate-300">{value}</span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
