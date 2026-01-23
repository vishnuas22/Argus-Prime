"use client";

import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { ModalityContribution } from '@/types';

interface ScoreBreakdownChartProps {
  data: ModalityContribution[];
}

const COLORS = {
  video: '#6366f1',
  audio: '#8b5cf6',
  text: '#f59e0b',
};

export function ScoreBreakdownChart({ data }: ScoreBreakdownChartProps) {
  const formattedData = data.map((item) => ({
    ...item,
    name: item.modality.charAt(0).toUpperCase() + item.modality.slice(1),
    fill: COLORS[item.modality] || '#6b7280',
  }));

  const maxContribution = Math.max(...data.map((d) => d.contribution));

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
      <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
        Score Breakdown
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={formattedData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 100]}
              tick={{ fontSize: 12, fill: '#6b7280' }}
              label={{
                value: 'Contribution (%)',
                position: 'bottom',
                offset: 0,
                style: { fontSize: 12, fill: '#6b7280' },
              }}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fontSize: 12, fill: '#6b7280' }}
              width={60}
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
              formatter={(value: number, name: string) => [
                `${value.toFixed(1)}%`,
                name === 'contribution' ? 'Contribution' : name,
              ]}
            />
            <Bar
              dataKey="contribution"
              radius={[0, 4, 4, 0]}
              maxBarSize={40}
            >
              {formattedData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-4">
        {data.map((item) => (
          <div key={item.modality} className="text-center">
            <div
              className="w-3 h-3 rounded-full mx-auto mb-1"
              style={{ backgroundColor: COLORS[item.modality] }}
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 capitalize">
              {item.modality}
            </p>
            <p className="text-sm font-semibold text-slate-900 dark:text-white">
              {item.contribution.toFixed(1)}%
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
