'use client';

import * as React from 'react';
import { cn } from '@/lib/utils';

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'gradient';
  showLabel?: boolean;
  animated?: boolean;
  striped?: boolean;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  (
    {
      className,
      value = 0,
      max = 100,
      size = 'md',
      variant = 'gradient',
      showLabel = false,
      animated = true,
      striped = false,
      ...props
    },
    ref
  ) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

    const sizes = {
      sm: 'h-1',
      md: 'h-2',
      lg: 'h-3',
    };

    const variants = {
      default: 'bg-secondary-200',
      success: 'from-emerald-500 to-teal-500',
      warning: 'from-amber-500 to-orange-500',
      danger: 'from-red-500 to-rose-500',
      gradient: 'from-primary-500 to-primary-600',
    };

    return (
      <div className="w-full" ref={ref} {...props}>
        <div
          className={cn(
            'w-full bg-secondary-100 rounded-full overflow-hidden',
            sizes[size]
          )}
        >
          <div
            className={cn(
              'h-full rounded-full bg-gradient-to-r transition-all duration-500 ease-out',
              variants[variant],
              animated && 'animate-pulse-subtle'
            )}
            style={{
              width: `${percentage}%`,
              backgroundImage:
                striped && variant === 'gradient'
                  ? 'linear-gradient(45deg,rgba(255,255,255,.15) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.15) 50%,rgba(255,255,255,.15) 75%,transparent 75%,transparent)'
                  : undefined,
              backgroundSize: striped ? '1rem 1rem' : undefined,
            }}
          />
        </div>
        {showLabel && (
          <div className="flex justify-between mt-1.5 text-xs text-secondary-500">
            <span>Progress</span>
            <span className="font-medium">{Math.round(percentage)}%</span>
          </div>
        )}
      </div>
    );
  }
);
Progress.displayName = 'Progress';

interface CircularProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number;
  max?: number;
  size?: number;
  strokeWidth?: number;
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'gradient';
  showLabel?: boolean;
  label?: string;
  animated?: boolean;
}

const CircularProgress = React.forwardRef<HTMLDivElement, CircularProgressProps>(
  (
    {
      className,
      value = 0,
      max = 100,
      size = 120,
      strokeWidth = 8,
      variant = 'gradient',
      showLabel = true,
      label,
      animated = true,
      ...props
    },
    ref
  ) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (percentage / 100) * circumference;

    const variants = {
      default: 'stroke-secondary-300',
      success: 'stroke-emerald-500',
      warning: 'stroke-amber-500',
      danger: 'stroke-red-500',
      gradient: 'stroke-primary-500',
    };

    const gradientId = `progress-gradient-${Math.random().toString(36).substr(2, 9)}`;

    return (
      <div
        ref={ref}
        className={cn('relative inline-flex items-center justify-center', className)}
        style={{ width: size, height: size }}
        {...props}
      >
        <svg
          width={size}
          height={size}
          className="transform -rotate-90"
          viewBox={`0 0 ${size} ${size}`}
        >
          <defs>
            <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#6366f1" />
              <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
          </defs>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth={strokeWidth}
            className="text-secondary-100"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={variant === 'gradient' ? `url(#${gradientId})` : 'currentColor'}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className={cn(
              'transition-all duration-500 ease-out',
              animated && 'animate-pulse-subtle'
            )}
            style={{
              stroke: variant === 'gradient' ? `url(#${gradientId})` : undefined,
            }}
          />
        </svg>
        {showLabel && (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-bold text-secondary-900">
              {Math.round(percentage)}
            </span>
            {label && (
              <span className="text-xs text-secondary-500">{label}</span>
            )}
          </div>
        )}
      </div>
    );
  }
);
CircularProgress.displayName = 'CircularProgress';

export { Progress, CircularProgress };
