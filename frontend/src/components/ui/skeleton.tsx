"use client";

import React from "react";
import { motion } from "framer-motion";

interface SkeletonProps {
  variant?: "text" | "circular" | "rectangular" | "card";
  width?: string | number;
  height?: string | number;
  className?: string;
  animation?: "pulse" | "wave" | "none";
}

export function Skeleton({
  variant = "text",
  width,
  height,
  className = "",
  animation = "pulse",
}: SkeletonProps) {
  const baseClasses = "bg-slate-200 dark:bg-slate-700";

  const variantClasses = {
    text: "rounded",
    circular: "rounded-full",
    rectangular: "rounded-lg",
    card: "rounded-xl",
  };

  const style: React.CSSProperties = {
    width: typeof width === "number" ? `${width}px` : width,
    height: typeof height === "number" ? `${height}px` : height,
  };

  if (animation === "none") {
    return (
      <div
        className={`${baseClasses} ${variantClasses[variant]} ${className}`}
        style={style}
      />
    );
  }

  if (animation === "wave") {
    return (
      <motion.div
        className={`${baseClasses} ${variantClasses[variant]} ${className} relative overflow-hidden`}
        style={{
          ...style,
          background: `linear-gradient(
            90deg,
            transparent 0%,
            rgba(255, 255, 255, 0.4) 50%,
            transparent 100%
          )`,
          backgroundSize: "200% 100%",
        }}
        animate={{
          backgroundPosition: ["0% 0%", "100% 0%"],
        }}
        transition={{
          duration: 1.5,
          repeat: Infinity,
          ease: "linear",
        }}
      />
    );
  }

  return (
    <motion.div
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      style={style}
      animate={{ opacity: [1, 0.5, 1] }}
      transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
    />
  );
}

interface SkeletonTextProps {
  lines?: number;
  className?: string;
  width?: string | number;
}

export function SkeletonText({ lines = 3, className = "", width }: SkeletonTextProps) {
  return (
    <div className={`space-y-2 ${className}`}>
      {Array.from({ length: lines }).map((_, index) => (
        <Skeleton
          key={index}
          variant="text"
          width={index === lines - 1 && width ? width : "100%"}
          height="1em"
          animation="pulse"
        />
      ))}
    </div>
  );
}

interface SkeletonCardProps {
  className?: string;
  showImage?: boolean;
  showTitle?: boolean;
  showDescription?: boolean;
  showFooter?: boolean;
}

export function SkeletonCard({
  className = "",
  showImage = true,
  showTitle = true,
  showDescription = true,
  showFooter = true,
}: SkeletonCardProps) {
  return (
    <div className={`bg-white dark:bg-slate-800 rounded-xl p-6 space-y-4 ${className}`}>
      {showImage && (
        <Skeleton
          variant="rectangular"
          width="100%"
          height="200px"
          animation="pulse"
        />
      )}
      {showTitle && (
        <div className="space-y-2">
          <Skeleton variant="text" width="70%" height="1.5em" animation="pulse" />
          <Skeleton variant="text" width="50%" height="1em" animation="pulse" />
        </div>
      )}
      {showDescription && (
        <div className="space-y-2">
          <Skeleton variant="text" width="100%" animation="pulse" />
          <Skeleton variant="text" width="85%" animation="pulse" />
          <Skeleton variant="text" width="60%" animation="pulse" />
        </div>
      )}
      {showFooter && (
        <div className="flex justify-end gap-2 pt-2">
          <Skeleton variant="circular" width="36px" height="36px" animation="pulse" />
          <Skeleton variant="circular" width="36px" height="36px" animation="pulse" />
        </div>
      )}
    </div>
  );
}

interface SkeletonTableProps {
  columns: number;
  rows?: number;
  className?: string;
}

export function SkeletonTable({ columns, rows = 5, className = "" }: SkeletonTableProps) {
  return (
    <div className={`space-y-3 ${className}`}>
      <div className="flex gap-3">
        {Array.from({ length: columns }).map((_, index) => (
          <Skeleton
            key={index}
            variant="text"
            width={`${100 / columns}%`}
            height="2.5em"
            animation="pulse"
          />
        ))}
      </div>
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div key={rowIndex} className="flex gap-3">
          {Array.from({ length: columns }).map((_, colIndex) => (
            <Skeleton
              key={colIndex}
              variant="text"
              width={`${100 / columns}%`}
              height="2em"
              animation="pulse"
            />
          ))}
        </div>
      ))}
    </div>
  );
}

interface SkeletonListProps {
  items?: number;
  className?: string;
}

export function SkeletonList({ items = 4, className = "" }: SkeletonListProps) {
  return (
    <div className={`space-y-3 ${className}`}>
      {Array.from({ length: items }).map((_, index) => (
        <div key={index} className="flex items-center gap-3 p-3">
          <Skeleton variant="circular" width="40px" height="40px" animation="pulse" />
          <div className="flex-1 space-y-2">
            <Skeleton variant="text" width="70%" animation="pulse" />
            <Skeleton variant="text" width="40%" height="0.8em" animation="pulse" />
          </div>
        </div>
      ))}
    </div>
  );
}
