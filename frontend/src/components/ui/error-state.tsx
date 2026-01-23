"use client";

import React from "react";
import { motion } from "framer-motion";
import { AlertTriangle, RefreshCw, Home, ChevronRight, FileWarning } from "lucide-react";
import { Button } from "./button";
import { Card } from "./card";

interface ErrorStateProps {
  title?: string;
  message?: string;
  code?: string | number;
  onRetry?: () => void;
  onGoHome?: () => void;
  className?: string;
}

export function ErrorState({
  title = "Something went wrong",
  message = "An unexpected error occurred. Please try again.",
  code,
  onRetry,
  onGoHome,
  className = "",
}: ErrorStateProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex flex-col items-center justify-center p-8 ${className}`}
    >
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 200, damping: 15, delay: 0.1 }}
        className="w-16 h-16 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center mb-6"
      >
        <AlertTriangle size={32} className="text-red-600 dark:text-red-400" />
      </motion.div>

      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-xl font-semibold text-slate-900 dark:text-white mb-2"
      >
        {title}
      </motion.h2>

      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="text-slate-500 dark:text-slate-400 text-center mb-6 max-w-md"
      >
        {message}
      </motion.p>

      {code && (
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.35 }}
          className="text-xs font-mono text-slate-400 dark:text-slate-500 mb-6 px-3 py-1 bg-slate-100 dark:bg-slate-800 rounded"
        >
          Error code: {code}
        </motion.span>
      )}

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="flex gap-3"
      >
        {onRetry && (
          <Button variant="primary" onClick={onRetry} leftIcon={<RefreshCw size={16} />}>
            Try Again
          </Button>
        )}
        {onGoHome && (
          <Button variant="outline" onClick={onGoHome} leftIcon={<Home size={16} />}>
            Go to Dashboard
          </Button>
        )}
      </motion.div>
    </motion.div>
  );
}

interface NotFoundStateProps {
  title?: string;
  message?: string;
  onGoBack?: () => void;
  onGoHome?: () => void;
  className?: string;
}

export function NotFoundState({
  title = "Page not found",
  message = "The page you're looking for doesn't exist or has been moved.",
  onGoBack,
  onGoHome,
  className = "",
}: NotFoundStateProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex flex-col items-center justify-center p-8 ${className}`}
    >
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 200, damping: 15, delay: 0.1 }}
        className="w-20 h-20 bg-amber-100 dark:bg-amber-900/30 rounded-full flex items-center justify-center mb-6"
      >
        <FileWarning size={40} className="text-amber-600 dark:text-amber-400" />
      </motion.div>

      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-2xl font-semibold text-slate-900 dark:text-white mb-2"
      >
        {title}
      </motion.h2>

      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="text-slate-500 dark:text-slate-400 text-center mb-6 max-w-md"
      >
        {message}
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="flex gap-3"
      >
        {onGoBack && (
          <Button variant="outline" onClick={onGoBack} leftIcon={<ChevronRight size={16} className="rotate-180" />}>
            Go Back
          </Button>
        )}
        {onGoHome && (
          <Button variant="primary" onClick={onGoHome} leftIcon={<Home size={16} />}>
            Go to Dashboard
          </Button>
        )}
      </motion.div>
    </motion.div>
  );
}

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className = "",
}: EmptyStateProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex flex-col items-center justify-center p-8 text-center ${className}`}
    >
      {icon && (
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 200, damping: 15, delay: 0.1 }}
          className="w-16 h-16 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center mb-6 text-slate-400"
        >
          {icon}
        </motion.div>
      )}

      <motion.h3
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-lg font-medium text-slate-900 dark:text-white mb-2"
      >
        {title}
      </motion.h3>

      {description && (
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="text-slate-500 dark:text-slate-400 text-sm mb-6 max-w-sm"
        >
          {description}
        </motion.p>
      )}

      {action && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          {action}
        </motion.div>
      )}
    </motion.div>
  );
}

interface ServerErrorStateProps {
  onRetry?: () => void;
  className?: string;
}

export function ServerErrorState({ onRetry, className = "" }: ServerErrorStateProps) {
  return (
    <Card className={`max-w-md mx-auto ${className}`}>
      <ErrorState
        title="Server Error"
        message="We're having trouble connecting to our servers. Please check your connection and try again."
        onRetry={onRetry}
      />
    </Card>
  );
}
