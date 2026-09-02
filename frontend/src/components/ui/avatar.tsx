"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { User } from "lucide-react";

interface AvatarProps {
  src?: string | null;
  alt?: string;
  fallback?: string;
  size?: "xs" | "sm" | "md" | "lg" | "xl";
  shape?: "circle" | "square";
  status?: "online" | "offline" | "busy" | "away";
  className?: string;
}

const sizeClasses = {
  xs: "w-6 h-6 text-xs",
  sm: "w-8 h-8 text-sm",
  md: "w-10 h-10 text-base",
  lg: "w-12 h-12 text-lg",
  xl: "w-16 h-16 text-xl",
};

const statusColors = {
  online: "bg-emerald-500",
  offline: "bg-slate-400",
  busy: "bg-red-500",
  away: "bg-amber-500",
};

const statusSizes = {
  xs: "w-1.5 h-1.5 border",
  sm: "w-2 h-2 border",
  md: "w-2.5 h-2.5 border-2",
  lg: "w-3 h-3 border-2",
  xl: "w-4 h-4 border-[3px]",
};

export function Avatar({
  src,
  alt = "Avatar",
  fallback,
  size = "md",
  shape = "circle",
  status,
  className = "",
}: AvatarProps) {
  const [imageError, setImageError] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);

  const getInitials = (name: string): string => {
    return name
      .split(" ")
      .map((part) => part[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const displayFallback = fallback || getInitials(alt);
  const shapeClass = shape === "circle" ? "rounded-full" : "rounded-lg";
  const borderClass = shape === "circle" ? "border-2" : "border";

  const shouldShowImage = src && !imageError;

  return (
    <div className={`relative inline-flex ${className}`}>
      <motion.div
        className={`${sizeClasses[size]} ${shapeClass} overflow-hidden bg-slate-100 dark:bg-slate-700 flex items-center justify-center text-slate-600 dark:text-slate-300 font-medium border border-slate-200 dark:border-slate-600`}
        whileHover={{ scale: 1.05 }}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
      >
        <AnimatePresence mode="wait">
          {shouldShowImage ? (
            <motion.img
              key="image"
              src={src}
              alt={alt}
              onError={() => setImageError(true)}
              onLoad={() => setIsLoaded(true)}
              initial={{ opacity: 0 }}
              animate={{ opacity: isLoaded ? 1 : 0 }}
              exit={{ opacity: 0 }}
              className="w-full h-full object-cover"
            />
          ) : (
            <motion.div
              key="fallback"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center justify-center w-full h-full"
            >
              {displayFallback ? (
                <span className={size === "xs" || size === "sm" ? "" : ""}>
                  {displayFallback}
                </span>
              ) : (
                <User
                  className={`${
                    size === "xs"
                      ? "w-3 h-3"
                      : size === "sm"
                      ? "w-4 h-4"
                      : size === "md"
                      ? "w-5 h-5"
                      : size === "lg"
                      ? "w-6 h-6"
                      : "w-8 h-8"
                  }`}
                />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {status && (
        <motion.span
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          className={`absolute bottom-0 right-0 ${statusColors[status]} ${statusSizes[size]} ${shapeClass} ${borderClass} border-white dark:border-slate-900`}
        />
      )}
    </div>
  );
}

interface AvatarGroupProps {
  avatars: Array<{
    src?: string | null;
    alt?: string;
    fallback?: string;
  }>;
  max?: number;
  size?: "xs" | "sm" | "md" | "lg";
  className?: string;
}

export function AvatarGroup({ avatars, max = 4, size = "sm", className = "" }: AvatarGroupProps) {
  const visibleAvatars = avatars.slice(0, max);
  const remainingCount = avatars.length - max;

  return (
    <div className={`flex -space-x-2 ${className}`}>
      {visibleAvatars.map((avatar, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          style={{ zIndex: visibleAvatars.length - index }}
          className="hover:z-10 transition-transform hover:scale-110"
        >
          <Avatar {...avatar} size={size} />
        </motion.div>
      ))}

      {remainingCount > 0 && (
        <motion.div
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: visibleAvatars.length * 0.1 }}
          className="relative"
        >
          <Avatar
            fallback={`+${remainingCount}`}
            size={size}
            className="bg-slate-200 dark:bg-slate-600 text-slate-600 dark:text-slate-300"
          />
        </motion.div>
      )}
    </div>
  );
}
