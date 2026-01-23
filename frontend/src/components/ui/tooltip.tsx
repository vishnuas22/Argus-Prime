"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactElement;
  position?: "top" | "bottom" | "left" | "right";
  align?: "start" | "center" | "end";
  delay?: number;
  className?: string;
  contentClassName?: string;
}

export function Tooltip({
  content,
  children,
  position = "top",
  align = "center",
  delay = 200,
  className = "",
  contentClassName = "",
}: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [positionStyle, setPositionStyle] = useState({});
  const timeoutRef = useRef<NodeJS.Timeout>();
  const childRef = useRef<HTMLElement>(null);

  const updatePosition = () => {
    if (!childRef.current) return;

    const rect = childRef.current.getBoundingClientRect();
    const tooltipWidth = 160;
    const tooltipHeight = 32;
    const gap = 8;

    let top = 0;
    let left = 0;

    switch (position) {
      case "top":
        top = rect.top - tooltipHeight - gap;
        left = rect.left + rect.width / 2;
        break;
      case "bottom":
        top = rect.bottom + gap;
        left = rect.left + rect.width / 2;
        break;
      case "left":
        top = rect.top + rect.height / 2;
        left = rect.left - tooltipWidth - gap;
        break;
      case "right":
        top = rect.top + rect.height / 2;
        left = rect.right + gap;
        break;
    }

    switch (align) {
      case "start":
        if (position === "top" || position === "bottom") {
          left = rect.left;
        } else {
          top = rect.top;
        }
        break;
      case "end":
        if (position === "top" || position === "bottom") {
          left = rect.right - tooltipWidth;
        } else {
          top = rect.bottom - tooltipHeight;
        }
        break;
    }

    setPositionStyle({
      top: `${top}px`,
      left: `${left}px`,
    });
  };

  useEffect(() => {
    if (isVisible) {
      updatePosition();
      window.addEventListener("scroll", updatePosition);
      window.addEventListener("resize", updatePosition);
    }

    return () => {
      window.removeEventListener("scroll", updatePosition);
      window.removeEventListener("resize", updatePosition);
    };
  }, [isVisible]);

  const handleMouseEnter = () => {
    timeoutRef.current = setTimeout(() => {
      setIsVisible(true);
    }, delay);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  const getPositionClasses = () => {
    const base = "absolute z-50";
    switch (position) {
      case "top":
        return `${base} bottom-full left-1/2 -translate-x-1/2 mb-2`;
      case "bottom":
        return `${base} top-full left-1/2 -translate-x-1/2 mt-2`;
      case "left":
        return `${base} right-full top-1/2 -translate-y-1/2 mr-2`;
      case "right":
        return `${base} left-full top-1/2 -translate-y-1/2 ml-2`;
    }
  };

  const getArrowClasses = () => {
    switch (position) {
      case "top":
        return "top-full left-1/2 -translate-x-1/2 border-t-slate-900";
      case "bottom":
        return "bottom-full left-1/2 -translate-x-1/2 border-b-slate-900";
      case "left":
        return "left-full top-1/2 -translate-y-1/2 border-l-slate-900";
      case "right":
        return "right-full top-1/2 -translate-y-1/2 border-r-slate-900";
    }
  };

  return (
    <div className={`relative inline-block ${className}`} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
      {React.cloneElement(children, { ref: childRef as React.Ref<HTMLElement> })}
      <AnimatePresence>
        {isVisible && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className={`${getPositionClasses()} fixed`}
            style={positionStyle}
          >
            <div
              className={`
                px-2 py-1 text-xs font-medium text-white bg-slate-900 rounded-md shadow-lg
                whitespace-nowrap max-w-[200px] overflow-hidden text-ellipsis
                ${contentClassName}
              `}
            >
              {content}
            </div>
            <div
              className={`
                absolute w-0 h-0 border-4 border-transparent
                ${getArrowClasses()}
              `}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface TooltipProviderProps {
  children: React.ReactNode;
  delay?: number;
}

export function TooltipProvider({ children, delay = 200 }: TooltipProviderProps) {
  return <>{children}</>;
}

interface TooltipTriggerProps {
  children: React.ReactElement;
  tooltip: React.ReactNode;
  asChild?: boolean;
}

export function TooltipTrigger({ children, tooltip }: TooltipTriggerProps) {
  return (
    <Tooltip content={tooltip}>
      {children}
    </Tooltip>
  );
}
