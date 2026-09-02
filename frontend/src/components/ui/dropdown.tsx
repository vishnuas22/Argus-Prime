"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown } from "lucide-react";

interface DropdownItem {
  label: string;
  value: string;
  icon?: React.ReactNode;
  disabled?: boolean;
  danger?: boolean;
  divider?: boolean;
}

interface DropdownProps {
  trigger: React.ReactNode;
  items: DropdownItem[];
  value?: string;
  onChange?: (value: string) => void;
  align?: "left" | "right";
  className?: string;
}

export function Dropdown({ trigger, items, value, onChange, align = "left", className = "" }: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleItemClick = (item: DropdownItem) => {
    if (!item.disabled && !item.divider) {
      onChange?.(item.value);
      setIsOpen(false);
    }
  };

  return (
    <div ref={dropdownRef} className={`relative inline-block ${className}`}>
      <div onClick={() => setIsOpen(!isOpen)}>{trigger}</div>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.15 }}
            className={`
              absolute z-50 min-w-[180px] mt-2 py-1
              bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-slate-200 dark:border-slate-700
              ${align === "left" ? "left-0" : "right-0"}
            `}
          >
            {items.map((item, index) => (
              <React.Fragment key={item.value}>
                {item.divider ? (
                  <div className="my-1 border-t border-slate-100 dark:border-slate-700" />
                ) : (
                  <button
                    onClick={() => handleItemClick(item)}
                    disabled={item.disabled}
                    className={`
                      w-full flex items-center gap-2 px-3 py-2 text-sm
                      transition-colors duration-150
                      ${
                        item.disabled
                          ? "opacity-50 cursor-not-allowed text-slate-400"
                          : item.danger
                          ? "text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
                          : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                      }
                    `}
                  >
                    {item.icon && <span className="w-4 h-4">{item.icon}</span>}
                    {item.label}
                    {item.value === value && (
                      <span className="ml-auto text-indigo-600 dark:text-indigo-400">✓</span>
                    )}
                  </button>
                )}
              </React.Fragment>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface DropdownButtonProps {
  children: React.ReactNode;
  variant?: "default" | "primary" | "ghost";
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function DropdownButton({ children, variant = "default", size = "md", className = "" }: DropdownButtonProps) {
  const variants = {
    default: "bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300",
    primary: "bg-indigo-600 hover:bg-indigo-700 text-white",
    ghost: "hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300",
  };

  const sizes = {
    sm: "h-8 px-3 text-xs",
    md: "h-10 px-4 text-sm",
    lg: "h-12 px-5 text-base",
  };

  return (
    <button
      className={`
        inline-flex items-center gap-2 rounded-lg font-medium transition-colors
        ${variants[variant]} ${sizes[size]} ${className}
      `}
    >
      {children}
      <ChevronDown size={16} className="opacity-60" />
    </button>
  );
}

interface DropdownMenuProps {
  children: React.ReactNode;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  align?: "left" | "right";
  className?: string;
}

export function DropdownMenu({ children, open, onOpenChange, align = "left", className = "" }: DropdownMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onOpenChange(false);
      }
    };

    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [open, onOpenChange]);

  return (
    <div ref={menuRef} className={`relative inline-block ${className}`}>
      {children}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.15 }}
            className={`
              absolute z-50 min-w-[200px] mt-2 py-1
              bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-slate-200 dark:border-slate-700
              ${align === "left" ? "left-0" : "right-0"}
            `}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface DropdownMenuItemProps {
  label: string;
  icon?: React.ReactNode;
  shortcut?: string;
  disabled?: boolean;
  danger?: boolean;
  onClick?: () => void;
}

export function DropdownMenuItem({ label, icon, shortcut, disabled, danger, onClick }: DropdownMenuItemProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        w-full flex items-center gap-2 px-3 py-2 text-sm
        transition-colors duration-150
        ${
          disabled
            ? "opacity-50 cursor-not-allowed text-slate-400"
            : danger
            ? "text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
            : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
        }
      `}
    >
      {icon && <span className="w-4 h-4">{icon}</span>}
      <span className="flex-1 text-left">{label}</span>
      {shortcut && <span className="text-xs text-slate-400">{shortcut}</span>}
    </button>
  );
}

interface DropdownMenuSeparatorProps {
  className?: string;
}

export function DropdownMenuSeparator({ className = "" }: DropdownMenuSeparatorProps) {
  return <div className={`my-1 border-t border-slate-100 dark:border-slate-700 ${className}`} />;
}
