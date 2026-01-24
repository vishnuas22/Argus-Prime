import React from "react";

export interface TabsProps extends React.HTMLAttributes<HTMLDivElement> {
  defaultValue?: string;
  value?: string;
  onValueChange?: (value: string) => void;
}

export function Tabs({ defaultValue, value, onValueChange, children, className = "", ...props }: TabsProps) {
  return (
    <div className={`space-y-4 ${className}`} {...props}>
      {children}
    </div>
  );
}

export function TabList({ children, className = "", ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={`flex space-x-1 rounded-lg bg-slate-100 p-1 ${className}`} {...props}>
      {children}
    </div>
  );
}

export function Tab({ value, children, className = "", leftIcon, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { value: string; leftIcon?: React.ReactNode }) {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-md px-3 py-1.5 text-sm font-medium transition-all focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ${className}`}
      {...props}
    >
      {leftIcon && <span className="mr-2">{leftIcon}</span>}
      {children}
    </button>
  );
}

export function TabPanel({ value, children, className = "", ...props }: React.HTMLAttributes<HTMLDivElement> & { value?: string }) {
  return (
    <div className={`mt-4 ${className}`} {...props}>
      {children}
    </div>
  );
}
