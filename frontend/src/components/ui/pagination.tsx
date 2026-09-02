"use client";

import React from "react";
import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, MoreHorizontal } from "lucide-react";

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  siblingCount?: number;
  className?: string;
}

export function Pagination({
  currentPage,
  totalPages,
  onPageChange,
  siblingCount = 1,
  className = "",
}: PaginationProps) {
  const pages = React.useMemo(() => {
    const dots = "...";
    const pageNumbers: (number | string)[] = [];

    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) {
        pageNumbers.push(i);
      }
    } else {
      const leftBound = Math.max(2, currentPage - siblingCount);
      const rightBound = Math.min(totalPages - 1, currentPage + siblingCount);

      pageNumbers.push(1);

      if (leftBound > 2) {
        pageNumbers.push(dots);
      }

      for (let i = leftBound; i <= rightBound; i++) {
        pageNumbers.push(i);
      }

      if (rightBound < totalPages - 1) {
        pageNumbers.push(dots);
      }

      pageNumbers.push(totalPages);
    }

    return pageNumbers;
  }, [currentPage, totalPages, siblingCount]);

  const goToPage = (page: number) => {
    if (page >= 1 && page <= totalPages) {
      onPageChange(page);
    }
  };

  const goToPreviousPage = () => {
    if (currentPage > 1) {
      onPageChange(currentPage - 1);
    }
  };

  const goToNextPage = () => {
    if (currentPage < totalPages) {
      onPageChange(currentPage + 1);
    }
  };

  return (
    <nav className={`flex items-center gap-1 ${className}`} aria-label="Pagination">
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={goToPreviousPage}
        disabled={currentPage === 1}
        className="p-2 rounded-lg text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        aria-label="Previous page"
      >
        <ChevronLeft size={20} />
      </motion.button>

      {pages.map((page, index) => {
        if (page === "...") {
          return (
            <span
              key={`dots-${index}`}
              className="px-2 text-slate-400 dark:text-slate-500"
            >
              <MoreHorizontal size={16} />
            </span>
          );
        }

        const isActive = page === currentPage;

        return (
          <motion.button
            key={page}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => goToPage(page as number)}
            className={`
              min-w-[36px] h-9 px-3 rounded-lg text-sm font-medium transition-colors
              ${
                isActive
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700"
              }
            `}
            aria-label={`Page ${page}`}
            aria-current={isActive ? "page" : undefined}
          >
            {page}
          </motion.button>
        );
      })}

      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={goToNextPage}
        disabled={currentPage === totalPages}
        className="p-2 rounded-lg text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        aria-label="Next page"
      >
        <ChevronRight size={20} />
      </motion.button>
    </nav>
  );
}

interface PaginationInfoProps {
  currentPage: number;
  pageSize: number;
  totalItems: number;
  className?: string;
}

export function PaginationInfo({
  currentPage,
  pageSize,
  totalItems,
  className = "",
}: PaginationInfoProps) {
  const startItem = (currentPage - 1) * pageSize + 1;
  const endItem = Math.min(currentPage * pageSize, totalItems);

  return (
    <span className={`text-sm text-slate-500 dark:text-slate-400 ${className}`}>
      {startItem}-{endItem} of {totalItems}
    </span>
  );
}

interface PaginationSizeProps {
  value: number;
  onChange: (size: number) => void;
  options?: number[];
  className?: string;
}

export function PaginationSize({
  value,
  onChange,
  options = [10, 25, 50, 100],
  className = "",
}: PaginationSizeProps) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <span className="text-sm text-slate-500 dark:text-slate-400">Rows per page:</span>
      <select
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="
          h-9 px-3 rounded-lg text-sm
          bg-white dark:bg-slate-800
          border border-slate-200 dark:border-slate-700
          text-slate-700 dark:text-slate-300
          hover:border-slate-300 dark:hover:border-slate-600
          focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
          cursor-pointer
        "
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}

interface PaginationWithInfoProps {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange?: (size: number) => void;
  className?: string;
}

export function PaginationWithInfo({
  currentPage,
  totalPages,
  totalItems,
  pageSize,
  onPageChange,
  onPageSizeChange,
  className = "",
}: PaginationWithInfoProps) {
  return (
    <div className={`flex flex-wrap items-center justify-between gap-4 ${className}`}>
      {onPageSizeChange && (
        <PaginationSize value={pageSize} onChange={onPageSizeChange} />
      )}
      <PaginationInfo currentPage={currentPage} pageSize={pageSize} totalItems={totalItems} />
      <Pagination currentPage={currentPage} totalPages={totalPages} onPageChange={onPageChange} />
    </div>
  );
}
