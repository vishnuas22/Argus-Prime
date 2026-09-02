"use client";

import React, { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Bell,
  Check,
  Trash2,
  CheckCircle,
  AlertTriangle,
  Info,
  FileText,
  Settings,
  Search,
  Filter,
  RefreshCw,
} from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/ui/error-state";
import { useToast } from "@/components/ui/toast";

interface Notification {
  id: string;
  type: "success" | "warning" | "error" | "info";
  title: string;
  message: string;
  read: boolean;
  createdAt: string;
  analysisId?: string;
  actionUrl?: string;
}

const mockNotifications: Notification[] = [
  {
    id: "1",
    type: "success",
    title: "Analysis Complete",
    message: "Your video analysis for 'sample_video.mp4' is ready with 85% trust score.",
    read: false,
    createdAt: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
    analysisId: "123",
  },
  {
    id: "2",
    type: "warning",
    title: "Low Trust Score Detected",
    message: "Document 'contract.pdf' received a 35% trust score. Manual review recommended.",
    read: false,
    createdAt: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
    analysisId: "456",
  },
  {
    id: "3",
    type: "info",
    title: "New Feature Available",
    message: "Try our enhanced audio deepfake detection with improved accuracy.",
    read: true,
    createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: "4",
    type: "success",
    title: "Report Generated",
    message: "Your forensic report for 'Q4 Financial Analysis' has been generated.",
    read: true,
    createdAt: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
    analysisId: "789",
  },
  {
    id: "5",
    type: "error",
    title: "Analysis Failed",
    message: "Unable to process 'interview.mp4'. Please try uploading again.",
    read: true,
    createdAt: new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString(),
  },
];

const TYPE_CONFIG = {
  success: { icon: CheckCircle, color: "text-emerald-500", bg: "bg-emerald-100 dark:bg-emerald-900/30" },
  warning: { icon: AlertTriangle, color: "text-amber-500", bg: "bg-amber-100 dark:bg-amber-900/30" },
  error: { icon: AlertTriangle, color: "text-red-500", bg: "bg-red-100 dark:bg-red-900/30" },
  info: { icon: Info, color: "text-blue-500", bg: "bg-blue-100 dark:bg-blue-900/30" },
};

export default function NotificationsPage() {
  const queryClient = useQueryClient();
  const { addToast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [showRead, setShowRead] = useState(true);

  const { data: notifications = mockNotifications, isLoading, refetch } = useQuery({
    queryKey: ["notifications"],
    queryFn: async () => mockNotifications,
    staleTime: 30000,
  });

  const markAsReadMutation = useMutation({
    mutationFn: (id: string) => Promise.resolve(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
    },
  });

  const markAllAsReadMutation = useMutation({
    mutationFn: () => Promise.resolve(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
      addToast("success", "Done", "All notifications marked as read");
    },
  });

  const deleteNotificationMutation = useMutation({
    mutationFn: (id: string) => Promise.resolve(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
      addToast("success", "Deleted", "Notification removed");
    },
  });

  const filteredNotifications = notifications.filter((notification) => {
    if (!showRead && notification.read) return false;
    if (typeFilter !== "all" && notification.type !== typeFilter) return false;
    if (
      searchQuery &&
      !notification.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
      !notification.message.toLowerCase().includes(searchQuery.toLowerCase())
    ) {
      return false;
    }
    return true;
  });

  const unreadCount = notifications.filter((n) => !n.read).length;

  const handleMarkAsRead = (id: string) => {
    markAsReadMutation.mutate(id);
  };

  const handleMarkAllAsRead = () => {
    markAllAsReadMutation.mutate();
  };

  const handleDelete = (id: string) => {
    if (confirm("Are you sure you want to delete this notification?")) {
      deleteNotificationMutation.mutate(id);
    }
  };

  const formatTimeAgo = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  if (isLoading) {
    return (
      <div className="p-6 max-w-4xl mx-auto">
        <div className="space-y-4">
          <Skeleton className="h-10 w-48" />
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-24 w-full rounded-xl" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-3">
              <Bell size={24} className="text-indigo-600" />
              Notifications
            </h1>
            <p className="text-slate-500 dark:text-slate-400 mt-1">
              {unreadCount > 0
                ? `You have ${unreadCount} unread notification${unreadCount > 1 ? "s" : ""}`
                : "All caught up!"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {unreadCount > 0 && (
              <Button variant="outline" onClick={handleMarkAllAsRead}>
                <Check size={16} className="mr-2" />
                Mark all as read
              </Button>
            )}
            <Button variant="outline" onClick={() => refetch()}>
              <RefreshCw size={16} className="mr-2" />
              Refresh
            </Button>
          </div>
        </div>

        <Card className="p-4">
          <div className="flex flex-col sm:flex-row gap-4 mb-4">
            <div className="flex-1">
              <Input
                placeholder="Search notifications..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                leftIcon={<Search size={16} />}
              />
            </div>
            <div className="flex gap-2">
              <Select
                options={[
                  { value: "all", label: "All Types" },
                  { value: "success", label: "Success" },
                  { value: "warning", label: "Warning" },
                  { value: "error", label: "Error" },
                  { value: "info", label: "Info" },
                ]}
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="w-36"
              />
              <Button
                variant={showRead ? "primary" : "outline"}
                onClick={() => setShowRead(!showRead)}
              >
                <Filter size={16} className="mr-2" />
                {showRead ? "Show All" : "Hide Read"}
              </Button>
            </div>
          </div>

          {filteredNotifications.length > 0 ? (
            <div className="space-y-3">
              {filteredNotifications.map((notification, index) => {
                const config = TYPE_CONFIG[notification.type];
                const Icon = config.icon;

                return (
                  <motion.div
                    key={notification.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.03 }}
                    className={`
                      flex items-start gap-4 p-4 rounded-xl transition-colors
                      ${
                        notification.read
                          ? "bg-slate-50 dark:bg-slate-800/50"
                          : "bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800"
                      }
                      hover:bg-slate-100 dark:hover:bg-slate-800
                    `}
                  >
                    <div
                      className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${config.bg}`}
                    >
                      <Icon size={20} className={config.color} />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <p className={`font-medium ${!notification.read ? "text-slate-900 dark:text-white" : "text-slate-700 dark:text-slate-300"}`}>
                            {notification.title}
                          </p>
                          <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">
                            {notification.message}
                          </p>
                        </div>
                        {!notification.read && (
                          <Badge variant="success" className="flex-shrink-0">
                            New
                          </Badge>
                        )}
                      </div>
                      <div className="flex items-center gap-4 mt-2">
                        <span className="text-xs text-slate-400 dark:text-slate-500">
                          {formatTimeAgo(notification.createdAt)}
                        </span>
                        {notification.analysisId && (
                          <Link
                            href={`/analyze/${notification.analysisId}`}
                            className="text-xs text-indigo-600 dark:text-indigo-400 hover:underline"
                          >
                            View Analysis
                          </Link>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-1 flex-shrink-0">
                      {!notification.read && (
                        <button
                          onClick={() => handleMarkAsRead(notification.id)}
                          className="p-2 rounded-lg text-slate-400 hover:text-emerald-500 hover:bg-emerald-50 dark:hover:bg-emerald-900/20 transition-colors"
                          title="Mark as read"
                        >
                          <Check size={16} />
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(notification.id)}
                        className="p-2 rounded-lg text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                        title="Delete"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          ) : (
            <EmptyState
              icon={<Bell size={40} />}
              title="No notifications found"
              description={
                searchQuery
                  ? "Try adjusting your search or filters"
                  : "You're all caught up!"
              }
              action={
                searchQuery ? (
                  <Button variant="outline" onClick={() => setSearchQuery("")}>
                    Clear Search
                  </Button>
                ) : undefined
              }
            />
          )}
        </Card>

        <div className="flex items-center justify-between text-sm text-slate-500 dark:text-slate-400">
          <span>
            Showing {filteredNotifications.length} of {notifications.length} notifications
          </span>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" disabled>
              Previous
            </Button>
            <span>Page 1 of 1</span>
            <Button variant="ghost" size="sm" disabled>
              Next
            </Button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
