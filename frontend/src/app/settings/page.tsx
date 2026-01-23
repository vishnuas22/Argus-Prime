"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Settings,
  User,
  Bell,
  Shield,
  Palette,
  Database,
  Key,
  Save,
  RefreshCw,
  Trash2,
  Download,
  Moon,
  Sun,
  Monitor,
} from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { analysisApi, healthApi } from "@/lib/api";
import { useThemeStore } from "@/stores/ui.store";
import { useToast } from "@/components/ui/toast";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabList, Tab, TabPanel } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Avatar } from "@/components/ui/avatar";
import { Skeleton } from "@/components/ui/skeleton";
import { Select } from "@/components/ui/select";

interface NotificationSettings {
  analysisComplete: boolean;
  highRiskAlert: boolean;
  weeklyDigest: boolean;
  securityUpdates: boolean;
}

interface ProfileSettings {
  fullName: string;
  email: string;
  organization: string;
  role: string;
}

const THEME_OPTIONS = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
] as const;

const ACCENT_COLORS = [
  { name: "indigo", class: "bg-indigo-500" },
  { name: "purple", class: "bg-purple-500" },
  { name: "blue", class: "bg-blue-500" },
  { name: "emerald", class: "bg-emerald-500" },
  { name: "amber", class: "bg-amber-500" },
] as const;

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const { theme, setTheme, accentColor, setAccentColor } = useThemeStore();
  const { addToast } = useToast();
  const [isSaving, setIsSaving] = useState(false);
  const [profile, setProfile] = useState<ProfileSettings>({
    fullName: "John Doe",
    email: "john@example.com",
    organization: "",
    role: "analyst",
  });
  const [notifications, setNotifications] = useState<NotificationSettings>({
    analysisComplete: true,
    highRiskAlert: true,
    weeklyDigest: false,
    securityUpdates: true,
  });

  const { data: healthData, isLoading: isLoadingHealth } = useQuery({
    queryKey: ["health", "detailed"],
    queryFn: () => healthApi.detailed(),
    staleTime: 60000,
  });

  const { data: configData, isLoading: isLoadingConfig } = useQuery({
    queryKey: ["config"],
    queryFn: () => healthApi.config(),
    staleTime: 300000,
  });

  const handleSave = async () => {
    setIsSaving(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    addToast("success", "Settings Saved", "Your preferences have been updated.");
    setIsSaving(false);
  };

  const handleProfileChange = (field: keyof ProfileSettings, value: string) => {
    setProfile((prev) => ({ ...prev, [field]: value }));
  };

  const handleNotificationChange = (field: keyof NotificationSettings) => {
    setNotifications((prev) => ({ ...prev, [field]: !prev[field] }));
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-3">
            <Settings size={24} className="text-indigo-600 dark:text-indigo-400" />
            Settings
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">
            Manage your account settings and preferences
          </p>
        </div>

        <Tabs defaultValue="profile">
          <TabList>
            <Tab value="profile">
              <User size={16} className="mr-2" />
              Profile
            </Tab>
            <Tab value="notifications">
              <Bell size={16} className="mr-2" />
              Notifications
            </Tab>
            <Tab value="appearance">
              <Palette size={16} className="mr-2" />
              Appearance
            </Tab>
            <Tab value="security">
              <Shield size={16} className="mr-2" />
              Security
            </Tab>
            <Tab value="api">
              <Key size={16} className="mr-2" />
              API Keys
            </Tab>
            <Tab value="data">
              <Database size={16} className="mr-2" />
              Data
            </Tab>
          </TabList>

          <TabPanel value="profile" className="mt-6">
            <Card className="p-6">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
                Profile Settings
              </h2>
              <div className="flex items-center gap-4 mb-6">
                <Avatar src={null} alt="Profile" fallback="JD" size="xl" />
                <div>
                  <Button variant="outline" size="sm">
                    Change Avatar
                  </Button>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                    JPG, GIF or PNG. Max size 2MB.
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="col-span-2 md:col-span-1">
                  <Input
                    label="Full Name"
                    placeholder="John Doe"
                    value={profile.fullName}
                    onChange={(e) => handleProfileChange("fullName", e.target.value)}
                  />
                </div>
                <div className="col-span-2 md:col-span-1">
                  <Input
                    label="Email"
                    type="email"
                    placeholder="john@example.com"
                    value={profile.email}
                    onChange={(e) => handleProfileChange("email", e.target.value)}
                  />
                </div>
                <div className="col-span-2 md:col-span-1">
                  <Input
                    label="Organization"
                    placeholder="Company Name"
                    value={profile.organization}
                    onChange={(e) => handleProfileChange("organization", e.target.value)}
                  />
                </div>
                <div className="col-span-2 md:col-span-1">
                  <Select
                    label="Role"
                    options={[
                      { value: "analyst", label: "Security Analyst" },
                      { value: "investigator", label: "Forensic Investigator" },
                      { value: "admin", label: "Administrator" },
                    ]}
                    value={profile.role}
                    onChange={(value) => handleProfileChange("role", value)}
                  />
                </div>
              </div>
              <div className="mt-6 flex justify-end">
                <Button variant="primary" onClick={handleSave} loading={isSaving}>
                  <Save size={16} className="mr-2" />
                  Save Changes
                </Button>
              </div>
            </Card>
          </TabPanel>

          <TabPanel value="notifications" className="mt-6">
            <Card className="p-6">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
                Notification Preferences
              </h2>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">Analysis Complete</p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Get notified when an analysis finishes
                    </p>
                  </div>
                  <Switch
                    checked={notifications.analysisComplete}
                    onChange={() => handleNotificationChange("analysisComplete")}
                  />
                </div>
                <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">High Risk Alert</p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Receive alerts for high-risk content
                    </p>
                  </div>
                  <Switch
                    checked={notifications.highRiskAlert}
                    onChange={() => handleNotificationChange("highRiskAlert")}
                  />
                </div>
                <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">Weekly Digest</p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Summary of weekly activity
                    </p>
                  </div>
                  <Switch
                    checked={notifications.weeklyDigest}
                    onChange={() => handleNotificationChange("weeklyDigest")}
                  />
                </div>
                <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">Security Updates</p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Important security announcements
                    </p>
                  </div>
                  <Switch
                    checked={notifications.securityUpdates}
                    onChange={() => handleNotificationChange("securityUpdates")}
                  />
                </div>
              </div>
            </Card>
          </TabPanel>

          <TabPanel value="appearance" className="mt-6">
            <Card className="p-6">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
                Appearance Settings
              </h2>
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                    Theme
                  </label>
                  <div className="flex gap-3">
                    {THEME_OPTIONS.map((option) => {
                      const Icon = option.icon;
                      return (
                        <button
                          key={option.value}
                          onClick={() => setTheme(option.value)}
                          className={`flex items-center gap-2 px-4 py-3 rounded-lg border-2 transition-colors capitalize ${
                            theme === option.value
                              ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-700 dark:text-indigo-300"
                              : "border-slate-200 dark:border-slate-700 hover:border-slate-300 text-slate-700 dark:text-slate-300"
                          }`}
                        >
                          <Icon size={18} />
                          {option.label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                    Accent Color
                  </label>
                  <div className="flex gap-3">
                    {ACCENT_COLORS.map((color) => (
                      <button
                        key={color.name}
                        onClick={() => setAccentColor(color.name)}
                        className={`w-10 h-10 rounded-full ${color.class} ring-2 ring-offset-2 ${
                          accentColor === color.name
                            ? "ring-indigo-500"
                            : "ring-transparent hover:ring-slate-300"
                        }`}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          </TabPanel>

          <TabPanel value="security" className="mt-6">
            <Card className="p-6">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
                Security Settings
              </h2>
              <div className="space-y-6">
                <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-slate-900 dark:text-white">
                        Two-Factor Authentication
                      </p>
                      <p className="text-sm text-slate-500 dark:text-slate-400">
                        Add an extra layer of security
                      </p>
                    </div>
                    <Badge variant="warning">Not Enabled</Badge>
                  </div>
                  <Button variant="outline" className="mt-4">
                    Enable 2FA
                  </Button>
                </div>

                <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-slate-900 dark:text-white">
                        Session Management
                      </p>
                      <p className="text-sm text-slate-500 dark:text-slate-400">
                        Manage active sessions
                      </p>
                    </div>
                    <Button variant="outline">
                      <RefreshCw size={16} className="mr-2" />
                      View Sessions
                    </Button>
                  </div>
                </div>

                <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-slate-900 dark:text-white">Password</p>
                      <p className="text-sm text-slate-500 dark:text-slate-400">
                        Last changed 30 days ago
                      </p>
                    </div>
                    <Button variant="outline">Change Password</Button>
                  </div>
                </div>
              </div>
            </Card>
          </TabPanel>

          <TabPanel value="api" className="mt-6">
            <Card className="p-6">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
                API Keys
              </h2>
              <div className="space-y-4">
                <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-slate-900 dark:text-white">Production Key</p>
                      <p className="text-sm text-slate-500 dark:text-slate-400 font-mono">
                        ak_••••••••••••••••
                      </p>
                    </div>
                    <Badge variant="success">Active</Badge>
                  </div>
                  <p className="text-xs text-slate-400 mt-2">Created on Jan 15, 2026</p>
                </div>

                <Button variant="outline">
                  <Key size={16} className="mr-2" />
                  Generate New Key
                </Button>
              </div>
            </Card>
          </TabPanel>

          <TabPanel value="data" className="mt-6">
            <Card className="p-6">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white mb-6">
                Data Management
              </h2>
              <div className="space-y-6">
                {isLoadingHealth || isLoadingConfig ? (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Array.from({ length: 3 }).map((_, index) => (
                      <Skeleton key={index} className="h-24 w-full rounded-lg" />
                    ))}
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card className="p-4 text-center">
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">
                        {healthData?.components?.database?.status === "healthy" ? "24" : "--"}
                      </p>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Total Analyses</p>
                    </Card>
                    <Card className="p-4 text-center">
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">
                        {configData?.maxFileSize
                          ? `${(configData.maxFileSize / 1024 / 1024).toFixed(0)} MB`
                          : "2.4 GB"}
                      </p>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Max File Size</p>
                    </Card>
                    <Card className="p-4 text-center">
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">7 days</p>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Retention Period</p>
                    </Card>
                  </div>
                )}

                <div className="flex gap-4">
                  <Button variant="outline">
                    <Download size={16} className="mr-2" />
                    Export All Data
                  </Button>
                  <Button variant="danger">
                    <Trash2 size={16} className="mr-2" />
                    Delete All Data
                  </Button>
                </div>
              </div>
            </Card>
          </TabPanel>
        </Tabs>
      </motion.div>
    </div>
  );
}
