import { create } from 'zustand';

interface UIState {
  theme: 'light' | 'dark' | 'system';
  accentColor: string;
  sidebarOpen: boolean;
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  setAccentColor: (color: string) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
}

export const useUIStore = create<UIState>((set) => ({
  theme: 'system',
  accentColor: 'indigo',
  sidebarOpen: true,
  setTheme: (theme) => set({ theme }),
  setAccentColor: (accentColor) => set({ accentColor }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
}));

export function useThemeStore() {
  return useUIStore((state) => ({
    theme: state.theme,
    setTheme: state.setTheme,
    accentColor: state.accentColor,
    setAccentColor: state.setAccentColor,
  }));
}
