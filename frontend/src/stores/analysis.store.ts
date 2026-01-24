import { create } from 'zustand';

interface AnalysisState {
  currentAnalysis: any | null;
  history: any[];
  setCurrentAnalysis: (analysis: any) => void;
  clearCurrentAnalysis: () => void;
  addToHistory: (analysis: any) => void;
}

export const useAnalysisStore = create<AnalysisState>((set) => ({
  currentAnalysis: null,
  history: [],
  setCurrentAnalysis: (analysis) => set({ currentAnalysis: analysis }),
  clearCurrentAnalysis: () => set({ currentAnalysis: null }),
  addToHistory: (analysis) => set((state) => ({ history: [...state.history, analysis] })),
}));
