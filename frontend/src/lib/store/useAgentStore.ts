import { create } from "zustand";

/**
 * Episodic Agent Store
 * Operates at roughly 1Hz. Since it updates slowly, React components 
 * CAN safely react to this hook to trigger DOM re-renders (like scrolling a chat feed).
 */

export interface AgentLogEntry {
  id: string;
  agent: "AnalyzeAgent" | "DecisionAgent" | "ActionAgent" | "ImprovementAgent" | "System";
  timestamp: number;
  message: string;
  actionPayload?: any;
  confidence?: number;
}

interface AgentStore {
  logs: AgentLogEntry[];
  addLog: (entry: Omit<AgentLogEntry, "id" | "timestamp">) => void;
  clearLogs: () => void;
}

export const useAgentStore = create<AgentStore>((set) => ({
  logs: [],

  addLog: (entry) =>
    set((state) => {
      const newLog: AgentLogEntry = {
        ...entry,
        id: Math.random().toString(36).substring(7),
        timestamp: Date.now(),
      };

      // Keep last 100 messages to prevent infinite DOM node accumulation
      const newLogs = [...state.logs, newLog];
      if (newLogs.length > 100) {
        newLogs.shift();
      }

      return { logs: newLogs };
    }),

  clearLogs: () => set({ logs: [] }),
}));
