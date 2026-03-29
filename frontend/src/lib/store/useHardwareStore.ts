import { create } from "zustand";

/**
 * Transient Hardware Store
 * Specifically engineered to bypass standard React re-renders. 
 * High frequency components (like the ECharts Oscilloscope) must use 
 * `useHardwareStore.getState()` via a requestAnimationFrame loop instead of 
 * binding via standard hook reactions to avoid locking the DOM at 20Hz.
 */

// Define the exact payload matching the Backend Python `main.py` state
export interface HardwareState {
  raw_adc: number;
  electrode_a: number;
  electrode_b: number;
  target_adc: number;
  system_mode: "auto" | "manual" | "emergency";
}

interface HardwareStore extends HardwareState {
  // We keep a rolling history for the charts
  history: {
    time: number;
    raw_adc: number;
    target_adc: number;
    electrode_a: number;
    electrode_b: number;
  }[];
  // Direct setter for the WebSocket
  updateHardwareState: (incoming: Partial<HardwareState>) => void;
  // Emergency specific override
  triggerEmergencyStop: () => void;
}

export const useHardwareStore = create<HardwareStore>((set) => ({
  raw_adc: 0,
  electrode_a: 2048,
  electrode_b: 2048,
  target_adc: 750,
  system_mode: "auto",
  history: [],

  updateHardwareState: (incoming) =>
    set((state) => {
      // Append to rolling history
      const now = Date.now();
      const newHistory = [
        ...state.history,
        {
          time: now,
          raw_adc: incoming.raw_adc ?? state.raw_adc,
          target_adc: incoming.target_adc ?? state.target_adc,
          electrode_a: incoming.electrode_a ?? state.electrode_a,
          electrode_b: incoming.electrode_b ?? state.electrode_b,
        },
      ];

      // Keep only last ~600 frames (roughly 30 seconds at 20Hz)
      if (newHistory.length > 600) {
        newHistory.shift();
      }

      return {
        ...incoming,
        history: newHistory,
      };
    }),

  triggerEmergencyStop: () =>
    set({
      system_mode: "emergency",
      electrode_a: 2048, // Return to center/safe voltages immediately
      electrode_b: 2048,
    }),
}));
