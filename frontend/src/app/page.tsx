"use client"
import React, { useEffect } from "react"
import { OscilloscopeView } from "@/components/dashboard/OscilloscopeView"
import { CyberCoreLog } from "@/components/dashboard/CyberCoreLog"
import { ControlMatrix } from "@/components/dashboard/ControlMatrix"
import { GlassPanel } from "@/components/ui/GlassPanel"
import { NeonHeader } from "@/components/ui/NeonHeader"
import { useHardwareStore } from "@/lib/store/useHardwareStore"
import { useAgentStore } from "@/lib/store/useAgentStore"
import { Target, Activity } from "lucide-react"

export default function MissionControlDashboard() {
  const systemMode = useHardwareStore((state) => state.system_mode)

  // MOCK HARDWARE WEBSOCKET ENGINE (20Hz)
  useEffect(() => {
    let mockTick = 0;
    const intervalId = setInterval(() => {
      if (useHardwareStore.getState().system_mode === 'emergency') return;

      mockTick++;
      const baseSignal = 700 + Math.sin(mockTick * 0.1) * 200; // Fake Gaussian wave
      
      // Arc spikes mathematically simulated
      const isSpike = Math.random() > 0.98;
      const rawADC = isSpike ? baseSignal + 300 : baseSignal + (Math.random() * 20 - 10);

      useHardwareStore.getState().updateHardwareState({
        raw_adc: Math.floor(rawADC),
        target_adc: 750
      });
    }, 50); // 20Hz polling loop

    return () => clearInterval(intervalId);
  }, []);

  // MOCK LLM WEBSOCKET ENGINE (1Hz)
  useEffect(() => {
    const aiIntervalId = setInterval(() => {
        if (useHardwareStore.getState().system_mode === 'emergency') return;
        
        const randomAction = Math.random();
        
        if (randomAction > 0.8) {
           useAgentStore.getState().addLog({
             agent: "AnalyzeAgent",
             message: "Beam intensity dropping. Warning: Possible arc discharge precursor.",
             confidence: 0.85
           });
        } else if (randomAction > 0.6) {
           useAgentStore.getState().addLog({
             agent: "DecisionAgent",
             message: "Voltage delta calculated. Interpolating optimal path to target_adc.",
             actionPayload: { action: "increase_both", delta: 125 },
             confidence: 0.92
           });
           
           // Automatically move sliders to prove store sync works
           const currentA = useHardwareStore.getState().electrode_a;
           useHardwareStore.getState().updateHardwareState({ 
             electrode_a: Math.min(4095, currentA + 125),
             electrode_b: Math.min(4095, currentA + 125) 
           });
           
        } else if (randomAction > 0.4) {
           // Simulate the Safety Clamp
           useAgentStore.getState().addLog({
             agent: "ActionAgent",
             message: "Movement block clamped to safe physical boundary (500mV max delta). Executing.",
             confidence: 0.99
           });
        }
    }, 2000); // Trigger every 2s to not overwhelm the TTS queue
    
    return () => clearInterval(aiIntervalId);
  }, []);

  return (
    <div className="flex-1 p-8 grid grid-cols-1 xl:grid-cols-[1fr_400px] gap-8 max-w-[2000px] mx-auto w-full">
      
      {/* LEFT COLUMN: Physical Hardware Visualizations */}
      <div className="flex flex-col gap-8 h-full">
        {/* Top Nav Status Bar */}
        <header className="flex items-center justify-between col-span-2 pb-4 border-b border-white/5">
          <div>
            <NeonHeader level={1} className="flex items-center gap-3">
              <Activity className="text-cyan animate-pulse" /> FUSION_CORE_OS
            </NeonHeader>
            <p className="text-slate-500 font-mono text-xs tracking-widest mt-2 uppercase">Subatomic Observer Protocol [Active]</p>
          </div>

          <div className="flex gap-4">
            <div className={`px-4 py-2 border flex items-center gap-2 rounded-sm font-space uppercase tracking-widest text-sm font-bold ${
              systemMode === 'emergency' ? 'border-danger/50 text-danger bg-danger/10 shadow-[0_0_15px_rgba(255,0,60,0.2)]'
              : 'border-cyan/50 text-cyan bg-cyan/10 shadow-[0_0_15px_rgba(0,163,255,0.2)]'
            }`}>
              <div className={`w-2 h-2 rounded-full ${systemMode === 'emergency' ? 'bg-danger' : 'bg-cyan animate-pulse'}`} />
              {systemMode === 'emergency' ? 'SYSTEM LOCKDOWN' : 'OPTIMIZING (GEN 12)'}
            </div>
            
            <div className="px-4 py-2 border border-amber/30 text-amber bg-amber/10 rounded-sm font-space uppercase tracking-widest text-sm font-bold flex items-center gap-2">
              <Target className="w-4 h-4" /> TRG: 750 ADC
            </div>
          </div>
        </header>

        {/* ECharts Viewport */}
        <GlassPanel className="flex-1 min-h-[500px] p-6 flex flex-col relative overflow-hidden">
          <div className="flex justify-between items-center mb-6">
            <h2 className="font-space text-slate-400 font-bold uppercase tracking-widest">Live Oscilloscope Telemetry</h2>
            <div className="flex gap-4 font-mono text-xs">
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-cyan shadow-[0_0_5px_#66FCF1]"></span> [RAW_ADC]
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 border-t border-dashed border-amber"></span> [TARGET]
              </div>
            </div>
          </div>
          <div className="flex-1 w-full bg-[#0a0c10]/50 rounded border border-white/5 relative z-10">
             <OscilloscopeView />
          </div>
        </GlassPanel>

        {/* Control Desk */}
        <ControlMatrix />
      </div>

      {/* RIGHT COLUMN: AI Agent Analytics */}
      <div className="flex flex-col h-full pl-0 xl:pl-4 xl:border-l xl:border-white/5">
         <CyberCoreLog />
      </div>

    </div>
  )
}
