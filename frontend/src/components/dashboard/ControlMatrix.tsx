"use client"
import React, { useEffect, useState } from "react"
import { GlassPanel } from "../ui/GlassPanel"
import { PrecisionSlider } from "../ui/PrecisionSlider"
import { useHardwareStore } from "@/lib/store/useHardwareStore"
import { ShieldAlert, Cpu } from "lucide-react"

export function ControlMatrix() {
  const triggerEmergency = useHardwareStore((state) => state.triggerEmergencyStop)
  const systemMode = useHardwareStore((state) => state.system_mode)
  
  // Since sliders are driven by hardware in AUTO, but human in MANUAL,
  // we need a local mirrored state to allow fluid CSS dragging 
  const [localA, setLocalA] = useState(2048)
  const [localB, setLocalB] = useState(2048)

  // Map arbitrary 0-4095 DAC integer range to human readable 0-12V 
  const dacToVolts = (dac: number) => (dac / 4095) * 12.0
  const voltsToDac = (volts: number) => Math.floor((volts / 12.0) * 4095)

  // Sync component local state with global store ONLY if system is automatically moving them
  useEffect(() => {
    const unsub = useHardwareStore.subscribe((state) => {
       if (state.system_mode !== "manual") {
          setLocalA(state.electrode_a)
          setLocalB(state.electrode_b)
       }
    })
    return unsub
  }, [])

  return (
    <GlassPanel className="p-6">
      <div className="flex justify-between items-center mb-6">
         <h2 className="font-space uppercase text-slate-300 font-bold tracking-widest flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan" /> Reticle Manual Override
         </h2>
         
         <div className="flex items-center gap-3">
            <span className={`font-mono text-xs px-3 py-1 border rounded-sm uppercase tracking-wider ${
               systemMode === 'emergency' 
               ? 'text-danger border-danger/30 bg-danger/10' 
               : systemMode === 'auto' 
                 ? 'text-cyan border-cyan/30 bg-cyan/10'
                 : 'text-amber border-amber/30 bg-amber/10'
            }`}>
              MODE: {systemMode}
            </span>
         </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-center cursor-default">
         <div className="md:col-span-2 space-y-8">
            <PrecisionSlider 
               label="Electrode X-AXIS (A)"
               value={dacToVolts(localA)}
               onChange={(e) => {
                 const volts = parseFloat(e.target.value);
                 setLocalA(voltsToDac(volts));
                 if(systemMode === 'manual') useHardwareStore.getState().updateHardwareState({ electrode_a: voltsToDac(volts) });
               }}
               // Lock sliders if the AI is actively controlling the system
               disabled={systemMode !== 'manual'}
               className={systemMode !== 'manual' ? 'opacity-50 grayscale transition-all' : ''}
            />
            
            <PrecisionSlider 
               label="Electrode Y-AXIS (B)"
               value={dacToVolts(localB)}
               onChange={(e) => {
                 const volts = parseFloat(e.target.value);
                 setLocalB(voltsToDac(volts));
                 if(systemMode === 'manual') useHardwareStore.getState().updateHardwareState({ electrode_b: voltsToDac(volts) });
               }}
               disabled={systemMode !== 'manual'}
               className={systemMode !== 'manual' ? 'opacity-50 grayscale transition-all' : ''}
            />
         </div>

         {/* High Stakes E-STOP Button */}
         <div className="flex items-center justify-center p-4">
            <button 
               onClick={triggerEmergency}
               className="group relative w-full aspect-square max-w-[180px] rounded-sm bg-[#181c22] border-t border-t-slate-700/50 flex flex-col items-center justify-center gap-2 transition-all active:scale-[0.98] drop-shadow-[0_12px_24px_rgba(255,0,60,0.15)] hover:drop-shadow-[0_12px_32px_rgba(255,0,60,0.3)] disabled:opacity-50"
               disabled={systemMode === 'emergency'}
            >
               <ShieldAlert className="w-12 h-12 text-danger group-active:scale-90 transition-transform" />
               <span className="font-space uppercase tracking-widest text-danger font-bold text-lg mt-2 group-active:text-danger/70">
                 E-STOP
               </span>
               <span className="font-mono text-[10px] text-slate-500 absolute bottom-3 uppercase">Drop Zero-Volt Interlock</span>
            </button>
         </div>
      </div>
    </GlassPanel>
  )
}
