"use client"
import React, { useEffect, useRef, useState } from "react"
import { useAgentStore } from "@/lib/store/useAgentStore"
import { GlassPanel } from "../ui/GlassPanel"
import { Terminal, Bot, Zap, ShieldAlert, Cpu, Volume2, VolumeX } from "lucide-react"
import { AGENT_VOICES, AgentType } from "@/lib/agents/voices"

export function CyberCoreLog() {
  const logs = useAgentStore((state) => state.logs)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [audioEnabled, setAudioEnabled] = useState(true)
  const audioQueue = useRef<string[]>([])
  const isPlaying = useRef(false)

  // Auto scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  // Process the ElevenLabs audio queue
  const processQueue = async () => {
    if (isPlaying.current || audioQueue.current.length === 0 || !audioEnabled) return;

    isPlaying.current = true;
    const { text, agent } = JSON.parse(audioQueue.current.shift()!);
    const voiceData = AGENT_VOICES[agent as AgentType] || AGENT_VOICES.System;

    try {
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          voiceId: voiceData.voiceId,
          settings: voiceData.settings
        })
      });

      if (response.ok) {
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          isPlaying.current = false;
          processQueue(); // Look for next one
        };

        audio.play();
      } else {
        console.error("TTS Fetch Failed");
        isPlaying.current = false;
        processQueue();
      }
    } catch (err) {
      console.error("TTS Error:", err);
      isPlaying.current = false;
      processQueue();
    }
  };

  // Trigger TTS on new logs
  useEffect(() => {
    if (logs.length === 0) return;
    const latest = logs[logs.length - 1];
    
    // Add to queue if enabled
    if (audioEnabled && latest.message) {
      audioQueue.current.push(JSON.stringify({ text: latest.message, agent: latest.agent }));
      processQueue();
    }
  }, [logs])

  const getAgentIcon = (agent: string) => {
    switch (agent) {
      case "AnalyzeAgent": return <Zap className="w-4 h-4 text-amber" />
      case "DecisionAgent": return <Bot className="w-4 h-4 text-cyan" />
      case "ActionAgent": return <ShieldAlert className="w-4 h-4 text-danger animate-pulse-fast" />
      case "ImprovementAgent": return <Cpu className="w-4 h-4 text-violet-400" />
      default: return <Terminal className="w-4 h-4 text-slate-400" />
    }
  }

  const getAgentColor = (agent: string) => {
    switch (agent) {
      case "AnalyzeAgent": return "text-amber"
      case "DecisionAgent": return "text-cyan"
      case "ActionAgent": return "text-danger"
      case "ImprovementAgent": return "text-violet-400"
      default: return "text-slate-400"
    }
  }

  return (
    <GlassPanel className="h-full flex flex-col overflow-hidden">
      <div className="p-4 border-b border-white/5 flex items-center justify-between">
        <h3 className="font-space uppercase tracking-widest text-sm text-slate-400 font-bold flex items-center gap-2">
          <Terminal className="w-4 h-4" /> Agent Pipeline
        </h3>
        <button 
          onClick={() => setAudioEnabled(!audioEnabled)}
          className={`p-2 rounded-sm transition-colors ${audioEnabled ? 'bg-cyan/10 text-cyan' : 'bg-white/5 text-slate-500'}`}
          title={audioEnabled ? "Disable Voice Engine" : "Enable Voice Engine"}
        >
          {audioEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-xs">
        {logs.map((log) => (
          <div 
            key={log.id} 
            className={`pl-3 border-l-2 transition-all duration-300 ${
              log.confidence && log.confidence > 0.9 
              ? "border-cyan bg-cyan/5 drop-shadow-[0_0_15px_rgba(0,163,255,0.2)]" 
              : "border-slate-700 hover:border-slate-500"
            }`}
          >
            <div className="flex items-center gap-2 mb-1 opacity-60">
              {getAgentIcon(log.agent)}
              <span className={`font-bold tracking-wider ${getAgentColor(log.agent)}`}>
                {log.agent}
              </span>
              <span className="text-slate-500 ml-auto">
                {new Date(log.timestamp).toLocaleTimeString('en-US', {hour12:false, minute:'2-digit', second:'2-digit', fractionalSecondDigits: 3})}
              </span>
            </div>
            
            <p className="text-slate-300 leading-relaxed mb-2">
              "{log.message}"
            </p>
            
            {log.actionPayload && (
               <div className="bg-void/50 p-2 rounded-sm border border-white/5 text-slate-400 overflow-x-auto">
                 <pre>{JSON.stringify(log.actionPayload, null, 2)}</pre>
               </div>
            )}
            
            {log.confidence && log.confidence < 0.6 && (
              <span className="inline-block mt-2 text-[10px] text-amber px-1 border border-amber/30 bg-amber/10 rounded-sm uppercase">
                Low Confidence Warning ({Math.round(log.confidence * 100)}%)
              </span>
            )}
          </div>
        ))}
      </div>
    </GlassPanel>
  )
}
