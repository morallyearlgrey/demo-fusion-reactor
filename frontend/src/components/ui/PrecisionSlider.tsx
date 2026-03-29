import * as React from "react"
import { cn } from "./GlassPanel"

interface PrecisionSliderProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  value: number;
  maxVoltage?: number;
}

export const PrecisionSlider = React.forwardRef<HTMLInputElement, PrecisionSliderProps>(
  ({ className, label, value, maxVoltage = 12.00, ...props }, ref) => {
    
    // Calculate percentage purely for the gradient track visual
    const percentage = (value / maxVoltage) * 100;

    return (
      <div className={cn("flex flex-col gap-2 w-full group", className)}>
        <div className="flex justify-between items-end font-mono text-sm">
          <span className="text-slate-400 capitalize tracking-wider">{label}</span>
          <span className="text-cyan font-bold tracking-widest">{value.toFixed(2)}V</span>
        </div>
        
        <div className="relative h-6 w-full flex items-center">
          {/* Custom Webkit Slider Replacement */}
          <input
            type="range"
            ref={ref}
            value={value}
            min={0}
            max={maxVoltage}
            step={0.01}
            className="w-full appearance-none bg-transparent focus:outline-none focus:ring-0 z-10 cursor-ew-resize [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:h-8 [&::-webkit-slider-thumb]:bg-cyan [&::-webkit-slider-thumb]:blur-[1px] [&::-webkit-slider-thumb]:shadow-[0_0_12px_#66FCF1]"
            style={{
              // This dynamically controls the track glow based on position
              background: `linear-gradient(to right, rgba(0, 163, 255, 0.4) ${percentage}%, rgba(24, 28, 34, 1) ${percentage}%)`
            }}
            {...props}
          />
        </div>
      </div>
    )
  }
)
PrecisionSlider.displayName = "PrecisionSlider"
