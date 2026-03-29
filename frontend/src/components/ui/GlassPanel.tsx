import * as React from "react"
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

const GlassPanel = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      // The Synthetic Observer Core Token: No Solid Borders.
      "bg-panel/60 backdrop-blur-xl",
      // Tonal separation via outline-variant (20% opacity ghost border)
      "ring-1 ring-slate-700/20",
      "text-slate-200",
      className
    )}
    {...props}
  />
))
GlassPanel.displayName = "GlassPanel"

export { GlassPanel }
