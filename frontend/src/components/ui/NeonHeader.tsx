import * as React from "react"
import { cn } from "./GlassPanel"

interface NeonHeaderProps extends React.HTMLAttributes<HTMLHeadingElement> {
  level?: 1 | 2 | 3 | 4;
}

export function NeonHeader({ className, level = 2, children, ...props }: NeonHeaderProps) {
  const Comp = `h${level}` as const
  
  return (
    <Comp
      className={cn(
        "font-space uppercase tracking-[0.15em] font-medium text-cyan drop-shadow-[0_0_8px_rgba(0,163,255,0.4)]",
        level === 1 && "text-4xl",
        level === 2 && "text-xl",
        level === 3 && "text-lg",
        level === 4 && "text-sm",
        className
      )}
      {...props}
    >
      {children}
    </Comp>
  )
}
