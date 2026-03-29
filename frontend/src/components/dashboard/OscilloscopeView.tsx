"use client"
import React, { useRef, useEffect } from "react";
import ReactECharts from "echarts-for-react";
import { useHardwareStore } from "@/lib/store/useHardwareStore";

export function OscilloscopeView() {
  const chartRef = useRef<ReactECharts>(null);

  useEffect(() => {
    // 60FPS RAF Loop pulling from Transient Store
    let animationFrameId: number;
    
    const updateChart = () => {
      const history = useHardwareStore.getState().history;
      const chartInstance = chartRef.current?.getEchartsInstance();
      
      if (chartInstance && history.length > 0) {
        chartInstance.setOption({
          series: [
            {
              name: 'RAW_ADC',
              data: history.map(d => [d.time, d.raw_adc])
            },
            {
              name: 'TARGET',
              data: history.map(d => [d.time, d.target_adc])
            }
          ]
        });
      }
      
      animationFrameId = requestAnimationFrame(updateChart);
    };

    updateChart();
    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  const baseOptions = {
    animation: false,
    tooltip: { trigger: 'axis', axisPointer: { animation: false } },
    grid: { top: 20, right: 30, bottom: 30, left: 60 },
    xAxis: {
      type: 'time',
      splitLine: { show: false },
      axisLabel: { color: '#88919d' }
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1023,
      splitLine: { 
        lineStyle: { color: '#181c22' } 
      },
      axisLabel: { color: '#88919d', fontFamily: 'monospace' }
    },
    series: [
      {
        name: 'RAW_ADC',
        type: 'line',
        showSymbol: false,
        lineStyle: { width: 3, color: '#66FCF1', shadowColor: '#66FCF1', shadowBlur: 10 },
        data: []
      },
      {
        name: 'TARGET',
        type: 'line',
        showSymbol: false,
        lineStyle: { type: 'dashed', width: 2, color: '#cc9200', opacity: 0.8 },
        data: []
      }
    ]
  };

  return (
    <div className="w-full h-full min-h-[400px]">
      <ReactECharts 
        ref={chartRef} 
        option={baseOptions}
        style={{ height: '100%', width: '100%' }}
        opts={{ renderer: 'webgl' }}
      />
    </div>
  );
}
