import { useMemo } from 'react'
// @ts-ignore
import Plot from 'react-plotly.js'
import { motion } from 'framer-motion'
import { LightCurveData, TransitCandidate } from '../../types/api'

interface LightCurveChartProps {
  data: LightCurveData
  candidates?: TransitCandidate[]
  showCandidates?: boolean
  height?: number
  className?: string
}

export default function LightCurveChart({ 
  data, 
  candidates = [], 
  showCandidates = true, 
  height = 400,
  className = '' 
}: LightCurveChartProps) {
  
  const plotData = useMemo(() => {
    const traces: any[] = []
    
    // Main light curve
    traces.push({
      x: data.time,
      y: data.flux,
      type: 'scattergl',
      mode: 'markers',
      marker: {
        size: 2,
        color: '#60a5fa',
        opacity: 0.7,
      },
      name: 'Кривая блеска',
      hovertemplate: 'Время: %{x:.3f} дней<br>Поток: %{y:.6f}<extra></extra>',
    })
    
    // Error bars if available
    if (data.flux_err) {
      traces.push({
        x: data.time,
        y: data.flux,
        error_y: {
          type: 'data',
          array: data.flux_err,
          visible: true,
          color: '#60a5fa',
          thickness: 0.5,
          width: 0,
        },
        type: 'scatter',
        mode: 'markers',
        marker: { opacity: 0 },
        showlegend: false,
        hoverinfo: 'skip',
      })
    }
    
    // Transit candidates
    if (showCandidates && candidates.length > 0) {
      candidates.forEach((candidate, index) => {
        const transitTimes = []
        const transitFluxes = []
        
        // Calculate transit times based on period and epoch
        const startTime = Math.min(...data.time)
        const endTime = Math.max(...data.time)
        
        let transitTime = candidate.epoch
        while (transitTime <= endTime) {
          if (transitTime >= startTime) {
            // Find closest flux value
            const closestIndex = data.time.reduce((prev, curr, idx) => 
              Math.abs(curr - transitTime) < Math.abs(data.time[prev] - transitTime) ? idx : prev, 0
            )
            
            transitTimes.push(transitTime)
            transitFluxes.push(data.flux[closestIndex])
          }
          transitTime += candidate.period
        }
        
        if (transitTimes.length > 0) {
          traces.push({
            x: transitTimes,
            y: transitFluxes,
            type: 'scatter',
            mode: 'markers',
            marker: {
              size: 8,
              color: index === 0 ? '#ec4899' : '#f59e0b',
              symbol: 'diamond',
              line: {
                width: 2,
                color: '#ffffff',
              },
            },
            name: `Кандидат ${index + 1} (P=${candidate.period.toFixed(2)}д)`,
            hovertemplate: `
              <b>Транзитный кандидат ${index + 1}</b><br>
              Время: %{x:.3f} дней<br>
              Поток: %{y:.6f}<br>
              Период: ${candidate.period.toFixed(3)} дней<br>
              Глубина: ${(candidate.depth * 1e6).toFixed(0)} ppm<br>
              SNR: ${candidate.snr.toFixed(1)}<br>
              Уверенность: ${(candidate.confidence * 100).toFixed(1)}%
              <extra></extra>
            `,
          })
        }
      })
    }
    
    return traces
  }, [data, candidates, showCandidates])
  
  const layout = useMemo(() => ({
    title: {
      text: `Кривая блеска: ${data.target_name}`,
      font: { color: '#ffffff', size: 16 },
    },
    xaxis: {
      title: 'Время (дни)',
      color: '#ffffff',
      gridcolor: '#374151',
      zerolinecolor: '#4b5563',
    },
    yaxis: {
      title: 'Нормализованный поток',
      color: '#ffffff',
      gridcolor: '#374151',
      zerolinecolor: '#4b5563',
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#ffffff' },
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(15, 15, 35, 0.8)',
      bordercolor: '#374151',
      borderwidth: 1,
    },
    margin: { l: 60, r: 20, t: 60, b: 60 },
    hovermode: 'closest',
  }), [data.target_name])
  
  const config = useMemo(() => ({
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: {
      format: 'png' as const,
      filename: `lightcurve_${data.target_name}`,
      height: height,
      width: 800,
      scale: 2,
    },
    responsive: true,
  }), [data.target_name, height])
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`bg-space-800/50 backdrop-blur-sm rounded-xl border border-space-600 p-4 ${className}`}
    >
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: `${height}px` }}
        useResizeHandler={true}
      />
      
      {/* Chart info */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="text-center">
          <p className="text-gray-400">Точек данных</p>
          <p className="text-white font-semibold">{data.time.length.toLocaleString()}</p>
        </div>
        <div className="text-center">
          <p className="text-gray-400">Миссия</p>
          <p className="text-white font-semibold">{data.mission}</p>
        </div>
        <div className="text-center">
          <p className="text-gray-400">Период наблюдений</p>
          <p className="text-white font-semibold">
            {(Math.max(...data.time) - Math.min(...data.time)).toFixed(1)} дней
          </p>
        </div>
        <div className="text-center">
          <p className="text-gray-400">Кандидатов</p>
          <p className="text-white font-semibold">{candidates.length}</p>
        </div>
      </div>
    </motion.div>
  )
}
