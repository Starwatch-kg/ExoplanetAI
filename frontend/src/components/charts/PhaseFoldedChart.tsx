import { useMemo } from 'react'
// @ts-ignore
import Plot from 'react-plotly.js'
import { motion } from 'framer-motion'
import { LightCurveData, TransitCandidate } from '../../types/api'

interface PhaseFoldedChartProps {
  data: LightCurveData
  candidate: TransitCandidate
  height?: number
  className?: string
}

export default function PhaseFoldedChart({ 
  data, 
  candidate, 
  height = 400,
  className = '' 
}: PhaseFoldedChartProps) {
  
  const foldedData = useMemo(() => {
    // Phase fold the light curve
    const period = candidate.period
    const epoch = candidate.epoch
    
    const phases = data.time.map(t => {
      let phase = ((t - epoch) / period) % 1
      if (phase < -0.5) phase += 1
      if (phase > 0.5) phase -= 1
      return phase
    })
    
    // Sort by phase
    const sortedIndices = phases
      .map((phase, index) => ({ phase, index }))
      .sort((a, b) => a.phase - b.phase)
    
    const sortedPhases = sortedIndices.map(item => item.phase)
    const sortedFlux = sortedIndices.map(item => data.flux[item.index])
    
    return { phases: sortedPhases, flux: sortedFlux }
  }, [data, candidate])
  
  const plotData = useMemo(() => {
    const traces: any[] = []
    
    // Phase-folded light curve
    traces.push({
      x: foldedData.phases,
      y: foldedData.flux,
      type: 'scattergl',
      mode: 'markers',
      marker: {
        size: 3,
        color: '#60a5fa',
        opacity: 0.6,
      },
      name: 'Сложенная кривая',
      hovertemplate: 'Фаза: %{x:.3f}<br>Поток: %{y:.6f}<extra></extra>',
    })
    
    // Binned data for better visualization
    const binSize = 0.01
    const bins: { [key: string]: number[] } = {}
    
    foldedData.phases.forEach((phase, index) => {
      const binCenter = Math.round(phase / binSize) * binSize
      const binKey = binCenter.toFixed(3)
      
      if (!bins[binKey]) {
        bins[binKey] = []
      }
      bins[binKey].push(foldedData.flux[index])
    })
    
    const binnedPhases: number[] = []
    const binnedFlux: number[] = []
    const binnedErrors: number[] = []
    
    Object.entries(bins).forEach(([binKey, fluxValues]) => {
      if (fluxValues.length >= 3) { // Only include bins with enough points
        const phase = parseFloat(binKey)
        const meanFlux = fluxValues.reduce((sum, val) => sum + val, 0) / fluxValues.length
        const stdFlux = Math.sqrt(
          fluxValues.reduce((sum, val) => sum + Math.pow(val - meanFlux, 2), 0) / fluxValues.length
        )
        
        binnedPhases.push(phase)
        binnedFlux.push(meanFlux)
        binnedErrors.push(stdFlux / Math.sqrt(fluxValues.length))
      }
    })
    
    // Binned data trace
    traces.push({
      x: binnedPhases,
      y: binnedFlux,
      error_y: {
        type: 'data',
        array: binnedErrors,
        visible: true,
        color: '#ec4899',
        thickness: 1,
        width: 3,
      },
      type: 'scatter',
      mode: 'markers',
      marker: {
        size: 6,
        color: '#ec4899',
        opacity: 0.8,
      },
      name: 'Усредненные данные',
      hovertemplate: 'Фаза: %{x:.3f}<br>Поток: %{y:.6f}<br>Ошибка: ±%{error_y.array:.6f}<extra></extra>',
    })
    
    // Transit model (simple box model)
    const transitPhases = []
    const transitModel = []
    const transitDuration = candidate.duration / candidate.period // Duration in phase units
    
    for (let phase = -0.5; phase <= 0.5; phase += 0.001) {
      transitPhases.push(phase)
      
      if (Math.abs(phase) < transitDuration / 2) {
        transitModel.push(1 - candidate.depth)
      } else {
        transitModel.push(1)
      }
    }
    
    traces.push({
      x: transitPhases,
      y: transitModel,
      type: 'scatter',
      mode: 'lines',
      line: {
        color: '#f59e0b',
        width: 3,
      },
      name: 'Модель транзита',
      hovertemplate: 'Фаза: %{x:.3f}<br>Модель: %{y:.6f}<extra></extra>',
    })
    
    return traces
  }, [foldedData, candidate])
  
  const layout = useMemo(() => ({
    title: {
      text: `Сложенная по фазе кривая (P = ${candidate.period.toFixed(3)} дней)`,
      font: { color: '#ffffff', size: 16 },
    },
    xaxis: {
      title: 'Фаза',
      color: '#ffffff',
      gridcolor: '#374151',
      zerolinecolor: '#4b5563',
      range: [-0.5, 0.5],
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
    shapes: [
      {
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: -candidate.duration / candidate.period / 2,
        y0: 0,
        x1: candidate.duration / candidate.period / 2,
        y1: 1,
        fillcolor: 'rgba(236, 72, 153, 0.1)',
        line: { width: 0 },
      }
    ],
  }), [candidate])
  
  const config = useMemo(() => ({
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: {
      format: 'png' as const,
      filename: `phase_folded_${data.target_name}_P${candidate.period.toFixed(3)}`,
      height: height,
      width: 800,
      scale: 2,
    },
    responsive: true,
  }), [data.target_name, candidate.period, height])
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className={`bg-space-800/50 backdrop-blur-sm rounded-xl border border-space-600 p-4 ${className}`}
    >
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: `${height}px` }}
        useResizeHandler={true}
      />
      
      {/* Transit parameters */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="text-center">
          <p className="text-gray-400">Период</p>
          <p className="text-white font-semibold">{candidate.period.toFixed(3)} дней</p>
        </div>
        <div className="text-center">
          <p className="text-gray-400">Глубина</p>
          <p className="text-white font-semibold">{(candidate.depth * 1e6).toFixed(0)} ppm</p>
        </div>
        <div className="text-center">
          <p className="text-gray-400">Длительность</p>
          <p className="text-white font-semibold">{(candidate.duration * 24).toFixed(1)} часов</p>
        </div>
        <div className="text-center">
          <p className="text-gray-400">SNR</p>
          <p className="text-white font-semibold">{candidate.snr.toFixed(1)}</p>
        </div>
      </div>
    </motion.div>
  )
}
