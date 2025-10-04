import React, { useMemo, memo } from 'react'
import Plot from 'react-plotly.js'
// import { useTranslation } from 'react-i18next' // Для будущих переводов
import { useTheme } from '../../contexts/ThemeContext'
import { BarChart3, Download, ZoomIn, Info } from 'lucide-react'

interface LightCurveData {
  time: number[]
  flux: number[]
  flux_err?: number[]
}

interface LightCurveGraphProps {
  data: LightCurveData
  targetName: string
  period?: number
  depth?: number
  duration?: number
  dataPoints: number
  observationDays: number
  className?: string
}

const LightCurveGraph: React.FC<LightCurveGraphProps> = memo(({
  data,
  targetName,
  period,
  depth,
  duration,
  dataPoints,
  observationDays,
  className = ''
}) => {
  // const { t } = useTranslation() // Будет использоваться для переводов
  const { theme } = useTheme()

  // Оптимизация: ограничиваем количество точек для отображения
  const optimizedData = useMemo(() => {
    if (data.time.length <= 10000) {
      // Если точек мало, используем все данные
      return data;
    }
    
    // Для больших массивов данных используем децимацию
    const decimationFactor = Math.ceil(data.time.length / 10000);
    const indices = [];
    
    for (let i = 0; i < data.time.length; i += decimationFactor) {
      indices.push(i);
    }
    
    return {
      time: indices.map(i => data.time[i]),
      flux: indices.map(i => data.flux[i]),
      flux_err: data.flux_err ? indices.map(i => data.flux_err![i]) : undefined
    };
  }, [data]);

  // Generate transit highlights if we have period data
  const transitHighlights = useMemo(() => {
    if (!period || !duration) return []
    
    const highlights: Array<{ start: number; end: number }> = []
    const transitDurationDays = duration / 24 // Convert hours to days
    const maxTime = Math.max(...optimizedData.time)
    
    // Find transit events within the observation period
    for (let i = 0; i < maxTime / period; i++) {
      const transitCenter = i * period
      const transitStart = transitCenter - transitDurationDays / 2
      const transitEnd = transitCenter + transitDurationDays / 2
      
      if (transitStart >= 0 && transitEnd <= maxTime) {
        highlights.push({ start: transitStart, end: transitEnd })
      }
    }
    
    return highlights
  }, [period, duration, optimizedData.time])

  const plotData = useMemo(() => {
    const traces: any[] = []

    // Main light curve trace
    const mainTrace: any = {
      x: optimizedData.time,
      y: optimizedData.flux,
      type: 'scattergl', // Use WebGL for better performance
      mode: 'lines', // Changed from markers to lines for performance
      line: {
        width: 1, // Reduced line width for performance
        color: theme === 'dark' ? '#60a5fa' : '#2563eb',
      },
      marker: {
        size: 1, // Reduced marker size for performance
        color: theme === 'dark' ? '#60a5fa' : '#2563eb',
        opacity: 0.5, // Reduced opacity for performance
      },
      name: 'Light Curve',
      hovertemplate:
        '<b>Time:</b> %{x:.3f} days<br>' +
        '<b>Flux:</b> %{y:.6f}<br>' +
        '<extra></extra>',
    }

    // Add error bars if available
    if (optimizedData.flux_err) {
      mainTrace.error_y = {
        type: 'data',
        array: optimizedData.flux_err,
        visible: true,
        color: theme === 'dark' ? '#94a3b8' : '#64748b',
        thickness: 0.5, // Reduced thickness for performance
        width: 0,
      }
    }

    traces.push(mainTrace)

    // Add transit highlights (only if there aren't too many)
    if (transitHighlights.length > 0 && transitHighlights.length <= 20) {
      transitHighlights.forEach((transit, index) => {
        // Create a mask for transit points
        const transitMask = optimizedData.time.map((t, i) =>
          t >= transit.start && t <= transit.end ? optimizedData.flux[i] : null
        )
        
        traces.push({
          x: optimizedData.time,
          y: transitMask,
          type: 'scattergl',
          mode: 'markers',
          marker: {
            size: 2, // Reduced marker size
            color: '#ef4444',
            opacity: 0.7, // Reduced opacity
          },
          name: `Transit ${index + 1}`,
          hovertemplate:
            '<b>Transit Event</b><br>' +
            '<b>Time:</b> %{x:.3f} days<br>' +
            '<b>Flux:</b> %{y:.6f}<br>' +
            '<extra></extra>',
        })
      })
    }

    return traces
  }, [optimizedData, theme, transitHighlights])

  const layout = useMemo(() => ({
    title: {
      text: `${targetName?.toString().replace(/[<>'"&]/g, (match) => {
        const escapeMap: Record<string, string> = {
          '<': '<',
          '>': '>',
          '"': '"',
          "'": '&#x27;',
          '&': '&'
        };
        return escapeMap[match] || match;
      })} - Light Curve`,
      font: {
        size: 16, // Reduced font size
        color: theme === 'dark' ? '#f1f5f9' : '#1e293b',
        family: 'Inter, system-ui, sans-serif'
      },
    },
    xaxis: {
      title: {
        text: 'Time (days)',
        font: {
          size: 12, // Reduced font size
          color: theme === 'dark' ? '#cbd5e1' : '#475569',
        },
      },
      gridcolor: theme === 'dark' ? '#374151' : '#e2e8f0',
      tickfont: {
        color: theme === 'dark' ? '#cbd5e1' : '#475569',
        size: 10, // Reduced tick font size
      },
      showgrid: true,
      zeroline: false,
      rangeslider: {
        visible: false, // Отключаем range slider для улучшения производительности
      }
    },
    yaxis: {
      title: {
        text: 'Normalized Flux',
        font: {
          size: 12, // Reduced font size
          color: theme === 'dark' ? '#cbd5e1' : '#475569',
        },
      },
      gridcolor: theme === 'dark' ? '#374151' : '#e2e8f0',
      tickfont: {
        color: theme === 'dark' ? '#cbd5e1' : '#475569',
        size: 10, // Reduced tick font size
      },
      showgrid: true,
      zeroline: false,
    },
    plot_bgcolor: 'transparent',
    paper_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: theme === 'dark' ? '#f1f5f9' : '#1e293b',
      size: 10, // Reduced base font size
    },
    legend: {
      bgcolor: 'transparent',
      bordercolor: 'transparent',
      font: {
        color: theme === 'dark' ? '#cbd5e1' : '#475569',
        size: 10, // Reduced legend font size
      },
      x: 1,
      y: 1,
      xanchor: 'right' as const,
      yanchor: 'top' as const
    },
    margin: {
      l: 50, // Reduced margins
      r: 30,
      t: 50,
      b: 50, // Reduced bottom margin (no range slider)
    },
    hovermode: 'closest' as const,
    showlegend: true,
  }), [theme, targetName])

  const config = useMemo(() => ({
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: [
      'lasso2d' as const,
      'select2d' as const,
      'zoomIn2d' as const,
      'zoomOut2d' as const,
      'autoScale2d' as const,
      'resetScale2d' as const,
      'hoverClosestCartesian' as const,
      'hoverCompareCartesian' as const,
      'zoom3d' as const,
      'pan3d' as const,
      'orbitRotation' as const,
      'tableRotation' as const,
      'handleDrag3d' as const,
      'resetCameraDefault3d' as const,
      'resetCameraLastSave3d' as const,
      'hoverClosest3d' as const,
    ],
    toImageButtonOptions: {
      format: 'png' as const,
      filename: `${targetName?.toString().replace(/[^a-zA-Z0-9\s\-_]/g, '_')}_lightcurve`,
      height: 600,
      width: 1000,
      scale: 2,
    },
    responsive: true,
  }), [targetName])


  return (
    <div className={`bg-white/10 dark:bg-gray-800/30 backdrop-blur-sm rounded-xl border border-white/20 dark:border-gray-700/50 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-white/10 dark:border-gray-700/50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg">
            <BarChart3 size={24} className="text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white dark:text-gray-100">
              Light Curve Analysis
            </h2>
            <p className="text-sm text-gray-300 dark:text-gray-400">
              Photometric time series data
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
            title="Download chart"
          >
            <Download size={16} className="text-gray-300" />
          </button>
          <button
            className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors"
            title="Zoom controls"
          >
            <ZoomIn size={16} className="text-gray-300" />
          </button>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="px-6 py-4 bg-white/5 dark:bg-gray-700/20">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-xs text-gray-400 dark:text-gray-500">Data Points</p>
            <p className="text-lg font-bold text-white dark:text-gray-100">
              {dataPoints.toLocaleString()}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400 dark:text-gray-500">Observation Period</p>
            <p className="text-lg font-bold text-white dark:text-gray-100">
              {observationDays} days
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400 dark:text-gray-500">Cadence</p>
            <p className="text-lg font-bold text-white dark:text-gray-100">
              {((observationDays * 24 * 60) / dataPoints).toFixed(1)} min
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400 dark:text-gray-500">Transits Found</p>
            <p className="text-lg font-bold text-white dark:text-gray-100">
              {transitHighlights.length}
            </p>
          </div>
        </div>
      </div>

      {/* Plot Container */}
      <div className="p-4">
        <Plot
          data={plotData}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '500px' }}
          useResizeHandler={true}
          className="plotly-chart"
        />
      </div>

      {/* Transit Info */}
      {period && depth && (
        <div className="px-6 pb-6">
          <div className="flex items-start gap-3 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <Info size={16} className="text-blue-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-blue-300 mb-1">
                Transit Parameters
              </p>
              <p className="text-xs text-blue-400">
                Period: {period.toFixed(2)} days • 
                Depth: {(depth * 100).toFixed(3)}% • 
                Duration: {duration ? (duration * 24).toFixed(1) : 'N/A'} hours
              </p>
              {transitHighlights.length > 0 && (
                <p className="text-xs text-blue-400 mt-1">
                  Red markers indicate predicted transit events based on BLS analysis
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
})

export default LightCurveGraph
