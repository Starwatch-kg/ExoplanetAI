import React, { useMemo } from 'react'
import Plot from 'react-plotly.js'
import { useTranslation } from 'react-i18next'
import { useTheme } from '../../../../front/frontend/src/contexts/ThemeContext'

interface LightCurveData {
  time: number[]
  flux: number[]
  flux_err?: number[]
}

interface LightCurveChartProps {
  data: LightCurveData
  title?: string
  width?: number
  height?: number
  showErrorBars?: boolean
  highlightTransits?: Array<{ start: number; end: number }>
}

const LightCurveChart: React.FC<LightCurveChartProps> = ({
  data,
  title,
  width = 800,
  height = 400,
  showErrorBars = false,
  highlightTransits = [],
}) => {
  const { t } = useTranslation()
  const { theme } = useTheme()

  const plotData = useMemo(() => {
    const traces: any[] = []

    // Main light curve trace
    const mainTrace: any = {
      x: data.time,
      y: data.flux,
      type: 'scatter',
      mode: 'markers',
      marker: {
        size: 3,
        color: theme === 'dark' ? '#60a5fa' : '#2563eb',
        opacity: 0.7,
      },
      name: t('results.lightcurve.title'),
      hovertemplate: 
        '<b>Time:</b> %{x:.3f} ' + t('units.days') + '<br>' +
        '<b>Flux:</b> %{y:.6f}<br>' +
        '<extra></extra>',
    }

    // Add error bars if available
    if (showErrorBars && data.flux_err) {
      mainTrace.error_y = {
        type: 'data',
        array: data.flux_err,
        visible: true,
        color: theme === 'dark' ? '#94a3b8' : '#64748b',
        thickness: 1,
        width: 0,
      }
    }

    traces.push(mainTrace)

    // Add transit highlights
    highlightTransits.forEach((transit, index) => {
      const transitMask = data.time.map((t, i) => 
        t >= transit.start && t <= transit.end ? data.flux[i] : null
      )
      
      traces.push({
        x: data.time,
        y: transitMask,
        type: 'scatter',
        mode: 'markers',
        marker: {
          size: 4,
          color: '#ef4444',
          opacity: 0.8,
        },
        name: `Transit ${index + 1}`,
        hovertemplate: 
          '<b>Transit Event</b><br>' +
          '<b>Time:</b> %{x:.3f} ' + t('units.days') + '<br>' +
          '<b>Flux:</b> %{y:.6f}<br>' +
          '<extra></extra>',
      })
    })

    return traces
  }, [data, theme, t, showErrorBars, highlightTransits])

  const layout = useMemo(() => ({
    title: {
      text: title || t('results.lightcurve.title'),
      font: {
        size: 18,
        color: theme === 'dark' ? '#f1f5f9' : '#1e293b',
      },
    },
    xaxis: {
      title: {
        text: t('results.lightcurve.xlabel'),
        font: {
          size: 14,
          color: theme === 'dark' ? '#cbd5e1' : '#475569',
        },
      },
      gridcolor: theme === 'dark' ? '#374151' : '#e2e8f0',
      tickfont: {
        color: theme === 'dark' ? '#cbd5e1' : '#475569',
      },
      showgrid: true,
      zeroline: false,
    },
    yaxis: {
      title: {
        text: t('results.lightcurve.ylabel'),
        font: {
          size: 14,
          color: theme === 'dark' ? '#cbd5e1' : '#475569',
        },
      },
      gridcolor: theme === 'dark' ? '#374151' : '#e2e8f0',
      tickfont: {
        color: theme === 'dark' ? '#cbd5e1' : '#475569',
      },
      showgrid: true,
      zeroline: false,
    },
    plot_bgcolor: 'transparent',
    paper_bgcolor: 'transparent',
    font: {
      family: 'Inter, system-ui, sans-serif',
      color: theme === 'dark' ? '#f1f5f9' : '#1e293b',
    },
    legend: {
      bgcolor: 'transparent',
      bordercolor: 'transparent',
      font: {
        color: theme === 'dark' ? '#cbd5e1' : '#475569',
      },
    },
    margin: {
      l: 60,
      r: 40,
      t: 60,
      b: 60,
    },
    hovermode: 'closest' as const,
    showlegend: true,
  }), [theme, t, title])

  const config = useMemo(() => ({
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: [
      'pan2d' as const,
      'lasso2d' as const,
      'select2d' as const,
      'autoScale2d' as const,
      'hoverClosestCartesian' as const,
      'hoverCompareCartesian' as const,
      'toggleSpikelines' as const,
    ],
    toImageButtonOptions: {
      format: 'png' as const,
      filename: 'exoplanet_lightcurve',
      height: height,
      width: width,
      scale: 2,
    },
    responsive: true,
  }), [width, height])

  return (
    <div className="w-full bg-white/5 dark:bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-white/10 dark:border-gray-700">
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: `${height}px` }}
        useResizeHandler={true}
        className="plotly-chart"
      />
    </div>
  )
}

export default LightCurveChart
