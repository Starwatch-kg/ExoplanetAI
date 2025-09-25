import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';
import { 
  TrendingUp, 
  Zap, 
  Target, 
  Eye, 
  BarChart3, 
  Info,
  Download,
  Maximize2
} from 'lucide-react';
import type { LightcurveData } from '../api/exoplanetApi';

interface Candidate {
  id: string;
  period: number;
  depth: number;
  duration: number;
  confidence: number;
  start_time: number;
  end_time: number;
  method: string;
}

interface LightcurveVisualizationProps {
  lightcurveData: LightcurveData;
  candidates?: Candidate[];
  isAnalyzing?: boolean;
}

const LightcurveVisualization: React.FC<LightcurveVisualizationProps> = ({
  lightcurveData,
  candidates = [],
  isAnalyzing = false
}) => {
  const [selectedCandidate, setSelectedCandidate] = useState<number | null>(null);
  const [showRawData, setShowRawData] = useState(true);
  const [showTransits, setShowTransits] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [stats, setStats] = useState<any>(null);

  // Вычисляем статистики кривой блеска
  useEffect(() => {
    if (lightcurveData) {
      const fluxes = lightcurveData.fluxes;
      const times = lightcurveData.times;
      
      const mean = fluxes.reduce((a, b) => a + b, 0) / fluxes.length;
      const variance = fluxes.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / fluxes.length;
      const stdDev = Math.sqrt(variance);
      const min = Math.min(...fluxes);
      const max = Math.max(...fluxes);
      const range = max - min;
      
      setStats({
        dataPoints: fluxes.length,
        timeSpan: times[times.length - 1] - times[0],
        meanFlux: mean,
        stdDev: stdDev,
        range: range,
        snr: mean / stdDev
      });
    }
  }, [lightcurveData]);

  // Подготавливаем данные для графика
  const getPlotData = () => {
    const traces: any[] = [];

    // Основная кривая блеска
    if (showRawData) {
      traces.push({
        x: lightcurveData.times,
        y: lightcurveData.fluxes,
        type: 'scattergl',
        mode: 'markers',
        marker: {
          size: 3,
          color: '#60a5fa',
          opacity: 0.7,
          line: { width: 0 }
        },
        name: 'Поток',
        hovertemplate: 
          '<b>Время:</b> %{x:.3f} дней<br>' +
          '<b>Поток:</b> %{y:.6f}<br>' +
          '<extra></extra>'
      });
    }

    // Сглаженная кривая
    if (lightcurveData.times.length > 10) {
      const smoothed = smoothData(lightcurveData.fluxes, 5);
      traces.push({
        x: lightcurveData.times,
        y: smoothed,
        type: 'scatter',
        mode: 'lines',
        line: {
          color: '#f59e0b',
          width: 2
        },
        name: 'Сглаженная',
        opacity: 0.8
      });
    }

    // Отмечаем транзиты
    if (showTransits && candidates.length > 0) {
      candidates.forEach((candidate, index) => {
        const isSelected = selectedCandidate === index;
        const color = isSelected ? '#ef4444' : '#10b981';
        const opacity = isSelected ? 1.0 : 0.6;

        // Вертикальные линии для транзитов
        traces.push({
          x: [candidate.start_time, candidate.start_time],
          y: [Math.min(...lightcurveData.fluxes), Math.max(...lightcurveData.fluxes)],
          type: 'scatter',
          mode: 'lines',
          line: {
            color: color,
            width: isSelected ? 3 : 2,
            dash: 'dash'
          },
          opacity: opacity,
          name: `Транзит ${index + 1}`,
          showlegend: false,
          hovertemplate: 
            `<b>Транзит ${index + 1}</b><br>` +
            `<b>Начало:</b> ${candidate.start_time.toFixed(3)} дней<br>` +
            `<b>Период:</b> ${candidate.period.toFixed(2)} дней<br>` +
            `<b>Глубина:</b> ${(candidate.depth * 100).toFixed(3)}%<br>` +
            `<b>Уверенность:</b> ${(candidate.confidence * 100).toFixed(1)}%<br>` +
            '<extra></extra>'
        });

        traces.push({
          x: [candidate.end_time, candidate.end_time],
          y: [Math.min(...lightcurveData.fluxes), Math.max(...lightcurveData.fluxes)],
          type: 'scatter',
          mode: 'lines',
          line: {
            color: color,
            width: isSelected ? 3 : 2,
            dash: 'dash'
          },
          opacity: opacity,
          name: `Конец транзита ${index + 1}`,
          showlegend: false
        });
      });
    }

    return traces;
  };

  // Функция сглаживания
  const smoothData = (data: number[], windowSize: number): number[] => {
    const smoothed: number[] = [];
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - Math.floor(windowSize / 2));
      const end = Math.min(data.length, i + Math.floor(windowSize / 2) + 1);
      const window = data.slice(start, end);
      const avg = window.reduce((a, b) => a + b, 0) / window.length;
      smoothed.push(avg);
    }
    return smoothed;
  };

  // Настройки графика
  const getLayout = () => ({
    title: {
      text: `🌟 TIC ${lightcurveData.tic_id} - Кривая блеска`,
      font: { 
        color: 'white', 
        size: 18,
        family: 'Inter, system-ui, sans-serif'
      },
      x: 0.5
    },
    xaxis: {
      title: {
        text: 'Время (дни)',
        font: { color: '#e2e8f0', size: 14 }
      },
      color: '#e2e8f0',
      gridcolor: '#374151',
      showgrid: true,
      zeroline: false,
      tickfont: { color: '#cbd5e1' }
    },
    yaxis: {
      title: {
        text: 'Нормализованный поток',
        font: { color: '#e2e8f0', size: 14 }
      },
      color: '#e2e8f0',
      gridcolor: '#374151',
      showgrid: true,
      zeroline: false,
      tickfont: { color: '#cbd5e1' }
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15,23,42,0.8)',
    font: { 
      color: 'white',
      family: 'Inter, system-ui, sans-serif'
    },
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: 'rgba(30,41,59,0.8)',
      bordercolor: '#475569',
      borderwidth: 1,
      font: { color: '#e2e8f0' }
    },
    margin: { l: 60, r: 40, t: 60, b: 60 },
    hovermode: 'closest' as const,
    showlegend: true
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-slate-900/50 backdrop-blur-sm rounded-lg border border-slate-700 overflow-hidden ${
        isFullscreen ? 'fixed inset-4 z-50' : ''
      }`}
    >
      {/* Заголовок с контролами */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-400" />
            Анализ кривой блеска
          </h2>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
              title="Полноэкранный режим"
            >
              <Maximize2 className="w-4 h-4 text-slate-300" />
            </button>
            
            <button
              className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
              title="Скачать данные"
            >
              <Download className="w-4 h-4 text-slate-300" />
            </button>
          </div>
        </div>

        {/* Контролы отображения */}
        <div className="flex flex-wrap gap-4">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={showRawData}
              onChange={(e) => setShowRawData(e.target.checked)}
              className="rounded bg-slate-700 border-slate-600"
            />
            <Eye className="w-4 h-4" />
            Исходные данные
          </label>
          
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={showTransits}
              onChange={(e) => setShowTransits(e.target.checked)}
              className="rounded bg-slate-700 border-slate-600"
            />
            <Target className="w-4 h-4" />
            Транзиты ({candidates.length})
          </label>
        </div>
      </div>

      <div className="grid lg:grid-cols-4 gap-4 p-4">
        {/* Статистики */}
        <div className="lg:col-span-1 space-y-4">
          {/* Основные статистики */}
          {stats && (
            <div className="bg-slate-800/50 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <Info className="w-4 h-4" />
                Статистики
              </h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">Точек данных:</span>
                  <span className="text-white font-mono">{stats.dataPoints.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Временной диапазон:</span>
                  <span className="text-white font-mono">{stats.timeSpan.toFixed(1)} дней</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Средний поток:</span>
                  <span className="text-white font-mono">{stats.meanFlux.toFixed(6)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Стд. отклонение:</span>
                  <span className="text-white font-mono">{stats.stdDev.toFixed(6)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">SNR:</span>
                  <span className="text-white font-mono">{stats.snr.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Размах:</span>
                  <span className="text-white font-mono">{(stats.range * 100).toFixed(3)}%</span>
                </div>
              </div>
            </div>
          )}

          {/* Список кандидатов */}
          {candidates.length > 0 && (
            <div className="bg-slate-800/50 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <Target className="w-4 h-4" />
                Кандидаты ({candidates.length})
              </h3>
              <div className="space-y-2">
                {candidates.map((candidate, index) => (
                  <motion.div
                    key={candidate.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={`p-3 rounded-lg cursor-pointer transition-all ${
                      selectedCandidate === index
                        ? 'bg-blue-600/20 border border-blue-500/50'
                        : 'bg-slate-700/50 hover:bg-slate-700'
                    }`}
                    onClick={() => setSelectedCandidate(selectedCandidate === index ? null : index)}
                  >
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-sm font-medium text-blue-400">
                        Кандидат {index + 1}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded ${
                        candidate.confidence > 0.7 
                          ? 'bg-green-600 text-white' 
                          : candidate.confidence > 0.4
                          ? 'bg-yellow-600 text-white'
                          : 'bg-red-600 text-white'
                      }`}>
                        {(candidate.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="text-xs text-slate-300 space-y-1">
                      <div>Период: <span className="text-white">{candidate.period.toFixed(2)} д</span></div>
                      <div>Глубина: <span className="text-white">{(candidate.depth * 100).toFixed(3)}%</span></div>
                      <div>Длительность: <span className="text-white">{candidate.duration.toFixed(2)} ч</span></div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {/* Индикатор анализа */}
          {isAnalyzing && (
            <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-blue-300 mb-2">
                <Zap className="w-4 h-4 animate-pulse" />
                <span className="text-sm font-medium">Анализ в процессе...</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
              </div>
            </div>
          )}
        </div>

        {/* График */}
        <div className="lg:col-span-3">
          <div className="bg-slate-800/30 rounded-lg p-4 h-full">
            <Plot
              data={getPlotData()}
              layout={getLayout()}
              style={{ 
                width: '100%', 
                height: isFullscreen ? 'calc(100vh - 200px)' : '500px' 
              }}
              config={{
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                displaylogo: false,
                toImageButtonOptions: {
                  format: 'png',
                  filename: `TIC_${lightcurveData.tic_id}_lightcurve`,
                  height: 600,
                  width: 1000,
                  scale: 2
                }
              }}
              useResizeHandler={true}
            />
          </div>
        </div>
      </div>

      {/* Подсказки */}
      <div className="p-4 border-t border-slate-700 bg-slate-800/20">
        <div className="flex flex-wrap gap-4 text-xs text-slate-400">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
            <span>Исходные данные</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-1 bg-yellow-500"></div>
            <span>Сглаженная кривая</span>
          </div>
          {candidates.length > 0 && (
            <div className="flex items-center gap-1">
              <div className="w-3 h-1 bg-green-500 border-dashed border"></div>
              <span>Обнаруженные транзиты</span>
            </div>
          )}
          <div className="flex items-center gap-1">
            <TrendingUp className="w-3 h-3 text-slate-400" />
            <span>Наведите курсор для деталей</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default LightcurveVisualization;
