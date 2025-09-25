import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
const Plot = React.lazy(() => import('react-plotly.js'));
import { 
  Download, 
  BarChart3, 
  TrendingUp, 
  AlertTriangle,
  Star,
  Circle
} from 'lucide-react';

interface Candidate {
  id: string;
  period: number;
  depth: number;
  duration: number;
  confidence: number;
  startTime: number;
  endTime: number;
  method: string;
}

interface ResultsProps {
  lightcurveData: { times: number[]; fluxes: number[] };
  candidates: Candidate[];
  modelType: string;
  onReset: () => void;
}

const Results: React.FC<ResultsProps> = ({
  lightcurveData,
  candidates,
  modelType,
  onReset
}) => {
  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);
  const [activeTab, setActiveTab] = useState<'lightcurve' | 'periodogram' | 'phased'>('lightcurve');

  // Generate mock periodogram data
  const periodogramData = useMemo(() => {
    const periods = Array.from({ length: 100 }, (_, i) => 0.5 + (i * 0.5));
    const power = periods.map(p => 
      Math.random() * 100 + 
      (candidates.length > 0 ? 50 * Math.exp(-Math.pow(Math.log(p / candidates[0].period), 2)) : 0)
    );
    return { periods, power };
  }, [candidates]);

  // Generate phased light curve for selected candidate
  const phasedData = useMemo(() => {
    if (!selectedCandidate) return { phases: [], fluxes: [] };
    
    const { times, fluxes } = lightcurveData;
    const period = selectedCandidate.period;
    const phases = times.map(t => ((t % period) / period) * 2 * Math.PI);
    
    // Sort by phase
    const sortedIndices = phases.map((_, i) => i).sort((a, b) => phases[a] - phases[b]);
    
    return {
      phases: sortedIndices.map(i => phases[i]),
      fluxes: sortedIndices.map(i => fluxes[i])
    };
  }, [selectedCandidate, lightcurveData]);

  const lightcurvePlot = {
    data: [
      {
        x: lightcurveData.times,
        y: lightcurveData.fluxes,
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'Flux',
        line: { color: '#0ea5e9', width: 1 },
        hovertemplate: 'Время: %{x:.2f}<br>Поток: %{y:.4f}<extra></extra>'
      },
      ...candidates.map((candidate, index) => ({
        x: [candidate.startTime, candidate.startTime, candidate.endTime, candidate.endTime, candidate.startTime],
        y: [1, 0.95, 0.95, 1, 1],
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: `Кандидат ${index + 1}`,
        line: { color: '#ef4444', width: 2 },
        fill: 'tonexty' as const,
        fillcolor: 'rgba(239, 68, 68, 0.2)',
        showlegend: false,
        hovertemplate: `Кандидат ${index + 1}<br>Период: ${candidate.period.toFixed(2)} дн<br>Глубина: ${candidate.depth.toFixed(4)}<extra></extra>`
      }))
    ],
    layout: {
      title: {
        text: 'Кривая блеска с обнаруженными транзитами',
        font: { color: 'white', size: 16 }
      },
      xaxis: {
        title: { text: 'Время (дни)' } as any,
        color: 'white',
        gridcolor: 'rgba(255,255,255,0.1)'
      },
      yaxis: {
        title: { text: 'Нормализованный поток' } as any,
        color: 'white',
        gridcolor: 'rgba(255,255,255,0.1)'
      },
      plot_bgcolor: 'rgba(0,0,0,0)',
      paper_bgcolor: 'rgba(0,0,0,0)',
      font: { color: 'white' },
      legend: { x: 0, y: 1 }
    },
    config: {
      displayModeBar: true,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'] as any,
      displaylogo: false
    }
  };

  const periodogramPlot = {
    data: [
      {
        x: periodogramData.periods,
        y: periodogramData.power,
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: 'BLS Power',
        line: { color: '#8b5cf6', width: 2 },
        hovertemplate: 'Период: %{x:.2f} дн<br>Мощность: %{y:.1f}<extra></extra>'
      }
    ],
    layout: {
      title: {
        text: 'Периодограмма BLS',
        font: { color: 'white', size: 16 }
      },
      xaxis: {
        title: { text: 'Период (дни)' } as any,
        color: 'white',
        gridcolor: 'rgba(255,255,255,0.1)',
        type: 'log' as any
      },
      yaxis: {
        title: { text: 'Мощность BLS' } as any,
        color: 'white',
        gridcolor: 'rgba(255,255,255,0.1)'
      },
      plot_bgcolor: 'rgba(0,0,0,0)',
      paper_bgcolor: 'rgba(0,0,0,0)',
      font: { color: 'white' }
    },
    config: {
      displayModeBar: true,
      displaylogo: false
    }
  };

  const phasedPlot = {
    data: [
      {
        x: phasedData.phases,
        y: phasedData.fluxes,
        type: 'scatter' as const,
        mode: 'markers' as const,
        name: 'Phased Flux',
        marker: { color: '#10b981', size: 4 },
        hovertemplate: 'Фаза: %{x:.2f}<br>Поток: %{y:.4f}<extra></extra>'
      }
    ],
    layout: {
      title: {
        text: `Фазовая кривая (Период: ${selectedCandidate?.period.toFixed(2)} дн)`,
        font: { color: 'white', size: 16 }
      },
      xaxis: {
        title: { text: 'Фаза (радианы)' } as any,
        color: 'white',
        gridcolor: 'rgba(255,255,255,0.1)',
        range: [0, 2 * Math.PI]
      },
      yaxis: {
        title: { text: 'Нормализованный поток' } as any,
        color: 'white',
        gridcolor: 'rgba(255,255,255,0.1)'
      },
      plot_bgcolor: 'rgba(0,0,0,0)',
      paper_bgcolor: 'rgba(0,0,0,0)',
      font: { color: 'white' }
    },
    config: {
      displayModeBar: true,
      displaylogo: false
    }
  };

  const exportResults = () => {
    const data = {
      model: modelType,
      timestamp: new Date().toISOString(),
      candidates: candidates,
      statistics: {
        totalCandidates: candidates.length,
        averageConfidence: candidates.reduce((sum, c) => sum + c.confidence, 0) / candidates.length,
        bestCandidate: candidates.reduce((best, c) => c.confidence > best.confidence ? c : best, candidates[0])
      }
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `exoplanet_results_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="space-card max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-3xl font-bold text-white mb-2">
              Результаты анализа
            </h2>
            <p className="text-slate-400">
              Модель: {modelType} • Найдено кандидатов: {candidates.length}
            </p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={exportResults}
              className="cosmic-button flex items-center space-x-2"
            >
              <Download className="w-5 h-5" />
              <span>Экспорт</span>
            </button>
            <button
              onClick={onReset}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
            >
              Новый анализ
            </button>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-slate-800/50 rounded-lg">
            <div className="text-2xl font-bold text-cyan-400">{candidates.length}</div>
            <div className="text-sm text-slate-400">Кандидатов</div>
          </div>
          <div className="text-center p-4 bg-slate-800/50 rounded-lg">
            <div className="text-2xl font-bold text-purple-400">
              {candidates.length > 0 ? (candidates.reduce((sum, c) => sum + c.confidence, 0) / candidates.length * 100).toFixed(1) : 0}%
            </div>
            <div className="text-sm text-slate-400">Средняя уверенность</div>
          </div>
          <div className="text-center p-4 bg-slate-800/50 rounded-lg">
            <div className="text-2xl font-bold text-green-400">
              {candidates.length > 0 ? Math.min(...candidates.map(c => c.period)).toFixed(1) : 0}
            </div>
            <div className="text-sm text-slate-400">Мин. период (дн)</div>
          </div>
          <div className="text-center p-4 bg-slate-800/50 rounded-lg">
            <div className="text-2xl font-bold text-yellow-400">
              {candidates.length > 0 ? Math.max(...candidates.map(c => c.period)).toFixed(1) : 0}
            </div>
            <div className="text-sm text-slate-400">Макс. период (дн)</div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="space-card max-w-6xl mx-auto">
        <div className="flex space-x-1 mb-6 bg-slate-800/50 p-1 rounded-lg">
          {[
            { id: 'lightcurve', label: 'Кривая блеска', icon: TrendingUp },
            { id: 'periodogram', label: 'Периодограмма', icon: BarChart3 },
            { id: 'phased', label: 'Фазовая кривая', icon: Star }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex-1 flex items-center justify-center space-x-2 py-2 px-4 rounded-md transition-all ${
                activeTab === tab.id
                  ? 'bg-cyan-500 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        <div className="h-96">
          {activeTab === 'lightcurve' && (
            <React.Suspense fallback={<div />}> 
              <Plot {...lightcurvePlot} style={{ width: '100%', height: '100%' }} />
            </React.Suspense>
          )}
          {activeTab === 'periodogram' && (
            <React.Suspense fallback={<div />}> 
              <Plot {...periodogramPlot} style={{ width: '100%', height: '100%' }} />
            </React.Suspense>
          )}
          {activeTab === 'phased' && (
            selectedCandidate ? (
              <React.Suspense fallback={<div />}> 
                <Plot {...phasedPlot} style={{ width: '100%', height: '100%' }} />
              </React.Suspense>
            ) : (
              <div className="flex items-center justify-center h-full text-slate-400">
                <div className="text-center">
                  <Circle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Выберите кандидата для просмотра фазовой кривой</p>
                </div>
              </div>
            )
          )}
        </div>
      </div>

      {/* Candidates List */}
      {candidates.length > 0 && (
        <div className="space-card max-w-6xl mx-auto">
          <h3 className="text-xl font-semibold text-white mb-6">
            Обнаруженные кандидаты
          </h3>
          <div className="space-y-4">
            {candidates.map((candidate, index) => (
              <motion.div
                key={candidate.id}
                whileHover={{ scale: 1.02 }}
                onClick={() => setSelectedCandidate(candidate)}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  selectedCandidate?.id === candidate.id
                    ? 'border-cyan-400 bg-cyan-500/10'
                    : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      {index + 1}
                    </div>
                    <div>
                      <h4 className="text-lg font-semibold text-white">
                        Кандидат {index + 1}
                      </h4>
                      <p className="text-sm text-slate-400">
                        Метод: {candidate.method} • Уверенность: {(candidate.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-cyan-400">
                      {(candidate.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-slate-400">уверенность</div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t border-slate-700">
                  <div>
                    <div className="text-sm text-slate-400">Период</div>
                    <div className="text-lg font-semibold text-white">{candidate.period.toFixed(2)} дн</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">Глубина</div>
                    <div className="text-lg font-semibold text-white">{candidate.depth.toFixed(4)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">Длительность</div>
                    <div className="text-lg font-semibold text-white">{candidate.duration.toFixed(2)} дн</div>
                  </div>
                  <div>
                    <div className="text-sm text-slate-400">Время начала</div>
                    <div className="text-lg font-semibold text-white">{candidate.startTime.toFixed(2)} дн</div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* No Candidates Message */}
      {candidates.length === 0 && (
        <div className="space-card max-w-2xl mx-auto text-center">
          <AlertTriangle className="w-16 h-16 text-yellow-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">
            Кандидаты не найдены
          </h3>
          <p className="text-slate-400 mb-6">
            В данной кривой блеска не обнаружено признаков транзитов экзопланет. 
            Попробуйте другой TIC ID или другую модель анализа.
          </p>
          <button
            onClick={onReset}
            className="cosmic-button"
          >
            Попробовать снова
          </button>
        </div>
      )}
    </motion.div>
  );
};

export default Results;
