import React, { useState, useEffect } from 'react';
import { Clock, Globe, Activity, Database } from 'lucide-react';
import { useApp } from '../App';

interface LatestAnalysis {
  tic_id: string;
  timestamp: number;
  candidates_count: number;
  processing_time: number;
}

interface LatestAnalysesResponse {
  success: boolean;
  latest_analyses: LatestAnalysis[];
  total_cached: number;
  error?: string;
}

const LatestAnalyses: React.FC = () => {
  const [latestAnalyses, setLatestAnalyses] = useState<LatestAnalysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { isOnline } = useApp();

  useEffect(() => {
    fetchLatestAnalyses();
  }, []);

  const fetchLatestAnalyses = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/latest-analyses');
      const data: LatestAnalysesResponse = await response.json();

      if (data.success) {
        setLatestAnalyses(data.latest_analyses);
        setError(null);
      } else {
        setError(data.error || 'Ошибка загрузки данных');
      }
    } catch (err) {
      setError('Не удалось подключиться к API');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return new Intl.DateTimeFormat('ru-RU', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    }).format(date);
  };

  const formatRelativeTime = (timestamp: number) => {
    const now = Date.now() / 1000;
    const diff = now - timestamp;

    if (diff < 60) return 'Только что';
    if (diff < 3600) return `${Math.floor(diff / 60)} мин назад`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} ч назад`;
    return `${Math.floor(diff / 86400)} д назад`;
  };

  const getConfidenceColor = (count: number) => {
    if (count === 0) return 'text-red-400 bg-red-500/20';
    if (count === 1) return 'text-yellow-400 bg-yellow-500/20';
    return 'text-green-400 bg-green-500/20';
  };

  if (loading) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Database className="w-6 h-6 text-blue-400" />
            Последние анализы
          </h2>
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
        </div>
        <p className="text-gray-400">Загрузка данных...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Database className="w-6 h-6 text-blue-400" />
            Последние анализы
          </h2>
          <button
            onClick={fetchLatestAnalyses}
            className="px-3 py-1 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 rounded text-blue-400 text-sm transition-all duration-200"
          >
            Обновить
          </button>
        </div>
        <div className="text-center py-8">
          <p className="text-red-400 mb-2">⚠️ {error}</p>
          {!isOnline && (
            <p className="text-yellow-400 text-sm">Бэкенд может быть недоступен</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Database className="w-6 h-6 text-blue-400" />
          Последние анализы
        </h2>
        <div className="flex items-center gap-4">
          <span className="text-gray-400 text-sm">
            Всего в кэше: {latestAnalyses.length}
          </span>
          <button
            onClick={fetchLatestAnalyses}
            className="px-3 py-1 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 rounded text-blue-400 text-sm transition-all duration-200"
          >
            Обновить
          </button>
        </div>
      </div>

      {latestAnalyses.length === 0 ? (
        <div className="text-center py-8">
          <Database className="w-12 h-12 text-gray-500 mx-auto mb-4" />
          <p className="text-gray-400">Нет данных об анализах</p>
          <p className="text-gray-500 text-sm mt-2">Выполните анализ, чтобы увидеть статистику</p>
        </div>
      ) : (
        <div className="space-y-3">
          {latestAnalyses.map((analysis, index) => (
            <div
              key={`${analysis.tic_id}-${analysis.timestamp}`}
              className="bg-white/5 rounded-lg p-4 border border-white/10 hover:border-white/20 transition-all duration-200"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-500/20 text-blue-400 font-semibold text-sm">
                    {index + 1}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <Globe className="w-4 h-4 text-gray-400" />
                      <span className="text-white font-medium">TIC {analysis.tic_id}</span>
                      <div className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(analysis.candidates_count)}`}>
                        {analysis.candidates_count} кандидатов
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-400 mt-1">
                      <Clock className="w-3 h-3" />
                      <span>{formatTime(analysis.timestamp)}</span>
                      <span>•</span>
                      <span>{formatRelativeTime(analysis.timestamp)}</span>
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="flex items-center gap-1 text-sm">
                    <Activity className="w-3 h-3 text-gray-400" />
                    <span className="text-gray-400">
                      {analysis.processing_time.toFixed(2)} сек
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Статистика */}
      <div className="mt-6 pt-4 border-t border-white/10">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold text-blue-400">{latestAnalyses.length}</p>
            <p className="text-gray-400 text-sm">Всего анализов</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-green-400">
              {latestAnalyses.reduce((sum, analysis) => sum + analysis.candidates_count, 0)}
            </p>
            <p className="text-gray-400 text-sm">Всего кандидатов</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-purple-400">
              {latestAnalyses.length > 0
                ? (latestAnalyses.reduce((sum, analysis) => sum + analysis.processing_time, 0) / latestAnalyses.length).toFixed(1)
                : '0'
              }
            </p>
            <p className="text-gray-400 text-sm">Среднее время</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LatestAnalyses;
