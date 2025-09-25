import React from 'react';
import { History, Trash2, Eye, Download, Calendar, Clock } from 'lucide-react';
import { useApp } from '../App';
import type { AnalysisHistoryItem } from '../App';

const AnalysisHistory: React.FC = () => {
  const { analysisHistory, isOnline } = useApp();

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('ru-RU', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  const getModeColor = (mode: string) => {
    return mode === 'pro' ? 'text-purple-400 bg-purple-500/20' : 'text-blue-400 bg-blue-500/20';
  };

  const getModeLabel = (mode: string) => {
    return mode === 'pro' ? 'Профессиональный' : 'Любительский';
  };

  const clearHistory = () => {
    if (confirm('Вы уверены, что хотите очистить всю историю анализа?')) {
      localStorage.removeItem('exoplanet-history');
      window.location.reload(); // Простой способ обновить состояние
    }
  };

  if (analysisHistory.length === 0) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center">
        <div className="text-center p-8 bg-white/5 backdrop-blur-lg rounded-2xl border border-white/10">
          <History className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-white mb-2">История пуста</h3>
          <p className="text-gray-400">
            Здесь будут отображаться ваши предыдущие анализы
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Заголовок и действия */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">История анализа</h2>
          <p className="text-gray-400">
            Всего записей: {analysisHistory.length}
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {!isOnline && (
            <div className="flex items-center space-x-2 text-yellow-400 text-sm">
              <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
              <span>Офлайн режим</span>
            </div>
          )}
          <button
            onClick={clearHistory}
            className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 rounded-lg text-red-400 transition-all duration-200 flex items-center space-x-2"
          >
            <Trash2 className="w-4 h-4" />
            <span>Очистить</span>
          </button>
        </div>
      </div>

      {/* Список записей */}
      <div className="grid gap-4">
        {analysisHistory.map((item, index) => (
          <div
            key={index}
            className="bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all duration-200"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${getModeColor(item.mode)}`}>
                  {getModeLabel(item.mode)}
                </div>
                <div className="text-gray-400 text-sm">
                  TIC {item.ticId}
                </div>
              </div>
              <div className="flex items-center space-x-2 text-gray-400 text-sm">
                <Calendar className="w-4 h-4" />
                <span>{formatDate(item.timestamp)}</span>
              </div>
            </div>

            {/* Результат анализа */}
            {item.result && (
              <div className="bg-white/5 rounded-lg p-4 mb-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  {item.result.candidate && (
                    <>
                      <div>
                        <p className="text-gray-400">Период</p>
                        <p className="text-white font-semibold">
                          {item.result.candidate.period.toFixed(2)} дней
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-400">Глубина</p>
                        <p className="text-white font-semibold">
                          {(item.result.candidate.depth * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-400">Уверенность</p>
                        <p className="text-white font-semibold">
                          {(item.result.candidate.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </>
                  )}
                  <div>
                    <p className="text-gray-400">Время анализа</p>
                    <p className="text-white font-semibold">
                      {item.result.processingTime?.toFixed(2)} сек
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Действия */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-gray-400 text-sm">
                <Clock className="w-4 h-4" />
                <span>Сохранено автоматически</span>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  className="px-3 py-1 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 rounded text-blue-400 text-sm transition-all duration-200 flex items-center space-x-1"
                >
                  <Eye className="w-3 h-3" />
                  <span>Просмотр</span>
                </button>
                <button
                  className="px-3 py-1 bg-green-500/20 hover:bg-green-500/30 border border-green-500/30 rounded text-green-400 text-sm transition-all duration-200 flex items-center space-x-1"
                >
                  <Download className="w-3 h-3" />
                  <span>Экспорт</span>
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Статистика */}
      <div className="bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Статистика</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-400">{analysisHistory.length}</p>
            <p className="text-gray-400">Всего анализов</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-purple-400">
              {analysisHistory.filter(item => item.mode === 'pro').length}
            </p>
            <p className="text-gray-400">Профессиональных</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-400">
              {analysisHistory.filter(item => item.mode === 'amateur').length}
            </p>
            <p className="text-gray-400">Любительских</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisHistory;
