import React, { useState } from 'react';
import { Search, Play, Download, Star } from 'lucide-react';
import { exoplanetApi } from '../api/exoplanetApi';
import DataSourceSelector from '../components/DataSourceSelector';
import type { AmateurAnalysisRequest, AmateurAnalysisResponse, Candidate } from '../api/exoplanetApi';

interface AnalysisResult {
  candidate: Candidate | null;
  summary: Record<string, any>;
  processingTime: number;
  error?: string;
}

const AmateurView: React.FC = () => {
  const [ticId, setTicId] = useState('');
  const [dataSource, setDataSource] = useState('simulated');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleAnalyze = async () => {
    if (!ticId.trim()) {
      alert('Пожалуйста, введите TIC ID');
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      const request: AmateurAnalysisRequest = {
        tic_id: ticId.trim()
      };

      const response: AmateurAnalysisResponse = await exoplanetApi.amateurAnalyze(request);

      if (response.success && response.candidate) {
        setResult({
          candidate: response.candidate,
          summary: response.summary,
          processingTime: response.processing_time
        });
      } else {
        setResult({
          candidate: null,
          summary: response.summary,
          processingTime: response.processing_time,
          error: response.error
        });
      }
    } catch (error) {
      console.error('Ошибка анализа:', error);
      setResult({
        candidate: null,
        summary: {},
        processingTime: 0,
        error: error instanceof Error ? error.message : 'Неизвестная ошибка'
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'Высокая уверенность';
    if (confidence >= 0.6) return 'Средняя уверенность';
    return 'Низкая уверенность';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Заголовок */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-4 flex items-center justify-center gap-3">
            <Star className="w-10 h-10 text-yellow-400" />
            Любительский режим
            <Star className="w-10 h-10 text-yellow-400" />
          </h1>
          <p className="text-xl text-gray-300">
            Простой анализ для поиска экзопланет
          </p>
        </div>

        {/* Выбор источника данных */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-white mb-3">Источник данных</h3>
          <DataSourceSelector
            selectedSource={dataSource}
            onSourceChange={setDataSource}
          />
        </div>
        <div className="flex gap-4 mb-6">
          <div className="flex-1">
            <label className="block text-white text-sm font-medium mb-2">
              TIC ID звезды
            </label>
            <input
              type="text"
              value={ticId}
              onChange={(e) => setTicId(e.target.value)}
              placeholder="Введите TIC ID (например: TIC 123456789)"
              className="w-full px-4 py-3 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isAnalyzing}
            />
            <div className="flex items-end">
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing || !ticId.trim()}
                className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-600 text-white rounded-lg font-medium transition-all duration-200 flex items-center gap-2 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Анализ...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Начать анализ
                  </>
                )}
              </button>
            </div>
          </div>

          <div className="text-sm text-gray-300">
            <p>💡 <strong>Совет:</strong> Начните с известных TIC ID, таких как TIC 123456789 или TIC 987654321</p>
          </div>
        </div>

        {/* Результат анализа */}
        {result && (
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
            <h2 className="text-2xl font-bold text-white mb-6">Результат анализа</h2>

            {result.error ? (
              <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-4">
                <p className="text-red-300">Ошибка: {result.error}</p>
              </div>
            ) : result.candidate ? (
              <div className="space-y-6">
                {/* Основной кандидат */}
                <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 border border-green-500/30 rounded-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-semibold text-white">Обнаружен кандидат</h3>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(result.candidate.confidence)}`}>
                      {getConfidenceLabel(result.candidate.confidence)}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <p className="text-gray-300 text-sm">Период</p>
                      <p className="text-white font-semibold">{result.candidate.period.toFixed(2)} дней</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-300 text-sm">Глубина</p>
                      <p className="text-white font-semibold">{(result.candidate.depth * 100).toFixed(3)}%</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-300 text-sm">Длительность</p>
                      <p className="text-white font-semibold">{result.candidate.duration.toFixed(2)} часов</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-300 text-sm">Уверенность</p>
                      <p className="text-white font-semibold">{(result.candidate.confidence * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                </div>

                {/* Краткий отчёт */}
                <div className="bg-white/5 border border-white/10 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Краткий отчёт</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <p className="text-gray-300">Всего кандидатов:</p>
                      <p className="text-white font-semibold">{result.summary.total_candidates || 0}</p>
                    </div>
                    <div>
                      <p className="text-gray-300">Время анализа:</p>
                      <p className="text-white font-semibold">{result.processingTime.toFixed(2)} сек</p>
                    </div>
                    <div className="md:col-span-2">
                      <p className="text-gray-300">Качество данных:</p>
                      <p className="text-white font-semibold">{result.summary.data_quality || 'Неизвестно'}</p>
                    </div>
                    {result.summary.recommendation && (
                      <div className="md:col-span-2">
                        <p className="text-gray-300">Рекомендация:</p>
                        <p className="text-white font-semibold">{result.summary.recommendation}</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Действия */}
                <div className="flex gap-4">
                  <button className="px-6 py-3 bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 text-white rounded-lg font-medium transition-all duration-200 flex items-center gap-2">
                    <Download className="w-5 h-5" />
                    Скачать отчёт
                  </button>
                  <button
                    onClick={() => handleAnalyze()}
                    className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white rounded-lg font-medium transition-all duration-200 flex items-center gap-2"
                  >
                    <Search className="w-5 h-5" />
                    Профессиональный анализ
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-gray-300 text-lg">Кандидаты не обнаружены</p>
                <p className="text-gray-400 mt-2">Попробуйте другой TIC ID или перейдите в профессиональный режим для более детального анализа</p>
              </div>
            )}
          </div>
        )}

        {/* Инструкция */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6 mt-8">
          <h3 className="text-lg font-semibold text-white mb-4">Как использовать</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-gray-300 font-medium mb-2">1. Введите TIC ID</p>
              <p className="text-gray-400">TIC ID можно найти на сайте TESS или в каталогах экзопланет</p>
            </div>
            <div>
              <p className="text-gray-300 font-medium mb-2">2. Нажмите "Анализ"</p>
              <p className="text-gray-400">ИИ-модель проанализирует кривую блеска и найдёт кандидатов</p>
            </div>
            <div>
              <p className="text-gray-300 font-medium mb-2">3. Просмотрите результат</p>
              <p className="text-gray-400">Получите информацию о потенциальных экзопланетах</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AmateurView;
