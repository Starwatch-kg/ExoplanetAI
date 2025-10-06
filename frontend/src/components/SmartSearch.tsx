import React, { useState, useCallback } from 'react';
import { Search, Filter, Star, TrendingUp, AlertCircle, CheckCircle, Clock } from 'lucide-react';

interface SearchFilters {
  confidenceMin: number;
  snrMin: number;
  dataQualityMin: number;
  missions: string[];
  planetTypes: string[];
  periodRange?: [number, number];
  depthRange?: [number, number];
}

interface SearchResult {
  name: string;
  predicted_class: string;
  confidence_score: number;
  ai_score: number;
  signal_characteristics: {
    snr_estimate: number;
    data_points: number;
    time_span_days: number;
  };
  disposition: string;
  orbital_period?: number;
  planet_radius?: number;
  is_similar?: boolean;
  similarity_reason?: string;
}

interface SmartSearchResponse {
  results: SearchResult[];
  total_found: number;
  ai_ranked: boolean;
  recommendations: string[];
  search_time_ms: number;
  filters_applied: SearchFilters;
}

const SmartSearch: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<SearchFilters>({
    confidenceMin: 0.7,
    snrMin: 5.0,
    dataQualityMin: 0.8,
    missions: ['TESS', 'Kepler'],
    planetTypes: ['confirmed', 'candidate']
  });
  
  const [results, setResults] = useState<SearchResult[]>([]);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchTime, setSearchTime] = useState<number>(0);
  const [totalFound, setTotalFound] = useState<number>(0);
  const [showFilters, setShowFilters] = useState(false);
  const [error, setError] = useState<string>('');

  const handleSmartSearch = useCallback(async () => {
    if (!searchQuery.trim()) return;
    
    setIsLoading(true);
    setError('');
    
    try {
      const response = await fetch('/api/v1/ai/smart_search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          filters: filters,
          use_ai_ranking: true,
          max_results: 20,
          include_similar: true
        })
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.status}`);
      }

      const data: SmartSearchResponse = await response.json();
      
      setResults(data.results);
      setRecommendations(data.recommendations);
      setSearchTime(data.search_time_ms);
      setTotalFound(data.total_found);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      console.error('Smart search error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [searchQuery, filters]);

  const handleDetailedAnalysis = async (targetName: string) => {
    try {
      const response = await fetch('/api/v1/ai/analyze_lightcurve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          target_name: targetName,
          mission: 'TESS',
          include_uncertainty: true,
          explain_prediction: true
        })
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status}`);
      }

      const analysisResult = await response.json();
      
      // Здесь можно открыть модальное окно с детальными результатами
      console.log('Detailed analysis:', analysisResult);
      
    } catch (err) {
      console.error('Analysis error:', err);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getClassIcon = (predictedClass: string) => {
    switch (predictedClass) {
      case 'CANDIDATE':
      case 'Confirmed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'PC':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-red-500" />;
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Search Header */}
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold text-white flex items-center justify-center gap-2">
          <Star className="w-8 h-8 text-blue-400" />
          Умный поиск экзопланет
        </h1>
        <p className="text-gray-300">
          Найдите экзопланеты с помощью ИИ анализа и интеллектуального ранжирования
        </p>
      </div>

      {/* Search Input */}
      <div className="relative">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSmartSearch()}
              placeholder="Введите TIC ID, координаты или параметры транзита..."
              className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="px-4 py-3 bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded-lg text-white transition-colors flex items-center gap-2"
          >
            <Filter className="w-5 h-5" />
            Фильтры
          </button>
          
          <button
            onClick={handleSmartSearch}
            disabled={isLoading || !searchQuery.trim()}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-white transition-colors flex items-center gap-2"
          >
            {isLoading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Search className="w-5 h-5" />
            )}
            Поиск
          </button>
        </div>
      </div>

      {/* Advanced Filters */}
      {showFilters && (
        <div className="bg-gray-800 border border-gray-600 rounded-lg p-6 space-y-4">
          <h3 className="text-lg font-semibold text-white mb-4">Расширенные фильтры</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Confidence Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Минимальная уверенность: {filters.confidenceMin.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={filters.confidenceMin}
                onChange={(e) => setFilters(prev => ({ ...prev, confidenceMin: parseFloat(e.target.value) }))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* SNR Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Минимальный SNR: {filters.snrMin.toFixed(1)}
              </label>
              <input
                type="range"
                min="1"
                max="20"
                step="0.5"
                value={filters.snrMin}
                onChange={(e) => setFilters(prev => ({ ...prev, snrMin: parseFloat(e.target.value) }))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* Data Quality Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Качество данных: {filters.dataQualityMin.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={filters.dataQualityMin}
                onChange={(e) => setFilters(prev => ({ ...prev, dataQualityMin: parseFloat(e.target.value) }))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>

          {/* Mission Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Миссии</label>
            <div className="flex flex-wrap gap-2">
              {['TESS', 'Kepler', 'K2'].map(mission => (
                <label key={mission} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={filters.missions.includes(mission)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setFilters(prev => ({ ...prev, missions: [...prev.missions, mission] }));
                      } else {
                        setFilters(prev => ({ ...prev, missions: prev.missions.filter(m => m !== mission) }));
                      }
                    }}
                    className="rounded border-gray-600 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-gray-300">{mission}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-900 border border-red-600 rounded-lg p-4 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <span className="text-red-200">{error}</span>
        </div>
      )}

      {/* Search Stats */}
      {(results.length > 0 || totalFound > 0) && (
        <div className="bg-gray-800 border border-gray-600 rounded-lg p-4">
          <div className="flex items-center justify-between text-sm text-gray-300">
            <span>Найдено: {totalFound} объектов</span>
            <span>Время поиска: {searchTime.toFixed(0)} мс</span>
            <span className="flex items-center gap-1">
              <TrendingUp className="w-4 h-4" />
              ИИ ранжирование активно
            </span>
          </div>
        </div>
      )}

      {/* AI Recommendations */}
      {recommendations.length > 0 && (
        <div className="bg-blue-900 border border-blue-600 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-blue-200 mb-2">Рекомендации ИИ</h3>
          <ul className="space-y-1">
            {recommendations.map((rec, index) => (
              <li key={index} className="text-blue-200 text-sm">• {rec}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Search Results */}
      <div className="space-y-4">
        {results.map((result, index) => (
          <div key={index} className="bg-gray-800 border border-gray-600 rounded-lg p-6 hover:border-gray-500 transition-colors">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  {getClassIcon(result.predicted_class)}
                  <h3 className="text-xl font-semibold text-white">{result.name}</h3>
                  {result.is_similar && (
                    <span className="px-2 py-1 bg-purple-600 text-purple-200 text-xs rounded-full">
                      Похожий объект
                    </span>
                  )}
                </div>
                
                {result.similarity_reason && (
                  <p className="text-purple-300 text-sm mb-2">{result.similarity_reason}</p>
                )}

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Класс:</span>
                    <p className="text-white font-medium">{result.predicted_class}</p>
                  </div>
                  
                  <div>
                    <span className="text-gray-400">Уверенность:</span>
                    <p className={`font-medium ${getConfidenceColor(result.confidence_score)}`}>
                      {(result.confidence_score * 100).toFixed(1)}%
                    </p>
                  </div>
                  
                  <div>
                    <span className="text-gray-400">ИИ балл:</span>
                    <p className="text-white font-medium">{(result.ai_score * 100).toFixed(0)}/100</p>
                  </div>
                  
                  <div>
                    <span className="text-gray-400">SNR:</span>
                    <p className="text-white font-medium">{result.signal_characteristics.snr_estimate.toFixed(1)}</p>
                  </div>
                </div>

                {result.orbital_period && (
                  <div className="mt-3 text-sm">
                    <span className="text-gray-400">Период:</span>
                    <span className="text-white ml-2">{result.orbital_period.toFixed(2)} дней</span>
                    {result.planet_radius && (
                      <>
                        <span className="text-gray-400 ml-4">Радиус:</span>
                        <span className="text-white ml-2">{result.planet_radius.toFixed(2)} R⊕</span>
                      </>
                    )}
                  </div>
                )}
              </div>

              <button
                onClick={() => handleDetailedAnalysis(result.name)}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm"
              >
                Детальный анализ
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* No Results */}
      {!isLoading && searchQuery && results.length === 0 && (
        <div className="text-center py-12">
          <AlertCircle className="w-16 h-16 text-gray-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-400 mb-2">Результаты не найдены</h3>
          <p className="text-gray-500">Попробуйте изменить поисковый запрос или ослабить фильтры</p>
        </div>
      )}
    </div>
  );
};

export default SmartSearch;
