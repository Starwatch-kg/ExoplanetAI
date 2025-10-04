import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Zap, Layers, Settings, CheckCircle } from 'lucide-react';

export type ModelType = 'autoencoder' | 'classifier' | 'hybrid' | 'ensemble';

interface ModelSelectorProps {
  selectedModel: ModelType;
  onModelChange: (model: ModelType) => void;
  onStartAnalysis: () => void;
  isAnalyzing: boolean;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onModelChange,
  onStartAnalysis,
  isAnalyzing
}) => {
  const models = [
    {
      id: 'autoencoder' as ModelType,
      name: 'Autoencoder',
      description: 'Автоэнкодер для детекции аномалий в кривых блеска',
      icon: <Brain className="w-8 h-8" />,
      color: 'from-cyan-500 to-blue-600',
      features: ['Обнаружение аномалий', 'Низкий уровень ложных срабатываний', 'Быстрая обработка']
    },
    {
      id: 'classifier' as ModelType,
      name: 'Classifier',
      description: 'Классификатор для бинарной классификации транзитов',
      icon: <Zap className="w-8 h-8" />,
      color: 'from-purple-500 to-pink-600',
      features: ['Высокая точность', 'Обучение на размеченных данных', 'Интерпретируемые результаты']
    },
    {
      id: 'hybrid' as ModelType,
      name: 'Hybrid (BLS + NN)',
      description: 'Гибридный подход: Box Least Squares + нейронные сети',
      icon: <Layers className="w-8 h-8" />,
      color: 'from-green-500 to-teal-600',
      features: ['Комбинированный анализ', 'Максимальная точность', 'Робастность к шуму']
    },
    {
      id: 'ensemble' as ModelType,
      name: 'Ensemble',
      description: 'Ансамбль всех моделей для финального решения',
      icon: <Settings className="w-8 h-8" />,
      color: 'from-orange-500 to-red-600',
      features: ['Максимальная надежность', 'Консенсус моделей', 'Высокое качество']
    }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-card max-w-6xl mx-auto"
    >
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-4">
          Выбор модели ИИ
        </h2>
        <p className="text-slate-400">
          Выберите алгоритм машинного обучения для анализа кривой блеска
        </p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {models.map((model) => (
          <motion.div
            key={model.id}
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onModelChange(model.id)}
            className={`relative cursor-pointer rounded-xl p-6 border-2 transition-all duration-300 ${
              selectedModel === model.id
                ? 'border-cyan-400 bg-cyan-500/10 shadow-lg shadow-cyan-500/25'
                : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
            }`}
          >
            {selectedModel === model.id && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="absolute -top-2 -right-2 w-6 h-6 bg-cyan-400 rounded-full flex items-center justify-center"
              >
                <CheckCircle className="w-4 h-4 text-white" />
              </motion.div>
            )}

            <div className={`inline-flex p-3 rounded-lg bg-gradient-to-r ${model.color} mb-4`}>
              {model.icon}
            </div>

            <h3 className="text-xl font-semibold text-white mb-2">
              {model.name}
            </h3>

            <p className="text-slate-400 text-sm mb-4 leading-relaxed">
              {model.description}
            </p>

            <ul className="space-y-1">
              {model.features.map((feature, index) => (
                <li key={index} className="text-xs text-slate-500 flex items-center">
                  <div className="w-1 h-1 bg-cyan-400 rounded-full mr-2" />
                  {feature}
                </li>
              ))}
            </ul>
          </motion.div>
        ))}
      </div>

      {/* Model Details */}
      <motion.div
        key={selectedModel}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-slate-800/50 rounded-lg p-6 mb-8"
      >
        <h3 className="text-lg font-semibold text-white mb-4">
          Детали выбранной модели
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-medium text-slate-300 mb-2">Параметры</h4>
            <div className="space-y-2 text-sm text-slate-400">
              {selectedModel === 'autoencoder' && (
                <>
                  <div>• Архитектура: Encoder-Decoder</div>
                  <div>• Скрытое пространство: 64D</div>
                  <div>• Порог аномалии: 0.1</div>
                </>
              )}
              {selectedModel === 'classifier' && (
                <>
                  <div>• Архитектура: 1D CNN + ResNet</div>
                  <div>• Классы: Транзит / Нет транзита</div>
                  <div>• Точность: 99.2%</div>
                </>
              )}
              {selectedModel === 'hybrid' && (
                <>
                  <div>• BLS периодограмма + CNN</div>
                  <div>• Диапазон периодов: 0.5-25 дней</div>
                  <div>• Комбинированная оценка</div>
                </>
              )}
              {selectedModel === 'ensemble' && (
                <>
                  <div>• Все модели + голосование</div>
                  <div>• Веса: Autoencoder(0.3), Classifier(0.4), Hybrid(0.3)</div>
                  <div>• Финальная уверенность</div>
                </>
              )}
            </div>
          </div>
          <div>
            <h4 className="text-sm font-medium text-slate-300 mb-2">Производительность</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Точность:</span>
                <span className="text-cyan-400">
                  {selectedModel === 'autoencoder' && '98.5%'}
                  {selectedModel === 'classifier' && '99.2%'}
                  {selectedModel === 'hybrid' && '99.7%'}
                  {selectedModel === 'ensemble' && '99.8%'}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Время обработки:</span>
                <span className="text-cyan-400">
                  {selectedModel === 'autoencoder' && '~2 сек'}
                  {selectedModel === 'classifier' && '~3 сек'}
                  {selectedModel === 'hybrid' && '~5 сек'}
                  {selectedModel === 'ensemble' && '~8 сек'}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">F1-Score:</span>
                <span className="text-cyan-400">
                  {selectedModel === 'autoencoder' && '0.97'}
                  {selectedModel === 'classifier' && '0.98'}
                  {selectedModel === 'hybrid' && '0.99'}
                  {selectedModel === 'ensemble' && '0.995'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Start Analysis Button */}
      <div className="text-center">
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onStartAnalysis}
          disabled={isAnalyzing}
          className="cosmic-button text-lg px-8 py-4 inline-flex items-center space-x-3 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isAnalyzing ? (
            <>
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Анализ в процессе...</span>
            </>
          ) : (
            <>
              <Brain className="w-6 h-6" />
              <span>Запустить анализ</span>
            </>
          )}
        </motion.button>
      </div>
    </motion.div>
  );
};

export default ModelSelector;
