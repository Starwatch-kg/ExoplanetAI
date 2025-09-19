import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  Brain,
  Camera,
  Settings,
  Activity,
  Target,
  Rocket,
  Menu,
  X
} from 'lucide-react';

// Контексты
import { BackgroundTasksProvider } from './contexts/BackgroundTasksContext';
import { ThemeProvider } from './contexts/ThemeContext';

// Компоненты
import BackgroundTasksIndicator from './components/BackgroundTasksIndicator';
import EnhancedMissionControl from './components/EnhancedMissionControl';
import EnhancedLightcurveAnalysis from './components/EnhancedLightcurveAnalysis';
import EnhancedCNNTraining from './components/EnhancedCNNTraining';
import WorkingImageClassification from './components/WorkingImageClassification';
import EnhancedSettings from './components/EnhancedSettings';
import CosmicParticles from './components/CosmicParticles';

type TabType = 'dashboard' | 'lightcurve' | 'cnn' | 'classification' | 'settings';

interface NavItem {
  id: TabType;
  name: string;
  icon: React.ReactNode;
  description: string;
  color: string;
}

const EnhancedApp: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isNavCollapsed, setIsNavCollapsed] = useState(false);

  // Навигационные элементы
  const navItems: NavItem[] = [
    {
      id: 'dashboard',
      name: 'Mission Control',
      icon: <Rocket className="w-5 h-5" />,
      description: 'Центр управления миссией',
      color: 'blue'
    },
    {
      id: 'lightcurve',
      name: 'Анализ кривых блеска',
      icon: <BarChart3 className="w-5 h-5" />,
      description: 'Поиск транзитов в данных TESS',
      color: 'green'
    },
    {
      id: 'cnn',
      name: 'CNN Обучение',
      icon: <Brain className="w-5 h-5" />,
      description: 'Обучение нейронных сетей',
      color: 'purple'
    },
    {
      id: 'classification',
      name: 'Классификация изображений',
      icon: <Camera className="w-5 h-5" />,
      description: 'Анализ астрономических изображений',
      color: 'orange'
    },
    {
      id: 'settings',
      name: 'Настройки',
      icon: <Settings className="w-5 h-5" />,
      description: 'Конфигурация системы',
      color: 'gray'
    }
  ];

  // Обработчик смены вкладки с анимацией
  const handleTabChange = (tabId: TabType) => {
    setActiveTab(tabId);
    setIsMobileMenuOpen(false);
  };

  // Рендер контента вкладки
  const renderTabContent = () => {
    const contentVariants = {
      initial: { opacity: 0, y: 20 },
      animate: { opacity: 1, y: 0 },
      exit: { opacity: 0, y: -20 }
    };

    switch (activeTab) {
      case 'dashboard':
        return (
          <motion.div
            key="dashboard"
            variants={contentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.3 }}
          >
            <EnhancedMissionControl />
          </motion.div>
        );
      case 'lightcurve':
        return (
          <motion.div
            key="lightcurve"
            variants={contentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.3 }}
          >
            <EnhancedLightcurveAnalysis />
          </motion.div>
        );
      case 'cnn':
        return (
          <motion.div
            key="cnn"
            variants={contentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.3 }}
          >
            <EnhancedCNNTraining />
          </motion.div>
        );
      case 'classification':
        return (
          <motion.div
            key="classification"
            variants={contentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.3 }}
          >
            <WorkingImageClassification />
          </motion.div>
        );
      case 'settings':
        return (
          <motion.div
            key="settings"
            variants={contentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.3 }}
          >
            <EnhancedSettings />
          </motion.div>
        );
      default:
        return null;
    }
  };

  return (
    <ThemeProvider>
      <BackgroundTasksProvider>
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 relative">
          {/* Космические частицы в фоне */}
          <CosmicParticles />
          
          {/* Фоновые задачи индикатор */}
          <BackgroundTasksIndicator />

          {/* Мобильная навигация */}
          <div className="lg:hidden">
            <div className="flex items-center justify-between p-4 bg-gray-800/50 backdrop-blur-sm border-b border-blue-500/30">
              <h1 className="text-xl font-bold text-white flex items-center gap-2">
                <Target className="w-6 h-6 text-blue-400" />
                Exoplanet AI
              </h1>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                className="p-2 text-white hover:bg-gray-700 rounded-lg transition-colors"
              >
                {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </motion.button>
            </div>

            {/* Мобильное меню */}
            <AnimatePresence>
              {isMobileMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="bg-gray-800/95 backdrop-blur-sm border-b border-blue-500/30"
                >
                  <div className="p-4 space-y-2">
                    {navItems.map((item) => (
                      <motion.button
                        key={item.id}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => handleTabChange(item.id)}
                        className={`w-full p-3 rounded-lg text-left transition-all duration-200 ${
                          activeTab === item.id
                            ? `bg-${item.color}-500/20 border-${item.color}-500/50 border`
                            : 'hover:bg-gray-700/50'
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <div className={`text-${item.color}-400`}>
                            {item.icon}
                          </div>
                          <div>
                            <div className="text-white font-medium">{item.name}</div>
                            <div className="text-xs text-gray-400">{item.description}</div>
                          </div>
                        </div>
                      </motion.button>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Десктопная навигация */}
          <div className="hidden lg:flex">
            {/* Боковая панель */}
            <motion.div
              animate={{ width: isNavCollapsed ? 80 : 320 }}
              transition={{ duration: 0.3 }}
              className="bg-gray-800/50 backdrop-blur-sm border-r border-blue-500/30 min-h-screen"
            >
              <div className="p-4">
                {/* Заголовок */}
                <div className="flex items-center justify-between mb-8">
                  {!isNavCollapsed && (
                    <motion.h1
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-xl font-bold text-white flex items-center gap-2"
                    >
                      <Target className="w-6 h-6 text-blue-400" />
                      Exoplanet AI
                    </motion.h1>
                  )}
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setIsNavCollapsed(!isNavCollapsed)}
                    className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                  >
                    <Menu className="w-5 h-5" />
                  </motion.button>
                </div>

                {/* Навигационные элементы */}
                <nav className="space-y-2">
                  {navItems.map((item, index) => (
                    <motion.button
                      key={item.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleTabChange(item.id)}
                      className={`w-full p-3 rounded-lg text-left transition-all duration-200 ${
                        activeTab === item.id
                          ? `bg-${item.color}-500/20 border-${item.color}-500/50 border shadow-lg shadow-${item.color}-500/20`
                          : 'hover:bg-gray-700/50'
                      }`}
                      title={isNavCollapsed ? item.name : undefined}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`text-${item.color}-400 ${activeTab === item.id ? 'animate-pulse' : ''}`}>
                          {item.icon}
                        </div>
                        {!isNavCollapsed && (
                          <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.1 }}
                          >
                            <div className="text-white font-medium">{item.name}</div>
                            <div className="text-xs text-gray-400">{item.description}</div>
                          </motion.div>
                        )}
                      </div>
                      {activeTab === item.id && (
                        <motion.div
                          layoutId="activeTab"
                          className={`absolute left-0 top-0 bottom-0 w-1 bg-${item.color}-400 rounded-r`}
                        />
                      )}
                    </motion.button>
                  ))}
                </nav>

                {/* Статус системы */}
                {!isNavCollapsed && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="mt-8 p-3 bg-gray-700/30 rounded-lg"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-4 h-4 text-green-400" />
                      <span className="text-sm font-medium text-white">Система активна</span>
                    </div>
                    <div className="text-xs text-gray-400">
                      <div>API: Подключен</div>
                      <div>ML модули: Загружены</div>
                      <div>Время работы: 2ч 15м</div>
                    </div>
                  </motion.div>
                )}
              </div>
            </motion.div>

            {/* Основной контент */}
            <div className="flex-1 overflow-auto">
              <div className="p-6">
                <AnimatePresence mode="wait">
                  {renderTabContent()}
                </AnimatePresence>
              </div>
            </div>
          </div>

          {/* Мобильный контент */}
          <div className="lg:hidden">
            <div className="p-4">
              <AnimatePresence mode="wait">
                {renderTabContent()}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </BackgroundTasksProvider>
    </ThemeProvider>
  );
};

export default EnhancedApp;
