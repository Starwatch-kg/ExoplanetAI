<<<<<<< HEAD
<<<<<<< HEAD
import React, { useState, useEffect, useMemo, useCallback, lazy, Suspense, createContext, useContext } from 'react';
import StarBackground from './components/StarBackground';
import Header from './components/Header';
import AnalysisHistory from './components/AnalysisHistory';
import LatestAnalyses from './components/LatestAnalyses';

// Lazy load компонентов
const AmateurView = lazy(() => import('./pages/AmateurView'));
const ProView = lazy(() => import('./pages/ProView'));

// Типы для контекста
export interface AnalysisHistoryItem {
  id: string;
  ticId: string;
  mode: 'amateur' | 'pro';
  timestamp: Date;
  result?: any;
}

interface AppContextType {
  expertMode: boolean;
  setExpertMode: (mode: boolean) => void;
  theme: 'dark' | 'light';
  setTheme: (theme: 'dark' | 'light') => void;
  isOnline: boolean;
  analysisHistory: AnalysisHistoryItem[];
  addToHistory: (item: AnalysisHistoryItem) => void;
  showHistory: boolean;
  setShowHistory: (show: boolean) => void;
}

// Создание контекста
const AppContext = createContext<AppContextType | null>(null);

// Хук для использования контекста
export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
};

const App: React.FC = () => {
  // Состояние приложения
  const [expertMode, setExpertMode] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // Обработка изменения статуса сети
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Загрузка истории из localStorage при монтировании
  useEffect(() => {
    const savedHistory = localStorage.getItem('exoplanet-history');
    if (savedHistory) {
      try {
        const parsed = JSON.parse(savedHistory);
        // Конвертируем строки дат обратно в Date объекты
        const historyWithDates = parsed.map((item: any) => ({
          ...item,
          timestamp: new Date(item.timestamp)
        }));
        setAnalysisHistory(historyWithDates);
      } catch (error) {
        console.error('Ошибка загрузки истории:', error);
      }
    }
  }, []);

  // Сохранение истории в localStorage
  useEffect(() => {
    localStorage.setItem('exoplanet-history', JSON.stringify(analysisHistory));
  }, [analysisHistory]);

  // Функция для добавления в историю
  const addToHistory = useCallback((item: AnalysisHistoryItem) => {
    setAnalysisHistory(prev => [item, ...prev.slice(0, 49)]); // Храним последние 50 записей
  }, []);

  // Мемоизированное значение контекста
  const contextValue = useMemo<AppContextType>(() => ({
    expertMode,
    setExpertMode,
    theme,
    setTheme,
    isOnline,
    analysisHistory,
    addToHistory,
    showHistory,
    setShowHistory
  }), [expertMode, theme, isOnline, analysisHistory, addToHistory, showHistory]);

  // Обработка переключения темы
  useEffect(() => {
    document.documentElement.classList.toggle('light-theme', theme === 'light');
  }, [theme]);

  return (
    <AppContext.Provider value={contextValue}>
      <div className={`min-h-screen text-white transition-all duration-300 ${theme === 'light' ? 'bg-gray-50 text-gray-900' : ''}`}>
        <StarBackground />
        <Header />
        <main className="relative z-10">
          <Suspense
            fallback={
              <div className="flex items-center justify-center min-h-[50vh]">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                  <p className="text-lg">Загрузка компонента...</p>
                </div>
              </div>
            }
          >
            {showHistory ? (
              <div className="min-h-screen p-6">
                <div className="max-w-6xl mx-auto">
                  <button
                    onClick={() => setShowHistory(false)}
                    className={`mb-6 px-4 py-2 rounded-lg transition-all duration-300 ${
                      theme === 'dark'
                        ? 'bg-white/10 hover:bg-white/20 text-white'
                        : 'bg-gray-200 hover:bg-gray-300 text-gray-900'
                    }`}
                  >
                    ← Назад к анализу
                  </button>
                  <AnalysisHistory />
                </div>
              </div>
            ) : expertMode ? (
              <ProView />
            ) : (
              <div className="max-w-7xl mx-auto p-6">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Основной анализ */}
                  <div className="lg:col-span-2">
                    <AmateurView />
                  </div>

                  {/* Боковая панель */}
                  <div className="space-y-6">
                    <LatestAnalyses />
                  </div>
                </div>
              </div>
            )}
          </Suspense>
        </main>

        {/* Индикатор статуса сети */}
        {!isOnline && (
          <div className="fixed bottom-4 left-4 bg-red-500/20 backdrop-blur-lg border border-red-500/30 rounded-lg px-4 py-2 text-red-300 text-sm">
            Отсутствует подключение к интернету
          </div>
        )}
      </div>
    </AppContext.Provider>
  );
};

export default App;
=======
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
import { Routes, Route } from 'react-router-dom'
import { motion } from 'framer-motion'
import Navbar from './components/layout/Navbar'
import HomePage from './pages/HomePage'
import SearchPage from './pages/SearchPage'
<<<<<<< HEAD
import AnalysisPage from './pages/AnalysisPage'
=======
import CatalogPage from './pages/CatalogPageNew'
import BLSPage from './pages/BLSPage'
>>>>>>> 975c3a7 (Версия 1.5.1)
import ResultsPage from './pages/ResultsPage'
import AboutPage from './pages/AboutPage'
import NotFoundPage from './pages/NotFoundPage'
import { Toaster } from './components/ui/Toaster'
<<<<<<< HEAD

function App() {
  return (
    <div className="min-h-screen bg-space-gradient">
=======
import CleanBackground from './components/CleanBackground'

function App() {
  return (
    <div className="min-h-screen relative">
      {/* Clean NASA Background */}
      <CleanBackground />
      
>>>>>>> 975c3a7 (Версия 1.5.1)
      <Navbar />
      
      <motion.main
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
<<<<<<< HEAD
        className="relative"
=======
        className="relative z-10"
>>>>>>> 975c3a7 (Версия 1.5.1)
      >
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/search" element={<SearchPage />} />
<<<<<<< HEAD
          <Route path="/analysis/:targetId?" element={<AnalysisPage />} />
=======
          <Route path="/catalog" element={<CatalogPage />} />
          <Route path="/bls" element={<BLSPage />} />
          {/* <Route path="/analysis/:targetId?" element={<AnalysisPage />} /> */}
>>>>>>> 975c3a7 (Версия 1.5.1)
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </motion.main>
      
<<<<<<< HEAD
=======
      {/* Toaster для уведомлений */}
>>>>>>> 975c3a7 (Версия 1.5.1)
      <Toaster />
    </div>
  )
}

export default App
<<<<<<< HEAD
>>>>>>> ef5c656 (Версия 1.5.1)
=======
>>>>>>> 975c3a7 (Версия 1.5.1)
