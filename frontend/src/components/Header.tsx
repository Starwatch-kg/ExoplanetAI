import React from 'react';
import { Telescope, Sparkles, Sun, Moon, History, Settings } from 'lucide-react';
import { useApp } from '../App';

const Header: React.FC = () => {
  const { expertMode, setExpertMode, theme, setTheme, isOnline, setShowHistory } = useApp();

  return (
    <header className="relative z-10 py-6 px-6">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <Telescope className={`w-8 h-8 transition-colors duration-300 ${
              theme === 'dark' ? 'text-cyan-400' : 'text-blue-600'
            } animate-pulse`} />
            <Sparkles className={`w-4 h-4 absolute -top-1 -right-1 transition-colors duration-300 ${
              theme === 'dark' ? 'text-yellow-400' : 'text-orange-400'
            } animate-twinkle`} />
          </div>
          <div>
            <h1 className={`text-2xl font-bold bg-gradient-to-r bg-clip-text text-transparent transition-all duration-300 ${
              theme === 'dark'
                ? 'from-cyan-400 to-purple-400'
                : 'from-blue-600 to-purple-600'
            }`}>
              Exoplanet AI
            </h1>
            <div className="flex items-center space-x-2">
              <p className={`text-sm transition-colors duration-300 ${
                theme === 'dark' ? 'text-slate-400' : 'text-gray-600'
              }`}>
                ИИ для поиска экзопланет
              </p>
              {!isOnline && (
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" title="Отсутствует подключение к интернету" />
              )}
            </div>
          </div>
        </div>

        <nav className="hidden md:flex space-x-6">
          <button className={`transition-colors duration-300 hover:scale-105 ${
            theme === 'dark'
              ? 'text-slate-300 hover:text-cyan-400'
              : 'text-gray-600 hover:text-blue-600'
          }`}>
            Главная
          </button>
          <button className={`transition-colors duration-300 hover:scale-105 ${
            theme === 'dark'
              ? 'text-slate-300 hover:text-cyan-400'
              : 'text-gray-600 hover:text-blue-600'
          }`}>
            Поиск
          </button>
          <button className={`transition-colors duration-300 hover:scale-105 ${
            theme === 'dark'
              ? 'text-slate-300 hover:text-cyan-400'
              : 'text-gray-600 hover:text-blue-600'
          }`}>
            Результаты
          </button>
          <button className={`transition-colors duration-300 hover:scale-105 ${
            theme === 'dark'
              ? 'text-slate-300 hover:text-cyan-400'
              : 'text-gray-600 hover:text-blue-600'
          }`}>
            О проекте
          </button>
        </nav>

        <div className="flex items-center space-x-6">
          {/* Переключатель темы */}
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className={`p-2 rounded-lg transition-all duration-300 hover:scale-110 ${
              theme === 'dark'
                ? 'bg-white/10 hover:bg-white/20 text-yellow-400'
                : 'bg-gray-200 hover:bg-gray-300 text-orange-500'
            }`}
            title={`Переключить на ${theme === 'dark' ? 'светлую' : 'тёмную'} тему`}
          >
            {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>

          {/* Режим работы */}
          <div className="flex items-center space-x-4 bg-white/10 backdrop-blur-lg rounded-2xl p-2 border border-white/20">
            <button
              onClick={() => setExpertMode(false)}
              className={`px-4 py-2 rounded-xl font-medium transition-all duration-300 ${
                !expertMode
                  ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg shadow-blue-500/25'
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              🌟 Любитель
            </button>
            <div className="w-px h-6 bg-white/20" />
            <button
              onClick={() => setExpertMode(true)}
              className={`px-4 py-2 rounded-xl font-medium transition-all duration-300 ${
                expertMode
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg shadow-purple-500/25'
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              🚀 Профи
            </button>
          </div>

          {/* Кнопка истории */}
          <button
            onClick={() => setShowHistory(true)}
            className={`p-2 rounded-lg transition-all duration-300 hover:scale-110 ${
              theme === 'dark'
                ? 'bg-white/10 hover:bg-white/20 text-blue-400'
                : 'bg-gray-200 hover:bg-gray-300 text-blue-600'
            }`}
            title="История анализа"
          >
            <History className="w-5 h-5" />
          </button>

          {/* Кнопка настроек */}
          <button
            className={`p-2 rounded-lg transition-all duration-300 hover:scale-110 ${
              theme === 'dark'
                ? 'bg-white/10 hover:bg-white/20 text-gray-400'
                : 'bg-gray-200 hover:bg-gray-300 text-gray-600'
            }`}
            title="Настройки"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
