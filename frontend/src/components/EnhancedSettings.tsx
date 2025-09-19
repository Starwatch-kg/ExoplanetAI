import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Settings,
  Monitor,
  Palette,
  Bell,
  Globe,
  Save,
  RotateCcw,
  Download,
  Upload,
  Zap,
  Eye,
  Sliders,
  BarChart3,
  Brain
} from 'lucide-react';

interface SettingsData {
  // Общие настройки
  theme: 'dark' | 'light' | 'auto';
  language: 'ru' | 'en';
  notifications: boolean;
  sounds: boolean;
  
  // Настройки производительности
  animationsEnabled: boolean;
  particlesEnabled: boolean;
  autoRefresh: boolean;
  refreshInterval: number;
  
  // Настройки анализа
  defaultTICSource: 'real' | 'simulated';
  analysisTimeout: number;
  maxConcurrentAnalyses: number;
  
  // Настройки CNN
  defaultCNNModel: string;
  imageQuality: 'low' | 'medium' | 'high';
  batchSize: number;
  
  // Настройки API
  apiTimeout: number;
  retryAttempts: number;
  cacheEnabled: boolean;
  
  // Настройки интерфейса
  compactMode: boolean;
  showAdvancedOptions: boolean;
  autoSaveResults: boolean;
}

const defaultSettings: SettingsData = {
  theme: 'dark',
  language: 'ru',
  notifications: true,
  sounds: true,
  animationsEnabled: true,
  particlesEnabled: true,
  autoRefresh: true,
  refreshInterval: 5000,
  defaultTICSource: 'real',
  analysisTimeout: 300000,
  maxConcurrentAnalyses: 3,
  defaultCNNModel: 'exoplanet_cnn',
  imageQuality: 'high',
  batchSize: 10,
  apiTimeout: 30000,
  retryAttempts: 3,
  cacheEnabled: true,
  compactMode: false,
  showAdvancedOptions: false,
  autoSaveResults: true
};

const EnhancedSettings: React.FC = () => {
  const [settings, setSettings] = useState<SettingsData>(defaultSettings);
  const [activeSection, setActiveSection] = useState('general');
  const [hasChanges, setHasChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Загрузка настроек из localStorage при монтировании
  useEffect(() => {
    const savedSettings = localStorage.getItem('exoplanet-ai-settings');
    if (savedSettings) {
      try {
        const parsed = JSON.parse(savedSettings);
        setSettings({ ...defaultSettings, ...parsed });
      } catch (error) {
        console.error('Ошибка загрузки настроек:', error);
      }
    }
  }, []);

  // Обновление настройки
  const updateSetting = (key: keyof SettingsData, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  // Сохранение настроек
  const saveSettings = async () => {
    setIsSaving(true);
    try {
      localStorage.setItem('exoplanet-ai-settings', JSON.stringify(settings));
      
      // Имитация API запроса
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setHasChanges(false);
      
      // Применение настроек к интерфейсу
      applySettings();
      
    } catch (error) {
      console.error('Ошибка сохранения настроек:', error);
    } finally {
      setIsSaving(false);
    }
  };

  // Применение настроек к интерфейсу
  const applySettings = () => {
    // Применение темы
    document.documentElement.setAttribute('data-theme', settings.theme);
    
    // Применение других настроек...
    if (!settings.animationsEnabled) {
      document.documentElement.style.setProperty('--transition-normal', '0s');
    } else {
      document.documentElement.style.removeProperty('--transition-normal');
    }
  };

  // Сброс настроек
  const resetSettings = () => {
    setSettings(defaultSettings);
    setHasChanges(true);
  };

  // Экспорт настроек
  const exportSettings = () => {
    const dataStr = JSON.stringify(settings, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'exoplanet-ai-settings.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  // Импорт настроек
  const importSettings = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const imported = JSON.parse(e.target?.result as string);
          setSettings({ ...defaultSettings, ...imported });
          setHasChanges(true);
        } catch (error) {
          console.error('Ошибка импорта настроек:', error);
        }
      };
      reader.readAsText(file);
    }
  };

  const sections = [
    { id: 'general', name: 'Общие', icon: <Settings className="w-5 h-5" /> },
    { id: 'interface', name: 'Интерфейс', icon: <Monitor className="w-5 h-5" /> },
    { id: 'performance', name: 'Производительность', icon: <Zap className="w-5 h-5" /> },
    { id: 'analysis', name: 'Анализ', icon: <BarChart3 className="w-5 h-5" /> },
    { id: 'cnn', name: 'CNN Модели', icon: <Brain className="w-5 h-5" /> },
    { id: 'api', name: 'API', icon: <Globe className="w-5 h-5" /> }
  ];

  const renderGeneralSettings = () => (
    <div className="space-y-lg">
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <Palette className="w-5 h-5 text-accent" />
            Внешний вид
          </h3>
        </div>
        <div className="card-body space-y-md">
          <div>
            <label className="block text-sm font-medium text-primary mb-2">Тема</label>
            <select
              value={settings.theme}
              onChange={(e) => updateSetting('theme', e.target.value)}
              className="form-select w-full"
            >
              <option value="dark">Темная</option>
              <option value="light">Светлая</option>
              <option value="auto">Автоматическая</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-primary mb-2">Язык</label>
            <select
              value={settings.language}
              onChange={(e) => updateSetting('language', e.target.value)}
              className="form-select w-full"
            >
              <option value="ru">Русский</option>
              <option value="en">English</option>
            </select>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <Bell className="w-5 h-5 text-warning" />
            Уведомления
          </h3>
        </div>
        <div className="card-body space-y-md">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-primary">Показывать уведомления</div>
              <div className="text-xs text-tertiary">Уведомления о завершении анализа</div>
            </div>
            <button
              onClick={() => updateSetting('notifications', !settings.notifications)}
              className={`toggle-switch ${settings.notifications ? 'active' : 'inactive'}`}
            >
              <span className="toggle-switch-thumb" />
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-primary">Звуковые уведомления</div>
              <div className="text-xs text-tertiary">Звуки при завершении задач</div>
            </div>
            <button
              onClick={() => updateSetting('sounds', !settings.sounds)}
              className={`toggle-switch ${settings.sounds ? 'active' : 'inactive'}`}
            >
              <span className="toggle-switch-thumb" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderInterfaceSettings = () => (
    <div className="space-y-lg">
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <Eye className="w-5 h-5 text-accent" />
            Отображение
          </h3>
        </div>
        <div className="card-body space-y-md">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-primary">Компактный режим</div>
              <div className="text-xs text-tertiary">Уменьшенные отступы и размеры</div>
            </div>
            <button
              onClick={() => updateSetting('compactMode', !settings.compactMode)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                settings.compactMode ? 'bg-accent' : 'bg-tertiary'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  settings.compactMode ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-primary">Расширенные опции</div>
              <div className="text-xs text-tertiary">Показывать дополнительные настройки</div>
            </div>
            <button
              onClick={() => updateSetting('showAdvancedOptions', !settings.showAdvancedOptions)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                settings.showAdvancedOptions ? 'bg-accent' : 'bg-tertiary'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  settings.showAdvancedOptions ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-primary">Автосохранение результатов</div>
              <div className="text-xs text-tertiary">Сохранять результаты анализа автоматически</div>
            </div>
            <button
              onClick={() => updateSetting('autoSaveResults', !settings.autoSaveResults)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                settings.autoSaveResults ? 'bg-accent' : 'bg-tertiary'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  settings.autoSaveResults ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderPerformanceSettings = () => (
    <div className="space-y-lg">
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <Zap className="w-5 h-5 text-warning" />
            Анимации и эффекты
          </h3>
        </div>
        <div className="card-body space-y-md">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-primary">Анимации</div>
              <div className="text-xs text-tertiary">Плавные переходы и анимации</div>
            </div>
            <button
              onClick={() => updateSetting('animationsEnabled', !settings.animationsEnabled)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                settings.animationsEnabled ? 'bg-accent' : 'bg-tertiary'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  settings.animationsEnabled ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-primary">Космические частицы</div>
              <div className="text-xs text-tertiary">Анимированные частицы в фоне</div>
            </div>
            <button
              onClick={() => updateSetting('particlesEnabled', !settings.particlesEnabled)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                settings.particlesEnabled ? 'bg-accent' : 'bg-tertiary'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  settings.particlesEnabled ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-primary mb-2">
              Интервал обновления (мс)
            </label>
            <input
              type="range"
              min="1000"
              max="30000"
              step="1000"
              value={settings.refreshInterval}
              onChange={(e) => updateSetting('refreshInterval', parseInt(e.target.value))}
              className="w-full range-slider"
            />
            <div className="text-xs text-tertiary mt-1">
              Текущее значение: {settings.refreshInterval}мс
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderContent = () => {
    switch (activeSection) {
      case 'general':
        return renderGeneralSettings();
      case 'interface':
        return renderInterfaceSettings();
      case 'performance':
        return renderPerformanceSettings();
      default:
        return (
          <div className="card">
            <div className="card-body text-center py-12">
              <Sliders className="w-16 h-16 text-tertiary mx-auto mb-4" />
              <p className="text-secondary">Раздел настроек в разработке...</p>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="space-y-lg">
      {/* Заголовок */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card hover-lift neon-glow"
      >
        <div className="card-header">
          <h2 className="card-title gradient-text">
            <Settings className="w-6 h-6 text-accent" />
            Настройки системы
          </h2>
          <div className="text-sm text-secondary">
            Конфигурация параметров Exoplanet AI
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-lg">
        {/* Боковое меню */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="space-y-sm"
        >
          <div className="card">
            <div className="card-body p-sm">
              <nav className="space-y-xs">
                {sections.map((section) => (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center gap-sm p-sm rounded-lg text-left transition-all ${
                      activeSection === section.id
                        ? 'bg-accent/20 text-accent border border-accent/30'
                        : 'text-secondary hover:bg-glass hover:text-primary'
                    }`}
                  >
                    {section.icon}
                    <span className="text-sm font-medium">{section.name}</span>
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Кнопки управления */}
          <div className="card">
            <div className="card-body p-sm space-y-sm">
              <button
                onClick={saveSettings}
                disabled={!hasChanges || isSaving}
                className="btn btn-primary w-full hover-lift-strong"
              >
                <Save className="w-4 h-4" />
                {isSaving ? 'Сохранение...' : 'Сохранить'}
              </button>
              
              <button
                onClick={resetSettings}
                className="btn btn-secondary w-full hover-glow"
              >
                <RotateCcw className="w-4 h-4" />
                Сбросить
              </button>
              
              <div className="flex gap-sm">
                <button
                  onClick={exportSettings}
                  className="btn btn-secondary flex-1 hover-glow"
                >
                  <Download className="w-4 h-4" />
                </button>
                <label className="btn btn-secondary flex-1 hover-glow cursor-pointer">
                  <Upload className="w-4 h-4" />
                  <input
                    type="file"
                    accept=".json"
                    onChange={importSettings}
                    className="hidden"
                  />
                </label>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Основной контент */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-3"
        >
          {renderContent()}
        </motion.div>
      </div>

      {/* Индикатор изменений */}
      {hasChanges && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-6 right-6 bg-warning/90 backdrop-blur-sm border border-warning/30 rounded-lg p-4 shadow-xl"
        >
          <div className="flex items-center gap-2 text-warning">
            <Bell className="w-5 h-5" />
            <span className="text-sm font-medium">Есть несохраненные изменения</span>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default EnhancedSettings;
