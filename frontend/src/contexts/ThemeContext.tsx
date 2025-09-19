import React, { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';

export type ThemeType = 'dark-space' | 'light-professional' | 'deep-space' | 'neon-cyber' | 'warm-sunset' | 'ocean-depths' | 'nasa-mission' | 'auto';

interface ThemeContextType {
  theme: ThemeType;
  setTheme: (theme: ThemeType) => void;
  effectiveTheme: string;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const themeConfigs = {
  'dark-space': {
    name: 'Dark Space',
    description: 'Глубокий космос с фиолетовыми градиентами',
    colors: {
      primary: '#8B5CF6',
      secondary: '#A855F7',
      accent: '#C084FC',
      background: '#0F0F23',
      surface: '#1A1A2E',
      text: '#E2E8F0'
    }
  },
  'light-professional': {
    name: 'Light Professional',
    description: 'Чистый современный светлый дизайн',
    colors: {
      primary: '#3B82F6',
      secondary: '#1E40AF',
      accent: '#60A5FA',
      background: '#F8FAFC',
      surface: '#FFFFFF',
      text: '#1E293B'
    }
  },
  'deep-space': {
    name: 'Deep Space (Cosmic)',
    description: 'Усиленная космическая тема с cyan/purple',
    colors: {
      primary: '#06B6D4',
      secondary: '#8B5CF6',
      accent: '#EC4899',
      background: '#0C0A1E',
      surface: '#1E1B3A',
      text: '#F1F5F9'
    }
  },
  'neon-cyber': {
    name: 'Neon Cyber',
    description: 'Киберпанк с зелеными неоновыми эффектами',
    colors: {
      primary: '#00FF88',
      secondary: '#00D4AA',
      accent: '#FF0080',
      background: '#000011',
      surface: '#0A0A0F',
      text: '#00FF88'
    }
  },
  'warm-sunset': {
    name: 'Warm Sunset',
    description: 'Теплая оранжево-розовая палитра',
    colors: {
      primary: '#F97316',
      secondary: '#EA580C',
      accent: '#FB923C',
      background: '#1C1917',
      surface: '#292524',
      text: '#FEF3C7'
    }
  },
  'ocean-depths': {
    name: 'Ocean Depths',
    description: 'Синяя океанская тема',
    colors: {
      primary: '#0EA5E9',
      secondary: '#0284C7',
      accent: '#38BDF8',
      background: '#0C1426',
      surface: '#1E293B',
      text: '#E0F2FE'
    }
  },
  'nasa-mission': {
    name: 'NASA Mission',
    description: 'Официальный стиль NASA с красно-синими акцентами',
    colors: {
      primary: '#FC3D21', // NASA Red
      secondary: '#0B3D91', // NASA Blue
      accent: '#FFB81C', // NASA Gold
      background: '#0A0E1A',
      surface: '#1A1F2E',
      text: '#FFFFFF'
    }
  }
};

export const ThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [theme, setTheme] = useState<ThemeType>(() => {
    const saved = localStorage.getItem('exoplanet-theme');
    return (saved as ThemeType) || 'auto';
  });

  const [effectiveTheme, setEffectiveTheme] = useState<string>('dark-space');

  useEffect(() => {
    let resolvedTheme = theme;
    
    if (theme === 'auto') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      resolvedTheme = prefersDark ? 'dark-space' : 'light-professional';
    }

    setEffectiveTheme(resolvedTheme);
    applyTheme(resolvedTheme);
    localStorage.setItem('exoplanet-theme', theme);
  }, [theme]);

  const applyTheme = (themeName: string) => {
    const config = themeConfigs[themeName as keyof typeof themeConfigs];
    if (!config) return;

    const root = document.documentElement;
    
    // Удаляем все существующие классы тем
    Object.keys(themeConfigs).forEach(t => {
      document.body.classList.remove(`theme-${t}`);
    });
    
    // Добавляем новый класс темы
    document.body.classList.add(`theme-${themeName}`);
    
    // Устанавливаем CSS переменные
    root.style.setProperty('--color-primary', config.colors.primary);
    root.style.setProperty('--color-secondary', config.colors.secondary);
    root.style.setProperty('--color-accent', config.colors.accent);
    root.style.setProperty('--color-background', config.colors.background);
    root.style.setProperty('--color-surface', config.colors.surface);
    root.style.setProperty('--color-text', config.colors.text);
    
    // Обновляем meta theme-color для мобильных браузеров
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      metaThemeColor.setAttribute('content', config.colors.primary);
    }
  };

  return (
    <ThemeContext.Provider value={{ theme, setTheme, effectiveTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

export const useThemeClasses = () => {
  const { effectiveTheme } = useTheme();
  
  const getThemeClasses = () => {
    switch (effectiveTheme) {
      case 'dark-space':
        return {
          background: 'bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900',
          card: 'bg-slate-800/50 border-purple-500/30',
          button: 'bg-purple-600 hover:bg-purple-700 text-white',
          text: 'text-slate-100',
          accent: 'text-purple-400'
        };
      case 'light-professional':
        return {
          background: 'bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100',
          card: 'bg-white/80 border-blue-200/50',
          button: 'bg-blue-600 hover:bg-blue-700 text-white',
          text: 'text-slate-900',
          accent: 'text-blue-600'
        };
      case 'deep-space':
        return {
          background: 'bg-gradient-to-br from-slate-900 via-cyan-900 to-purple-900',
          card: 'bg-slate-800/60 border-cyan-500/30',
          button: 'bg-cyan-600 hover:bg-cyan-700 text-white',
          text: 'text-slate-100',
          accent: 'text-cyan-400'
        };
      case 'neon-cyber':
        return {
          background: 'bg-gradient-to-br from-black via-gray-900 to-black',
          card: 'bg-gray-900/80 border-green-500/50 shadow-green-500/20',
          button: 'bg-green-500 hover:bg-green-600 text-black',
          text: 'text-green-400',
          accent: 'text-green-300'
        };
      case 'warm-sunset':
        return {
          background: 'bg-gradient-to-br from-orange-900 via-red-900 to-yellow-900',
          card: 'bg-orange-800/50 border-orange-500/30',
          button: 'bg-orange-600 hover:bg-orange-700 text-white',
          text: 'text-orange-100',
          accent: 'text-orange-400'
        };
      case 'ocean-depths':
        return {
          background: 'bg-gradient-to-br from-blue-900 via-cyan-900 to-teal-900',
          card: 'bg-blue-800/50 border-cyan-500/30',
          button: 'bg-cyan-600 hover:bg-cyan-700 text-white',
          text: 'text-blue-100',
          accent: 'text-cyan-400'
        };
      case 'nasa-mission':
        return {
          background: 'bg-gradient-to-br from-slate-900 via-blue-900 to-red-900',
          card: 'bg-slate-800/60 border-red-500/30 shadow-red-500/10',
          button: 'bg-red-600 hover:bg-red-700 text-white',
          text: 'text-white',
          accent: 'text-red-400'
        };
      default:
        return {
          background: 'bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900',
          card: 'bg-slate-800/50 border-purple-500/30',
          button: 'bg-purple-600 hover:bg-purple-700 text-white',
          text: 'text-slate-100',
          accent: 'text-purple-400'
        };
    }
  };

  return getThemeClasses();
};

export { themeConfigs };
