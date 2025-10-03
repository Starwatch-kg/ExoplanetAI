import React from 'react'
import { Sun, Moon } from 'lucide-react'
import { useTheme } from '../../../../front/frontend/src/contexts/ThemeContext'

const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useTheme()

  return (
    <button
      onClick={toggleTheme}
      className="relative inline-flex items-center justify-center w-12 h-6 bg-gray-200 dark:bg-gray-700 rounded-full transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
      aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
    >
      {/* Toggle background */}
      <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-400 to-purple-500 opacity-0 dark:opacity-100 transition-opacity duration-300" />
      
      {/* Toggle circle */}
      <div
        className={`
          relative flex items-center justify-center w-5 h-5 bg-white dark:bg-gray-800 rounded-full shadow-lg transform transition-transform duration-300 ease-in-out
          ${theme === 'dark' ? 'translate-x-3' : '-translate-x-3'}
        `}
      >
        {/* Icons */}
        <Sun
          size={12}
          className={`
            absolute text-yellow-500 transition-opacity duration-300
            ${theme === 'light' ? 'opacity-100' : 'opacity-0'}
          `}
        />
        <Moon
          size={12}
          className={`
            absolute text-blue-400 transition-opacity duration-300
            ${theme === 'dark' ? 'opacity-100' : 'opacity-0'}
          `}
        />
      </div>
    </button>
  )
}

export default ThemeToggle
