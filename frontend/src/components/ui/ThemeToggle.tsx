import React, { useState, useEffect } from 'react'
import { Sparkles, Star } from 'lucide-react'

interface BackgroundToggleProps {
  useSimpleBackground?: boolean
  setUseSimpleBackground?: (value: boolean | ((prev: boolean) => boolean)) => void
}

const BackgroundToggle: React.FC<BackgroundToggleProps> = ({ 
  useSimpleBackground = false, 
  setUseSimpleBackground 
}) => {
  // Локальное состояние для немедленной визуальной обратной связи
  const [localState, setLocalState] = useState(useSimpleBackground)

  // Синхронизируем локальное состояние с пропсом
  useEffect(() => {
    setLocalState(useSimpleBackground)
  }, [useSimpleBackground])

  if (!setUseSimpleBackground) {
    return null
  }

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    // Немедленно обновляем локальное состояние для визуального эффекта
    const newState = !localState
    setLocalState(newState)
    
    // Обновляем глобальное состояние
    setUseSimpleBackground(newState)
  }

  return (
    <div className="relative group">
      {/* Glow effect */}
      <div className={`absolute -inset-1 rounded-full opacity-0 group-hover:opacity-100 blur-sm transition-all duration-500 ease-out ${
        localState 
          ? 'bg-gradient-to-r from-gray-500 via-gray-600 to-gray-700' 
          : 'bg-gradient-to-r from-gray-500 via-gray-600 to-gray-700'
      }`} />
      
      <button
        onClick={handleClick}
        className="relative inline-flex items-center justify-center w-16 h-8 rounded-full transition-all duration-500 ease-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 bg-gray-800 hover:bg-gray-700 hover:scale-105 active:scale-95 border border-gray-600 hover:border-gray-500"
        title={`${localState ? 'Простой' : 'Сложный'} фон`}
      >
        {/* Background track */}
        <div className={`absolute inset-1 rounded-full transition-all duration-700 ease-in-out ${
          localState 
            ? 'bg-gradient-to-r from-gray-800 via-gray-900 to-black shadow-inner' 
            : 'bg-gradient-to-r from-gray-600 via-gray-700 to-gray-800 shadow-lg'
        }`} />
        {/* Toggle circle */}
        <div className={`absolute w-6 h-6 bg-gradient-to-br from-white to-gray-100 rounded-full shadow-xl transition-all duration-700 ease-in-out z-10 hover:shadow-2xl ${
          localState ? 'right-1 rotate-180' : 'left-1 rotate-0'
        }`}>
          {/* Icon container with rotation */}
          <div className="absolute inset-0 flex items-center justify-center transition-all duration-500 ease-in-out">
            {/* Sparkles icon */}
            <Sparkles
              size={14}
              className={`absolute text-gray-600 transition-all duration-500 ease-in-out transform ${
                !localState 
                  ? 'opacity-100 scale-100 rotate-0' 
                  : 'opacity-0 scale-50 rotate-180'
              }`}
            />
            {/* Star icon */}
            <Star
              size={14}
              className={`absolute text-gray-800 transition-all duration-500 ease-in-out transform ${
                localState 
                  ? 'opacity-100 scale-100 rotate-0' 
                  : 'opacity-0 scale-50 -rotate-180'
              }`}
            />
          </div>
          
          {/* Inner glow */}
          <div className={`absolute inset-0.5 rounded-full transition-all duration-700 ${
            localState 
              ? 'bg-gradient-to-br from-gray-200 to-transparent opacity-30' 
              : 'bg-gradient-to-br from-gray-200 to-transparent opacity-50'
          }`} />
        </div>

        {/* Status indicator */}
        <div className={`absolute -bottom-1 left-1/2 transform -translate-x-1/2 transition-all duration-500 ease-out ${
          localState 
            ? 'w-3 h-0.5 bg-gray-400 rounded-full' 
            : 'w-2 h-0.5 bg-gray-400 rounded-full shadow-lg shadow-gray-400/50'
        }`} />
        
        {/* Pulse effect */}
        <div className={`absolute inset-0 rounded-full transition-all duration-1000 ease-out ${
          localState 
            ? 'bg-gray-400 opacity-0' 
            : 'bg-gray-400 opacity-0 animate-ping'
        }`} />
      </button>

    </div>
  )
}

export default BackgroundToggle
