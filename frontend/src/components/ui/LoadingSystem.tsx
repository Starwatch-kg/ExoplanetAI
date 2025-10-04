import React, { memo, useState } from 'react'

interface LoadingSystemProps {
  message?: string
  size?: 'sm' | 'md' | 'lg'
  variant?: 'spinner' | 'pulse' | 'dots' | 'cosmic'
}

const LoadingSystem: React.FC<LoadingSystemProps> = ({
  message = 'Loading...',
  size = 'md',
  variant = 'cosmic'
}) => {
  const sizeClasses = {
    sm: 'w-6 h-6',
    md: 'w-12 h-12',
    lg: 'w-16 h-16'
  }

  const renderSpinner = () => (
    <div className={`animate-spin rounded-full border-b-2 border-blue-500 ${sizeClasses[size]}`} />
  )

  const renderPulse = () => (
    <div className={`animate-pulse rounded-full bg-blue-500 ${sizeClasses[size]}`} />
  )

  const renderDots = () => (
    <div className="flex space-x-2">
      <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
      <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
      <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
    </div>
  )

  const renderCosmic = () => (
    <div className="relative">
      <div className={`animate-spin rounded-full border-2 border-transparent bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 ${sizeClasses[size]}`}>
        <div className="absolute inset-2 bg-black rounded-full"></div>
      </div>
      <div className="absolute inset-0 animate-ping rounded-full bg-blue-500 opacity-20"></div>
    </div>
  )

  const renderLoader = () => {
    switch (variant) {
      case 'pulse':
        return renderPulse()
      case 'dots':
        return renderDots()
      case 'cosmic':
        return renderCosmic()
      default:
        return renderSpinner()
    }
  }

  return (
    <div className="flex flex-col items-center justify-center p-8">
      {renderLoader()}
      {message && (
        <p className="mt-4 text-gray-600 dark:text-gray-300 text-center animate-pulse">
          {message}
        </p>
      )}
    </div>
  )
}

export default memo(LoadingSystem)

// Loading state hook
export const useLoading = (initialState = false) => {
  const [isLoading, setIsLoading] = useState(initialState)
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState('')

  const startLoading = (msg = 'Loading...') => {
    setIsLoading(true)
    setMessage(msg)
    setProgress(0)
  }

  const updateProgress = (value: number, msg?: string) => {
    setProgress(Math.max(0, Math.min(100, value)))
    if (msg) setMessage(msg)
  }

  const stopLoading = () => {
    setIsLoading(false)
    setProgress(0)
    setMessage('')
  }

  return {
    isLoading,
    progress,
    message,
    startLoading,
    updateProgress,
    stopLoading
  }
}
