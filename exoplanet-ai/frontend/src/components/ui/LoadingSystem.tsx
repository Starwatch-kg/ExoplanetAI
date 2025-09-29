import React from 'react'
import { Loader2, Telescope, Zap, Brain, Database } from 'lucide-react'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  variant?: 'default' | 'pulse' | 'bounce' | 'orbit'
  color?: 'blue' | 'purple' | 'green' | 'yellow' | 'red'
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  variant = 'default',
  color = 'blue'
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  }

  const colorClasses = {
    blue: 'text-blue-500',
    purple: 'text-purple-500',
    green: 'text-green-500',
    yellow: 'text-yellow-500',
    red: 'text-red-500'
  }

  if (variant === 'orbit') {
    return (
      <div className="relative inline-flex items-center justify-center">
        <div className={`${sizeClasses[size]} relative`}>
          <div className={`absolute inset-0 border-2 border-transparent border-t-current ${colorClasses[color]} rounded-full animate-spin`} />
          <div className={`absolute inset-1 border-2 border-transparent border-t-current ${colorClasses[color]} rounded-full animate-spin`} style={{ animationDirection: 'reverse', animationDuration: '0.75s' }} />
        </div>
      </div>
    )
  }

  if (variant === 'bounce') {
    return (
      <div className="flex space-x-1">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className={`${sizeClasses.sm} ${colorClasses[color]} bg-current rounded-full animate-bounce`}
            style={{ animationDelay: `${i * 0.1}s` }}
          />
        ))}
      </div>
    )
  }

  if (variant === 'pulse') {
    return (
      <div className={`${sizeClasses[size]} ${colorClasses[color]} bg-current rounded-full animate-pulse`} />
    )
  }

  return (
    <Loader2 className={`${sizeClasses[size]} ${colorClasses[color]} animate-spin`} />
  )
}

interface LoadingOverlayProps {
  isVisible: boolean
  message?: string
  progress?: number
  variant?: 'default' | 'astro' | 'gpi' | 'ai'
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  isVisible,
  message = 'Loading...',
  progress,
  variant = 'default'
}) => {
  if (!isVisible) return null

  const getIcon = () => {
    switch (variant) {
      case 'astro':
        return <Telescope className="w-12 h-12 text-blue-400 animate-pulse" />
      case 'gpi':
        return <Zap className="w-12 h-12 text-purple-400 animate-pulse" />
      case 'ai':
        return <Brain className="w-12 h-12 text-green-400 animate-pulse" />
      default:
        return <Database className="w-12 h-12 text-blue-400 animate-pulse" />
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-gray-900/90 backdrop-blur-md border border-white/10 rounded-lg p-8 max-w-sm w-full mx-4 text-center">
        <div className="flex justify-center mb-4">
          {getIcon()}
        </div>
        
        <h3 className="text-lg font-semibold text-white mb-2">
          {message}
        </h3>
        
        {progress !== undefined && (
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span>Progress</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
        
        <LoadingSpinner size="lg" variant="orbit" />
      </div>
    </div>
  )
}

interface SkeletonProps {
  className?: string
  variant?: 'text' | 'rectangular' | 'circular'
  animation?: 'pulse' | 'wave'
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  variant = 'rectangular',
  animation = 'pulse'
}) => {
  const baseClasses = 'bg-gray-700/50'
  
  const variantClasses = {
    text: 'h-4 rounded',
    rectangular: 'rounded-lg',
    circular: 'rounded-full'
  }
  
  const animationClasses = {
    pulse: 'animate-pulse',
    wave: 'shimmer-dark'
  }

  return (
    <div
      className={`
        ${baseClasses}
        ${variantClasses[variant]}
        ${animationClasses[animation]}
        ${className}
      `}
    />
  )
}

interface LoadingCardProps {
  title?: string
  lines?: number
  showAvatar?: boolean
  showActions?: boolean
}

export const LoadingCard: React.FC<LoadingCardProps> = ({
  title,
  lines = 3,
  showAvatar = false,
  showActions = false
}) => {
  return (
    <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-6">
      <div className="flex items-start space-x-4">
        {showAvatar && (
          <Skeleton variant="circular" className="w-12 h-12" />
        )}
        
        <div className="flex-1 space-y-3">
          {title && (
            <Skeleton className="h-6 w-3/4" />
          )}
          
          {Array.from({ length: lines }).map((_, i) => (
            <Skeleton
              key={i}
              className={`h-4 ${i === lines - 1 ? 'w-1/2' : 'w-full'}`}
            />
          ))}
          
          {showActions && (
            <div className="flex space-x-2 pt-2">
              <Skeleton className="h-8 w-20" />
              <Skeleton className="h-8 w-16" />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

interface LoadingListProps {
  items?: number
  showAvatar?: boolean
}

export const LoadingList: React.FC<LoadingListProps> = ({
  items = 5,
  showAvatar = false
}) => {
  return (
    <div className="space-y-4">
      {Array.from({ length: items }).map((_, i) => (
        <LoadingCard
          key={i}
          lines={2}
          showAvatar={showAvatar}
        />
      ))}
    </div>
  )
}

interface LoadingTableProps {
  rows?: number
  columns?: number
}

export const LoadingTable: React.FC<LoadingTableProps> = ({
  rows = 5,
  columns = 4
}) => {
  return (
    <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="border-b border-white/10 p-4">
        <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
          {Array.from({ length: columns }).map((_, i) => (
            <Skeleton key={i} className="h-5 w-20" />
          ))}
        </div>
      </div>
      
      {/* Rows */}
      <div className="divide-y divide-white/10">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div key={rowIndex} className="p-4">
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
              {Array.from({ length: columns }).map((_, colIndex) => (
                <Skeleton
                  key={colIndex}
                  className={`h-4 ${colIndex === 0 ? 'w-24' : 'w-16'}`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// Loading state hook
export const useLoading = (initialState = false) => {
  const [isLoading, setIsLoading] = React.useState(initialState)
  const [progress, setProgress] = React.useState(0)
  const [message, setMessage] = React.useState('')

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
