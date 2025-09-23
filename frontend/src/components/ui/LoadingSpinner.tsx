import { motion } from 'framer-motion'
import { Loader2, Radar } from 'lucide-react'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  message?: string
  progress?: number
  variant?: 'default' | 'cosmic'
  className?: string
}

export default function LoadingSpinner({ 
  size = 'md', 
  message, 
  progress,
  variant = 'default',
  className = '' 
}: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-8 w-8',
    lg: 'h-12 w-12'
  }

  const containerSizeClasses = {
    sm: 'p-2',
    md: 'p-4',
    lg: 'p-6'
  }

  if (variant === 'cosmic') {
    return (
      <div className={`flex flex-col items-center justify-center ${containerSizeClasses[size]} ${className}`}>
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className={`${sizeClasses[size]} text-cosmic-400 mb-2`}
        >
          <Radar className="w-full h-full" />
        </motion.div>
        
        {message && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-sm text-gray-300 text-center max-w-xs"
          >
            {message}
          </motion.p>
        )}
        
        {progress !== undefined && (
          <div className="w-32 bg-space-700 rounded-full h-2 mt-2">
            <motion.div
              className="bg-cosmic-gradient h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        )}
      </div>
    )
  }

  return (
    <div className={`flex flex-col items-center justify-center ${containerSizeClasses[size]} ${className}`}>
      <Loader2 className={`${sizeClasses[size]} text-primary-400 animate-spin`} />
      
      {message && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-sm text-gray-300 text-center max-w-xs mt-2"
        >
          {message}
        </motion.p>
      )}
      
      {progress !== undefined && (
        <div className="w-32 bg-space-700 rounded-full h-2 mt-2">
          <motion.div
            className="bg-primary-600 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      )}
    </div>
  )
}
