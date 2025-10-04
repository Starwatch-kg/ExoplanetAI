import React, { useState, useEffect, createContext, useContext } from 'react'
import { X, CheckCircle, AlertTriangle, Info, AlertCircle } from 'lucide-react'

interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  duration?: number
  persistent?: boolean
  action?: {
    label: string
    onClick: () => void
  }
}

interface NotificationContextType {
  notifications: Notification[]
  addNotification: (notification: Omit<Notification, 'id'>) => void
  removeNotification: (id: string) => void
  clearAll: () => void
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined)

export const useNotifications = () => {
  const context = useContext(NotificationContext)
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider')
  }
  return context
}

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [notifications, setNotifications] = useState<Notification[]>([])

  const addNotification = (notification: Omit<Notification, 'id'>) => {
    const id = Math.random().toString(36).substr(2, 9)
    const newNotification: Notification = {
      id,
      duration: 5000,
      ...notification
    }

    setNotifications(prev => [...prev, newNotification])

    // Auto remove if not persistent
    if (!newNotification.persistent && newNotification.duration) {
      setTimeout(() => {
        removeNotification(id)
      }, newNotification.duration)
    }
  }

  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id))
  }

  const clearAll = () => {
    setNotifications([])
  }

  return (
    <NotificationContext.Provider value={{
      notifications,
      addNotification,
      removeNotification,
      clearAll
    }}>
      {children}
      <NotificationContainer />
    </NotificationContext.Provider>
  )
}

const NotificationContainer: React.FC = () => {
  const { notifications, removeNotification } = useNotifications()

  return (
    <div className="fixed top-4 right-4 z-50 space-y-3 max-w-sm">
      {notifications.map((notification) => (
        <NotificationItem
          key={notification.id}
          notification={notification}
          onRemove={removeNotification}
        />
      ))}
    </div>
  )
}

const NotificationItem: React.FC<{
  notification: Notification
  onRemove: (id: string) => void
}> = ({ notification, onRemove }) => {
  const [isVisible, setIsVisible] = useState(false)
  const [isRemoving, setIsRemoving] = useState(false)

  useEffect(() => {
    // Trigger entrance animation
    setTimeout(() => setIsVisible(true), 10)
  }, [])

  const handleRemove = () => {
    setIsRemoving(true)
    setTimeout(() => onRemove(notification.id), 300)
  }

  const getIcon = () => {
    switch (notification.type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-400" />
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-400" />
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />
      case 'info':
        return <Info className="w-5 h-5 text-blue-400" />
    }
  }

  const getColorClasses = () => {
    switch (notification.type) {
      case 'success':
        return 'border-green-500/30 bg-green-500/10'
      case 'error':
        return 'border-red-500/30 bg-red-500/10'
      case 'warning':
        return 'border-yellow-500/30 bg-yellow-500/10'
      case 'info':
        return 'border-blue-500/30 bg-blue-500/10'
    }
  }

  return (
    <div
      className={`
        transform transition-all duration-300 ease-out
        ${isVisible && !isRemoving ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}
        ${getColorClasses()}
        backdrop-blur-md border rounded-lg p-4 shadow-xl
        max-w-sm w-full
      `}
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 mt-0.5">
          {getIcon()}
        </div>
        
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-semibold text-white mb-1">
            {notification.title}
          </h4>
          <p className="text-sm text-gray-300 leading-relaxed">
            {notification.message}
          </p>
          
          {notification.action && (
            <button
              onClick={notification.action.onClick}
              className="mt-2 text-xs font-medium text-blue-400 hover:text-blue-300 transition-colors"
            >
              {notification.action.label}
            </button>
          )}
        </div>
        
        <button
          onClick={handleRemove}
          className="flex-shrink-0 p-1 text-gray-400 hover:text-white transition-colors rounded"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
      
      {/* Progress bar for timed notifications */}
      {!notification.persistent && notification.duration && (
        <div className="mt-3 h-1 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full animate-shrink"
            style={{
              animation: `shrink ${notification.duration}ms linear forwards`
            }}
          />
        </div>
      )}
    </div>
  )
}

// Utility hooks for common notification types
export const useNotificationHelpers = () => {
  const { addNotification } = useNotifications()

  const notifySuccess = (title: string, message: string, options?: Partial<Notification>) => {
    addNotification({ type: 'success', title, message, ...options })
  }

  const notifyError = (title: string, message: string, options?: Partial<Notification>) => {
    addNotification({ type: 'error', title, message, persistent: true, ...options })
  }

  const notifyWarning = (title: string, message: string, options?: Partial<Notification>) => {
    addNotification({ type: 'warning', title, message, ...options })
  }

  const notifyInfo = (title: string, message: string, options?: Partial<Notification>) => {
    addNotification({ type: 'info', title, message, ...options })
  }

  return { notifySuccess, notifyError, notifyWarning, notifyInfo }
}

// CSS for the shrink animation
const style = document.createElement('style')
style.textContent = `
  @keyframes shrink {
    from { width: 100%; }
    to { width: 0%; }
  }
`
document.head.appendChild(style)

export default NotificationProvider
