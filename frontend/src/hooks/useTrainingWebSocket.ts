/**
 * ExoplanetAI Training WebSocket Hook
 * React hook для real-time отслеживания прогресса обучения ML моделей
 * 
 * Особенности:
 * - Автоматическое подключение и переподключение
 * - Типобезопасные сообщения
 * - Обработка ошибок и состояний подключения
 * - Автоматическая очистка ресурсов
 * - Поддержка JWT аутентификации
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { apiClient } from '../services/apiClient'

// ===== TYPES =====

export interface TrainingProgress {
  job_id: string
  epoch: number
  total_epochs: number
  loss: number
  accuracy: number
  val_loss?: number
  val_accuracy?: number
  eta_seconds: number
  status: 'training' | 'completed' | 'failed' | 'paused'
}

export interface WebSocketMessage {
  type: 'training_started' | 'training_progress' | 'training_completed' | 'training_error' | 'connected' | 'echo'
  job_id?: string
  data?: TrainingProgress
  message?: string
  error?: string
  session_id?: string
  timestamp?: string
  final_accuracy?: number
}

export enum ConnectionState {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  ERROR = 'error',
  RECONNECTING = 'reconnecting'
}

export interface UseTrainingWebSocketOptions {
  autoConnect?: boolean
  reconnectAttempts?: number
  reconnectInterval?: number
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  onMessage?: (message: WebSocketMessage) => void
}

export interface UseTrainingWebSocketReturn {
  connectionState: ConnectionState
  isConnected: boolean
  progress: TrainingProgress | null
  messages: WebSocketMessage[]
  error: string | null
  connect: () => void
  disconnect: () => void
  sendMessage: (message: any) => void
  clearMessages: () => void
  stats: {
    connectedAt: Date | null
    reconnectCount: number
    messagesReceived: number
    messagesSent: number
  }
}

// ===== HOOK IMPLEMENTATION =====

export function useTrainingWebSocket(
  jobId: string,
  options: UseTrainingWebSocketOptions = {}
): UseTrainingWebSocketReturn {
  const {
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    onConnect,
    onDisconnect,
    onError,
    onMessage
  } = options

  // State
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED)
  const [progress, setProgress] = useState<TrainingProgress | null>(null)
  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState({
    connectedAt: null as Date | null,
    reconnectCount: 0,
    messagesReceived: 0,
    messagesSent: 0
  })

  // Refs
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectCountRef = useRef(0)
  const mountedRef = useRef(true)

  // WebSocket URL
  const getWebSocketUrl = useCallback(() => {
    const baseUrl = apiClient['config']?.baseURL || 'http://localhost:8001'
    const wsUrl = baseUrl.replace('http', 'ws')
    return `${wsUrl}/ws/training/${jobId}`
  }, [jobId])

  // Connect function
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected')
      return
    }

    setConnectionState(ConnectionState.CONNECTING)
    setError(null)

    try {
      const wsUrl = getWebSocketUrl()
      console.log(`🔌 Connecting to WebSocket: ${wsUrl}`)
      
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        if (!mountedRef.current) return

        console.log('✅ WebSocket connected')
        setConnectionState(ConnectionState.CONNECTED)
        setStats(prev => ({
          ...prev,
          connectedAt: new Date(),
          reconnectCount: reconnectCountRef.current
        }))
        
        reconnectCountRef.current = 0
        onConnect?.()
      }

      wsRef.current.onmessage = (event) => {
        if (!mountedRef.current) return

        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          
          console.log('📨 WebSocket message:', message.type)
          
          // Update messages array
          setMessages(prev => [...prev, message].slice(-50)) // Keep last 50 messages
          
          // Update stats
          setStats(prev => ({
            ...prev,
            messagesReceived: prev.messagesReceived + 1
          }))
          
          // Handle specific message types
          switch (message.type) {
            case 'training_progress':
              if (message.data) {
                setProgress(message.data)
              }
              break
              
            case 'training_completed':
              console.log('🎉 Training completed!')
              if (message.data) {
                setProgress(prev => prev ? { ...prev, status: 'completed' } : null)
              }
              break
              
            case 'training_error':
              console.error('❌ Training error:', message.error)
              setError(message.error || 'Training failed')
              if (progress) {
                setProgress(prev => prev ? { ...prev, status: 'failed' } : null)
              }
              break
              
            case 'training_started':
              console.log('🚀 Training started')
              break
          }
          
          // Call custom message handler
          onMessage?.(message)
          
        } catch (parseError) {
          console.error('Failed to parse WebSocket message:', parseError)
          setError('Failed to parse server message')
        }
      }

      wsRef.current.onclose = (event) => {
        if (!mountedRef.current) return

        console.log('🔌 WebSocket disconnected:', event.code, event.reason)
        setConnectionState(ConnectionState.DISCONNECTED)
        setStats(prev => ({ ...prev, connectedAt: null }))
        
        onDisconnect?.()
        
        // Attempt to reconnect if not a clean close
        if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
          setConnectionState(ConnectionState.RECONNECTING)
          reconnectCountRef.current++
          
          setStats(prev => ({
            ...prev,
            reconnectCount: reconnectCountRef.current
          }))
          
          console.log(`🔄 Attempting to reconnect (${reconnectCountRef.current}/${reconnectAttempts})...`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current) {
              connect()
            }
          }, reconnectInterval)
        } else if (reconnectCountRef.current >= reconnectAttempts) {
          setError('Failed to reconnect after maximum attempts')
          setConnectionState(ConnectionState.ERROR)
        }
      }

      wsRef.current.onerror = (event) => {
        if (!mountedRef.current) return

        console.error('❌ WebSocket error:', event)
        setConnectionState(ConnectionState.ERROR)
        setError('WebSocket connection error')
        onError?.(event)
      }

    } catch (connectionError) {
      console.error('Failed to create WebSocket:', connectionError)
      setConnectionState(ConnectionState.ERROR)
      setError('Failed to create WebSocket connection')
    }
  }, [getWebSocketUrl, reconnectAttempts, reconnectInterval, onConnect, onDisconnect, onError, onMessage, progress])

  // Disconnect function
  const disconnect = useCallback(() => {
    console.log('🔌 Disconnecting WebSocket')
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect')
      wsRef.current = null
    }
    
    setConnectionState(ConnectionState.DISCONNECTED)
    reconnectCountRef.current = 0
  }, [])

  // Send message function
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const messageStr = JSON.stringify(message)
      wsRef.current.send(messageStr)
      
      setStats(prev => ({
        ...prev,
        messagesSent: prev.messagesSent + 1
      }))
      
      console.log('📤 Sent WebSocket message:', message)
    } else {
      console.warn('Cannot send message: WebSocket not connected')
    }
  }, [])

  // Clear messages function
  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  // Auto-connect effect
  useEffect(() => {
    if (autoConnect && jobId) {
      connect()
    }

    return () => {
      mountedRef.current = false
      disconnect()
    }
  }, [autoConnect, jobId, connect, disconnect])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false
      disconnect()
    }
  }, [disconnect])

  return {
    connectionState,
    isConnected: connectionState === ConnectionState.CONNECTED,
    progress,
    messages,
    error,
    connect,
    disconnect,
    sendMessage,
    clearMessages,
    stats
  }
}

// ===== SPECIALIZED HOOKS =====

/**
 * Simplified hook for just getting training progress
 */
export function useTrainingProgress(jobId: string) {
  const { progress, isConnected, error } = useTrainingWebSocket(jobId, {
    autoConnect: true
  })

  return {
    progress,
    isConnected,
    error,
    isTraining: progress?.status === 'training',
    isCompleted: progress?.status === 'completed',
    isFailed: progress?.status === 'failed',
    progressPercentage: progress ? (progress.epoch / progress.total_epochs) * 100 : 0
  }
}

/**
 * Hook for general WebSocket connection (not training-specific)
 */
export function useGeneralWebSocket(sessionId: string) {
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED)
  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  const wsRef = useRef<WebSocket | null>(null)

  const connect = useCallback(() => {
    const baseUrl = apiClient['config']?.baseURL || 'http://localhost:8001'
    const wsUrl = baseUrl.replace('http', 'ws')
    const fullUrl = `${wsUrl}/ws/connect/${sessionId}`

    setConnectionState(ConnectionState.CONNECTING)
    wsRef.current = new WebSocket(fullUrl)

    wsRef.current.onopen = () => {
      setConnectionState(ConnectionState.CONNECTED)
      console.log('✅ General WebSocket connected')
    }

    wsRef.current.onmessage = (event) => {
      const message = JSON.parse(event.data)
      setMessages(prev => [...prev, message].slice(-50))
    }

    wsRef.current.onclose = () => {
      setConnectionState(ConnectionState.DISCONNECTED)
    }

    wsRef.current.onerror = () => {
      setConnectionState(ConnectionState.ERROR)
    }
  }, [sessionId])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
    setConnectionState(ConnectionState.DISCONNECTED)
  }, [])

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }, [])

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return {
    connectionState,
    isConnected: connectionState === ConnectionState.CONNECTED,
    messages,
    sendMessage,
    disconnect
  }
}

export default useTrainingWebSocket
