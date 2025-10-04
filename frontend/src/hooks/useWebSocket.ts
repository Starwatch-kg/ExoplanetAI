import { useEffect, useRef, useState, useCallback } from 'react'
import { TokenManager } from '../services/api-enhanced'

// WebSocket message types
export interface WebSocketMessage {
  type: string
  data?: any
  timestamp: number
  session_id?: string
  job_id?: string
  target_id?: string
}

export interface TrainingProgressMessage extends WebSocketMessage {
  type: 'training_progress'
  job_id: string
  data: {
    progress: number
    current_epoch: number
    total_epochs: number
    metrics: {
      loss: number
      accuracy: number
      val_loss?: number
      val_accuracy?: number
    }
    estimated_time_remaining: number
    status: 'running' | 'completed' | 'failed'
  }
}

export interface AnalysisUpdateMessage extends WebSocketMessage {
  type: 'analysis_update'
  target_id: string
  data: {
    status: 'processing' | 'completed' | 'failed'
    progress: number
    current_step: string
    results?: any
  }
}

export interface SystemNotificationMessage extends WebSocketMessage {
  type: 'system_notification'
  data: {
    title: string
    message: string
    level: 'info' | 'warning' | 'error' | 'success'
    action?: {
      label: string
      url: string
    }
  }
}

// WebSocket connection states
export enum WebSocketState {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  ERROR = 'error'
}

// Hook options
export interface UseWebSocketOptions {
  url?: string
  autoConnect?: boolean
  reconnectAttempts?: number
  reconnectInterval?: number
  heartbeatInterval?: number
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
}

// Hook return type
export interface UseWebSocketReturn {
  state: WebSocketState
  isConnected: boolean
  connect: () => void
  disconnect: () => void
  sendMessage: (message: any) => void
  subscribe: (topics: string[]) => void
  unsubscribe: (topics: string[]) => void
  lastMessage: WebSocketMessage | null
  connectionStats: {
    connectedAt: Date | null
    reconnectCount: number
    messagesReceived: number
    messagesSent: number
  }
}

// Default WebSocket URL
const DEFAULT_WS_URL = import.meta.env.VITE_API_URL?.replace('http', 'ws') || 'ws://localhost:8001'

export function useWebSocket(
  endpoint: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const {
    url = DEFAULT_WS_URL,
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    heartbeatInterval = 30000,
    onMessage,
    onConnect,
    onDisconnect,
    onError
  } = options

  // State
  const [state, setState] = useState<WebSocketState>(WebSocketState.DISCONNECTED)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [connectionStats, setConnectionStats] = useState({
    connectedAt: null as Date | null,
    reconnectCount: 0,
    messagesReceived: 0,
    messagesSent: 0
  })

  // Refs
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectCountRef = useRef(0)
  const subscriptionsRef = useRef<string[]>([])

  // Generate session ID
  const sessionId = useRef(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`)

  // Build WebSocket URL with authentication
  const buildWebSocketUrl = useCallback(() => {
    const token = TokenManager.getToken()
    const wsUrl = `${url}${endpoint}/${sessionId.current}`
    return token ? `${wsUrl}?token=${token}` : wsUrl
  }, [url, endpoint])

  // Send heartbeat
  const sendHeartbeat = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'ping',
        timestamp: Date.now()
      }))
    }
  }, [])

  // Start heartbeat
  const startHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearInterval(heartbeatTimeoutRef.current)
    }
    
    heartbeatTimeoutRef.current = setInterval(sendHeartbeat, heartbeatInterval)
  }, [sendHeartbeat, heartbeatInterval])

  // Stop heartbeat
  const stopHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearInterval(heartbeatTimeoutRef.current)
      heartbeatTimeoutRef.current = null
    }
  }, [])

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return // Already connected
    }

    setState(WebSocketState.CONNECTING)
    
    try {
      const wsUrl = buildWebSocketUrl()
      console.log(`Connecting to WebSocket: ${wsUrl}`)
      
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        console.log('WebSocket connected')
        setState(WebSocketState.CONNECTED)
        setConnectionStats(prev => ({
          ...prev,
          connectedAt: new Date(),
          reconnectCount: reconnectCountRef.current
        }))
        
        reconnectCountRef.current = 0
        startHeartbeat()
        
        // Resubscribe to topics
        if (subscriptionsRef.current.length > 0) {
          wsRef.current?.send(JSON.stringify({
            type: 'subscribe',
            topics: subscriptionsRef.current
          }))
        }
        
        onConnect?.()
      }

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          
          setLastMessage(message)
          setConnectionStats(prev => ({
            ...prev,
            messagesReceived: prev.messagesReceived + 1
          }))
          
          // Handle pong messages
          if (message.type === 'pong') {
            console.log('Received pong from server')
            return
          }
          
          onMessage?.(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        setState(WebSocketState.DISCONNECTED)
        stopHeartbeat()
        
        setConnectionStats(prev => ({
          ...prev,
          connectedAt: null
        }))
        
        onDisconnect?.()
        
        // Attempt to reconnect if not a clean close
        if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++
          setConnectionStats(prev => ({
            ...prev,
            reconnectCount: reconnectCountRef.current
          }))
          
          console.log(`Attempting to reconnect (${reconnectCountRef.current}/${reconnectAttempts})...`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        }
      }

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setState(WebSocketState.ERROR)
        onError?.(error)
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      setState(WebSocketState.ERROR)
    }
  }, [
    buildWebSocketUrl,
    reconnectAttempts,
    reconnectInterval,
    startHeartbeat,
    stopHeartbeat,
    onConnect,
    onDisconnect,
    onError,
    onMessage
  ])

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    
    stopHeartbeat()
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect')
      wsRef.current = null
    }
    
    setState(WebSocketState.DISCONNECTED)
    reconnectCountRef.current = 0
  }, [stopHeartbeat])

  // Send message
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const messageStr = JSON.stringify(message)
      wsRef.current.send(messageStr)
      
      setConnectionStats(prev => ({
        ...prev,
        messagesSent: prev.messagesSent + 1
      }))
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message)
    }
  }, [])

  // Subscribe to topics
  const subscribe = useCallback((topics: string[]) => {
    subscriptionsRef.current = [...new Set([...subscriptionsRef.current, ...topics])]
    
    sendMessage({
      type: 'subscribe',
      topics: topics
    })
  }, [sendMessage])

  // Unsubscribe from topics
  const unsubscribe = useCallback((topics: string[]) => {
    subscriptionsRef.current = subscriptionsRef.current.filter(topic => !topics.includes(topic))
    
    sendMessage({
      type: 'unsubscribe',
      topics: topics
    })
  }, [sendMessage])

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [autoConnect, connect, disconnect])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    state,
    isConnected: state === WebSocketState.CONNECTED,
    connect,
    disconnect,
    sendMessage,
    subscribe,
    unsubscribe,
    lastMessage,
    connectionStats
  }
}

// Specialized hooks for different use cases
export function useTrainingProgress(jobId: string, options: Omit<UseWebSocketOptions, 'onMessage'> = {}) {
  const [progress, setProgress] = useState<TrainingProgressMessage['data'] | null>(null)
  
  const { isConnected, ...rest } = useWebSocket(`/ws/training`, {
    ...options,
    onMessage: (message) => {
      if (message.type === 'training_progress' && message.job_id === jobId) {
        setProgress((message as TrainingProgressMessage).data)
      }
    }
  })
  
  return {
    progress,
    isConnected,
    ...rest
  }
}

export function useAnalysisProgress(targetId: string, options: Omit<UseWebSocketOptions, 'onMessage'> = {}) {
  const [analysis, setAnalysis] = useState<AnalysisUpdateMessage['data'] | null>(null)
  
  const { isConnected, ...rest } = useWebSocket(`/ws/analysis`, {
    ...options,
    onMessage: (message) => {
      if (message.type === 'analysis_update' && message.target_id === targetId) {
        setAnalysis((message as AnalysisUpdateMessage).data)
      }
    }
  })
  
  return {
    analysis,
    isConnected,
    ...rest
  }
}

export function useSystemNotifications(options: Omit<UseWebSocketOptions, 'onMessage'> = {}) {
  const [notifications, setNotifications] = useState<SystemNotificationMessage['data'][]>([])
  
  const { isConnected, subscribe, ...rest } = useWebSocket(`/ws/connect`, {
    ...options,
    onMessage: (message) => {
      if (message.type === 'system_notification') {
        const notification = (message as SystemNotificationMessage).data
        setNotifications(prev => [notification, ...prev].slice(0, 50)) // Keep last 50 notifications
      }
    }
  })
  
  // Auto-subscribe to system notifications
  useEffect(() => {
    if (isConnected) {
      subscribe(['system_notifications'])
    }
  }, [isConnected, subscribe])
  
  const clearNotifications = useCallback(() => {
    setNotifications([])
  }, [])
  
  const removeNotification = useCallback((index: number) => {
    setNotifications(prev => prev.filter((_, i) => i !== index))
  }, [])
  
  return {
    notifications,
    clearNotifications,
    removeNotification,
    isConnected,
    subscribe,
    ...rest
  }
}
