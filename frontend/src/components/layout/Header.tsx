import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  Telescope, Menu, X, Home, Brain, Zap, Database, Globe, Info,
  Activity, Cpu, Wifi, WifiOff, AlertTriangle, CheckCircle, Clock,
  TrendingUp, Users, Search
} from 'lucide-react'
import { useTranslation } from 'react-i18next'
import BackgroundToggle from '../ui/ThemeToggle'
import LanguageToggle from '../ui/LanguageToggle'
import type { HealthStatus } from '../../types/api'

interface HeaderProps {
  healthStatus?: HealthStatus | null
  useSimpleBackground?: boolean
  setUseSimpleBackground?: (value: boolean | ((prev: boolean) => boolean)) => void
}

interface SystemStats {
  activeUsers: number
  totalSearches: number
  cppStatus: 'loaded' | 'fallback' | 'unknown'
  uptime: string
}

const Header: React.FC<HeaderProps> = React.memo(({ healthStatus, useSimpleBackground, setUseSimpleBackground }) => {
  const { t } = useTranslation()
  const location = useLocation()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null)
  const [showSystemPanel, setShowSystemPanel] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'online' | 'offline' | 'checking'>('checking')

  // Memoized navigation items to prevent recreation on every render
  const navigationItems = useMemo(() => [
    { path: '/', label: t('navigation.home'), icon: Home },
    { path: '/ai-training', label: t('navigation.aiTraining'), icon: Brain },
    { path: '/gpi', label: t('navigation.gpi'), icon: Zap },
    { path: '/search', label: 'Search', icon: Search },
    { path: '/catalog', label: t('navigation.catalog'), icon: Globe },
    { path: '/database', label: t('navigation.database'), icon: Database },
    { path: '/about', label: t('navigation.about'), icon: Info },
  ], [t])

  // Memoized isActivePath function
  const isActivePath = useCallback((path: string) => {
    return location.pathname === path
  }, [location.pathname])

  // Fetch system stats with useCallback to prevent recreation
  const fetchSystemStats = useCallback(async () => {
    try {
      // Check if health status indicates healthy connection
      if (healthStatus && healthStatus.status === 'healthy') {
        setSystemStats({
          activeUsers: 1, // Default value
          totalSearches: 0, // Default value  
          cppStatus: 'loaded', // Default value
          uptime: '00:00:00' // Default value
        })
        setConnectionStatus('online')
      } else {
        // If no health status or unhealthy, set offline
        setConnectionStatus('offline')
        setSystemStats(null)
      }
    } catch (error) {
      console.error('Failed to fetch system stats:', error)
      setConnectionStatus('offline')
      setSystemStats(null)
    }
  }, [healthStatus])
  // Effect for fetching system stats
  useEffect(() => {
    // Initial check - if no healthStatus, set to offline immediately
    if (!healthStatus) {
      setConnectionStatus('offline')
      setSystemStats(null)
    } else {
      fetchSystemStats()
    }
    
    const interval = setInterval(fetchSystemStats, 60000) // Update every 60s instead of 30s
    return () => clearInterval(interval)
  }, [fetchSystemStats, healthStatus])

  // Memoized handler for mobile menu toggle
  const toggleMobileMenu = useCallback(() => {
    setMobileMenuOpen(prev => !prev)
  }, [])

  // Memoized handler for system panel toggle
  const toggleSystemPanel = useCallback(() => {
    setShowSystemPanel(prev => !prev)
  }, [])

  return (
    <header className="relative z-20 w-full">
      {/* Top Navigation Bar */}
      <nav className="flex items-center justify-between p-4 bg-white/5 dark:bg-gray-900/50 backdrop-blur-md border-b border-white/10 dark:border-gray-700/50">
        {/* Logo and Title */}
        <Link to="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <div className="flex items-center gap-2">
            <Telescope size={32} className="text-amber-400 animate-pulse-slow drop-shadow-lg" />
            <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-100 via-amber-200 to-orange-300 bg-clip-text text-transparent">
              AstroManas
            </h1>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden lg:flex items-center gap-1">
          {navigationItems.map((item) => {
            const Icon = item.icon
            const isActive = isActivePath(item.path)
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                  isActive
                    ? 'bg-gray-600/20 text-gray-300 border border-gray-500/30'
                    : 'text-gray-300 hover:text-white hover:bg-white/10'
                }`}
              >
                <Icon size={16} />
                {item.label}
              </Link>
            )
          })}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          {/* System Stats Panel Toggle */}
          <button
            onClick={() => setShowSystemPanel(!showSystemPanel)}
            className="hidden lg:flex items-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 rounded-lg transition-all duration-300 text-gray-300 hover:text-white"
          >
            <Activity size={16} className="animate-pulse" />
            <span className="text-xs font-medium">System</span>
          </button>

          {/* Connection Status */}
          <div className={`hidden md:flex items-center gap-2 px-3 py-1 rounded-full text-sm transition-all duration-300 ${
            connectionStatus === 'online' ? 
              (useSimpleBackground ? 'bg-green-500/15 text-green-300' : 'bg-green-500/20 text-green-300 shadow-green-glow') :
            connectionStatus === 'offline' ? 
              (useSimpleBackground ? 'bg-red-500/15 text-red-300' : 'bg-red-500/20 text-red-300 shadow-red-glow') :
              (useSimpleBackground ? 'bg-yellow-500/15 text-yellow-300' : 'bg-yellow-500/20 text-yellow-300 shadow-amber-glow')
          }`}>
            {connectionStatus === 'online' ? <Wifi size={12} /> : 
             connectionStatus === 'offline' ? <WifiOff size={12} /> : 
             <Clock size={12} className="animate-spin" />}
            <span className="text-xs font-medium">
              {connectionStatus === 'online' ? 'Connected' :
               connectionStatus === 'offline' ? 'Offline' : 'Connecting...'}
            </span>
          </div>

          {/* C++ Status Indicator */}
          {systemStats && (
            <div className={`hidden lg:flex items-center gap-2 px-2 py-1 rounded-md text-xs ${
              systemStats.cppStatus === 'loaded' ? 'bg-gray-600/20 text-gray-300' :
              systemStats.cppStatus === 'fallback' ? 'bg-orange-500/20 text-orange-300' :
              'bg-gray-500/20 text-gray-400'
            }`}>
              <Cpu size={12} />
              <span className="font-medium">
                {systemStats.cppStatus === 'loaded' ? 'C++ ⚡' :
                 systemStats.cppStatus === 'fallback' ? 'Python' : 'Unknown'}
              </span>
            </div>
          )}

          {/* Quick Stats */}
          {systemStats && (
            <div className="hidden xl:flex items-center gap-4 px-3 py-2 bg-white/5 rounded-lg">
              <div className="flex items-center gap-1 text-xs text-gray-400">
                <Users size={12} />
                <span>{systemStats.activeUsers}</span>
              </div>
              <div className="flex items-center gap-1 text-xs text-gray-400">
                <Search size={12} />
                <span>{systemStats.totalSearches}</span>
              </div>
              <div className="flex items-center gap-1 text-xs text-gray-400">
                <Clock size={12} />
                <span>{systemStats.uptime}</span>
              </div>
            </div>
          )}

          {/* Language Toggle */}
          <LanguageToggle />

          {/* Background Toggle */}
          <BackgroundToggle 
            useSimpleBackground={useSimpleBackground}
            setUseSimpleBackground={setUseSimpleBackground}
          />

          {/* Mobile Menu Button */}
          <button
            onClick={toggleMobileMenu}
            className="lg:hidden p-2 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
            aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </nav>

      {/* Mobile Navigation Menu */}
      {mobileMenuOpen && (
        <div className="lg:hidden absolute top-full left-0 right-0 bg-gray-900/95 backdrop-blur-md border-b border-white/10 z-50">
          <div className="p-4 space-y-2">
            {navigationItems.map((item) => {
              const Icon = item.icon
              const isActive = isActivePath(item.path)
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={toggleMobileMenu}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-300 ${
                    isActive
                      ? 'bg-gray-600/20 text-gray-300 border border-gray-500/30'
                      : 'text-gray-300 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <Icon size={18} />
                  {item.label}
                </Link>
              )
            })}
            
            {/* Mobile System Status */}
            <div className="mt-4 space-y-2">
              {healthStatus && (
                <div className={`flex items-center gap-2 px-4 py-3 rounded-lg text-sm ${
                  healthStatus.status === 'healthy' ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
                }`}>
                  {healthStatus.status === 'healthy' ? 
                    <CheckCircle size={14} /> : 
                    <AlertTriangle size={14} />
                  }
                  <span>
                    API: {healthStatus.status} • Uptime: {Math.floor((healthStatus.uptime || 0) / 3600)}h
                  </span>
                </div>
              )}
              
              {systemStats && (
                <div className="flex items-center gap-2 px-4 py-3 bg-gray-600/20 rounded-lg text-gray-300 text-sm">
                  <Cpu size={14} />
                  <span>
                    C++: {systemStats.cppStatus === 'loaded' ? '⚡ Active' : '🐍 Fallback'} • 
                    {systemStats.totalSearches} searches
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Page Title Section - Only show on non-home pages */}
      {location.pathname !== '/' && (
        <div className="text-center py-6 px-4 border-b border-white/10">
          <div className="max-w-4xl mx-auto">
            {navigationItems.map((item) => {
              if (isActivePath(item.path)) {
                const Icon = item.icon
                return (
                  <div key={item.path} className="flex items-center justify-center gap-3 mb-2">
                    <Icon className="w-8 h-8 text-amber-400 drop-shadow-glow" />
                    <h2 className="text-3xl font-bold text-white">
                      {item.label}
                    </h2>
                  </div>
                )
              }
              return null
            })}
          </div>
        </div>
      )}

      
            {/* System Stats Panel */}
            {showSystemPanel && (
              <div className="absolute top-full right-4 mt-2 w-80 bg-gray-900/95 backdrop-blur-md border border-white/10 rounded-lg shadow-xl z-50">
                <div className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                      <Activity size={18} className="text-amber-400 drop-shadow-sm" />
                      System Status
                    </h3>
                    <button
                      onClick={toggleSystemPanel}
                      className="p-1 text-gray-400 hover:text-white rounded"
                      aria-label="Close system panel"
                    >
                      <X size={16} />
                    </button>
                  </div>
                  
                  {/* API Health */}
                  {healthStatus && (
                    <div className="mb-4 p-3 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-300">API Status</span>
                        <div className={`flex items-center gap-1 text-xs ${
                          healthStatus.status === 'healthy' ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {healthStatus.status === 'healthy' ?
                            <CheckCircle size={12} /> :
                            <AlertTriangle size={12} />
                          }
                          {healthStatus.status}
                        </div>
                      </div>
                      <div className="text-xs text-gray-400">
                        Version: {healthStatus.version?.toString().replace(/[<>'"&]/g, (match: string) => {
                          const escapeMap: Record<string, string> = {
                            '<': '<',
                            '>': '>',
                            '"': '"',
                            "'": '&#x27;',
                            '&': '&'
                          };
                          return escapeMap[match] || match;
                        })}
                      </div>
                      {healthStatus.services && (
                        <div className="mt-2 space-y-1">
                          {Object.entries(healthStatus.services).map(([service, status]) => (
                            <div key={service} className="flex justify-between text-xs">
                              <span className="text-gray-400">{service}:</span>
                              <span className={status === 'healthy' ? 'text-green-400' : 'text-yellow-400'}>
                                {String(status)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* System Metrics */}
                  {systemStats && (
                    <div className="space-y-3">
                      <div className="p-3 bg-white/5 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-gray-300">Performance</span>
                          <TrendingUp size={14} className="text-amber-400 drop-shadow-sm" />
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-gray-400">Active Users:</span>
                            <div className="text-white font-medium">{systemStats.activeUsers}</div>
                          </div>
                          <div>
                            <span className="text-gray-400">Total Searches:</span>
                            <div className="text-white font-medium">{systemStats.totalSearches}</div>
                          </div>
                          <div>
                            <span className="text-gray-400">Uptime:</span>
                            <div className="text-white font-medium">{systemStats.uptime}</div>
                          </div>
                          <div>
                            <span className="text-gray-400">C++ Status:</span>
                            <div className={`font-medium ${
                              systemStats.cppStatus === 'loaded' ? 'text-gray-300' :
                              systemStats.cppStatus === 'fallback' ? 'text-orange-400' :
                              'text-gray-400'
                            }`}>
                              {systemStats.cppStatus === 'loaded' ? '⚡ Loaded' :
                               systemStats.cppStatus === 'fallback' ? '🐍 Fallback' : 'Unknown'}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </header>
        )
      }, (prevProps, nextProps) => {
        // Custom comparison function for React.memo
        return prevProps.healthStatus?.status === nextProps.healthStatus?.status &&
               prevProps.healthStatus?.version === nextProps.healthStatus?.version
      })
      
      export default Header
