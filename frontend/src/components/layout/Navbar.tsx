import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Menu, 
  X, 
  Radar, 
  Search, 
  BarChart3, 
  FileText, 
  Info,
  Home
} from 'lucide-react'
import { useAppStore } from '../../store/useAppStore'

const navigation = [
  { name: 'Главная', href: '/', icon: Home },
  { name: 'Поиск', href: '/search', icon: Search },
  { name: 'Анализ', href: '/analysis', icon: BarChart3 },
  { name: 'Результаты', href: '/results', icon: FileText },
  { name: 'О проекте', href: '/about', icon: Info },
]

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const location = useLocation()
  const { healthStatus } = useAppStore()

  const isActive = (path: string) => {
    if (path === '/') {
      return location.pathname === '/'
    }
    return location.pathname.startsWith(path)
  }

  return (
    <nav className="bg-space-900/80 backdrop-blur-md border-b border-space-700 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo and brand */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2 group">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                className="p-2 bg-cosmic-gradient rounded-lg"
              >
                <Radar className="h-6 w-6 text-white" />
              </motion.div>
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold text-gradient">
                  Exoplanet AI
                </h1>
                <p className="text-xs text-gray-400 -mt-1">
                  Advanced Transit Detection
                </p>
              </div>
            </Link>
          </div>

          {/* Desktop navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => {
              const Icon = item.icon
              const active = isActive(item.href)
              
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`
                    relative px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200
                    flex items-center space-x-2 group
                    ${active 
                      ? 'text-white bg-primary-600/20 border border-primary-500/30' 
                      : 'text-gray-300 hover:text-white hover:bg-space-700/50'
                    }
                  `}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.name}</span>
                  
                  {active && (
                    <motion.div
                      layoutId="navbar-indicator"
                      className="absolute inset-0 bg-primary-600/10 rounded-lg border border-primary-500/30"
                      initial={false}
                      transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    />
                  )}
                </Link>
              )
            })}
          </div>

          {/* Status indicator and mobile menu button */}
          <div className="flex items-center space-x-4">
            {/* Health status indicator */}
            {healthStatus && (
              <div className="hidden sm:flex items-center space-x-2">
                <div className={`
                  w-2 h-2 rounded-full
                  ${healthStatus.status === 'healthy' ? 'bg-green-500' : 
                    healthStatus.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'}
                `} />
                <span className="text-xs text-gray-400">
                  {healthStatus.status === 'healthy' ? 'Онлайн' : 
                   healthStatus.status === 'degraded' ? 'Ограничено' : 'Офлайн'}
                </span>
              </div>
            )}

            {/* Mobile menu button */}
            <button
              type="button"
              className="md:hidden p-2 rounded-lg text-gray-400 hover:text-white hover:bg-space-700 transition-colors"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <motion.div
        initial={false}
        animate={{ height: mobileMenuOpen ? 'auto' : 0 }}
        className="md:hidden overflow-hidden bg-space-800/95 backdrop-blur-md border-t border-space-700"
      >
        <div className="px-4 py-2 space-y-1">
          {navigation.map((item) => {
            const Icon = item.icon
            const active = isActive(item.href)
            
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`
                  block px-3 py-2 rounded-lg text-sm font-medium transition-colors
                  flex items-center space-x-3
                  ${active 
                    ? 'text-white bg-primary-600/20 border-l-2 border-primary-500' 
                    : 'text-gray-300 hover:text-white hover:bg-space-700/50'
                  }
                `}
                onClick={() => setMobileMenuOpen(false)}
              >
                <Icon className="h-4 w-4" />
                <span>{item.name}</span>
              </Link>
            )
          })}
          
          {/* Mobile health status */}
          {healthStatus && (
            <div className="px-3 py-2 border-t border-space-600 mt-2">
              <div className="flex items-center space-x-2 text-xs text-gray-400">
                <div className={`
                  w-2 h-2 rounded-full
                  ${healthStatus.status === 'healthy' ? 'bg-green-500' : 
                    healthStatus.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'}
                `} />
                <span>
                  Статус: {healthStatus.status === 'healthy' ? 'Онлайн' : 
                           healthStatus.status === 'degraded' ? 'Ограничено' : 'Офлайн'}
                </span>
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </nav>
  )
}
