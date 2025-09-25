import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Search, 
  BarChart3, 
  Brain, 
  Database,
  Rocket,
  Globe,
  Star,
  Zap
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import ApiService from '../services/api'
import { useAppStore } from '../store/useAppStore'

const stats = [
  { label: 'Stars Analyzed', value: '50,000+', icon: Star },
  { label: 'Candidates Found', value: '1,200+', icon: Globe },
  { label: 'AI Accuracy', value: '94.5%', icon: Brain },
  { label: 'Analysis Time', value: '<30s', icon: Zap },
]

const features = [
  {
    icon: Brain,
    title: 'AI Analysis',
    description: 'Advanced neural networks for precise exoplanet detection'
  },
  {
    icon: BarChart3,
    title: 'BLS Algorithm',
    description: 'Box Least Squares transit detection with optimization'
  },
  {
    icon: Database,
    title: 'Mission Data',
    description: 'Support for TESS, Kepler, and K2 space missions'
  }
]

export default function HomePage() {
  const { setHealthStatus } = useAppStore()

  // Health check query
  const { data: healthStatus } = useQuery({
    queryKey: ['health'],
    queryFn: ApiService.getHealth,
    refetchInterval: 30000,
    onSuccess: (data) => {
      setHealthStatus(data)
    },
    onError: () => {
      setHealthStatus({
        status: 'down',
        timestamp: new Date().toISOString(),
        services_available: false,
        scientific_libs: false,
        services: {
          bls: 'unavailable',
          data: 'unavailable',
          ai: 'unavailable'
        }
      })
    }
  })

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="py-32 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            {/* NASA Logo */}
            <div className="mb-16">
              <div className="w-24 h-24 mx-auto mb-8 bg-gradient-to-br from-blue-600 to-blue-800 rounded-full flex items-center justify-center shadow-2xl">
                <Rocket className="h-12 w-12 text-white" />
              </div>
              
              <h1 className="text-6xl md:text-7xl font-light text-white mb-6 tracking-tight">
                Exoplanet AI
              </h1>
              
              <p className="text-xl text-gray-400 mb-12 max-w-2xl mx-auto leading-relaxed">
                NASA-grade artificial intelligence for discovering new worlds beyond our solar system
              </p>
            </div>
            
            {/* Status Indicators */}
            <div className="flex items-center justify-center space-x-8 mb-16">
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-gray-300">System Operational</span>
              </div>
              <div className="w-px h-6 bg-gray-600"></div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-gray-300">AI Models Ready</span>
              </div>
            </div>

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="flex flex-col sm:flex-row gap-4 justify-center mb-20"
            >
              <Link
                to="/search"
                className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-3 shadow-lg hover:shadow-xl"
              >
                <Search className="h-5 w-5" />
                <span>Start Discovery</span>
              </Link>
              
              <Link
                to="/catalog"
                className="bg-gray-800 hover:bg-gray-700 text-white px-8 py-4 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-3 border border-gray-600 hover:border-gray-500"
              >
                <Database className="h-5 w-5" />
                <span>Browse Catalog</span>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Statistics */}
      <section className="py-20 px-4 border-t border-gray-800">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl font-light text-white mb-4">Mission Statistics</h2>
            <p className="text-gray-400">Real-time data from our exoplanet discovery missions</p>
          </motion.div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => {
              const Icon = stat.icon
              return (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="text-center p-6 bg-gray-900/50 rounded-lg border border-gray-800 hover:border-gray-700 transition-colors"
                >
                  <Icon className="h-8 w-8 text-blue-500 mx-auto mb-4" />
                  <div className="text-2xl font-bold text-white mb-2">
                    {stat.value}
                  </div>
                  <div className="text-sm text-gray-400">
                    {stat.label}
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl font-light text-white mb-4">Advanced Capabilities</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Cutting-edge technology for precise exoplanet detection and analysis
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="p-8 bg-gray-900/50 rounded-lg border border-gray-800 hover:border-gray-700 transition-colors"
                >
                  <Icon className="h-10 w-10 text-blue-500 mb-6" />
                  <h3 className="text-xl font-semibold text-white mb-4">
                    {feature.title}
                  </h3>
                  <p className="text-gray-400 leading-relaxed">
                    {feature.description}
                  </p>
                </motion.div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-20 px-4 border-t border-gray-800">
        <div className="max-w-3xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="p-12 bg-gradient-to-br from-blue-900/20 to-gray-900/20 rounded-2xl border border-gray-800"
          >
            <Globe className="h-16 w-16 text-blue-500 mx-auto mb-8" />
            
            <h2 className="text-3xl font-light text-white mb-6">
              Ready to Discover New Worlds?
            </h2>
            <p className="text-gray-400 mb-10 text-lg leading-relaxed">
              Join the search for exoplanets and help expand our understanding of the universe. 
              Every discovery brings us closer to finding potentially habitable worlds.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/search"
                className="bg-green-600 hover:bg-green-700 text-white px-8 py-4 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-3 shadow-lg"
              >
                <Rocket className="h-5 w-5" />
                <span>Launch Search</span>
              </Link>
              
              <Link
                to="/bls"
                className="bg-purple-600 hover:bg-purple-700 text-white px-8 py-4 rounded-lg font-medium transition-all duration-200 flex items-center justify-center space-x-3 shadow-lg"
              >
                <BarChart3 className="h-5 w-5" />
                <span>BLS Analysis</span>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
