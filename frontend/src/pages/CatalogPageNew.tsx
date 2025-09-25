import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Database, 
  Search, 
  Filter,
  Star,
  Globe,
  Eye,
  ChevronRight
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import ApiService from '../services/api'
import LoadingSpinner from '../components/ui/LoadingSpinner'

interface CatalogFilter {
  search: string
  catalog: 'all' | 'TIC' | 'KIC' | 'EPIC'
  minMagnitude: number
  maxMagnitude: number
}

const mockStars = [
  { id: 'TIC 123456789', name: 'HD 209458', magnitude: 7.65, temperature: 6065, distance: 159, catalog: 'TIC' },
  { id: 'KIC 11804465', name: 'Kepler-442', magnitude: 14.76, temperature: 4402, distance: 1206, catalog: 'KIC' },
  { id: 'EPIC 201367065', name: 'K2-18', magnitude: 13.04, temperature: 3457, distance: 124, catalog: 'EPIC' },
  { id: 'TIC 307210830', name: 'TOI-715', magnitude: 16.25, temperature: 3450, distance: 137, catalog: 'TIC' },
  { id: 'KIC 8462852', name: 'Tabby\'s Star', magnitude: 11.7, temperature: 6750, distance: 1470, catalog: 'KIC' },
]

export default function CatalogPage() {
  const [filters, setFilters] = useState<CatalogFilter>({
    search: '',
    catalog: 'all',
    minMagnitude: 0,
    maxMagnitude: 20
  })

  const [filteredStars, setFilteredStars] = useState(mockStars)

  useEffect(() => {
    let filtered = mockStars.filter(star => {
      const matchesSearch = star.name.toLowerCase().includes(filters.search.toLowerCase()) ||
                           star.id.toLowerCase().includes(filters.search.toLowerCase())
      const matchesCatalog = filters.catalog === 'all' || star.catalog === filters.catalog
      const matchesMagnitude = star.magnitude >= filters.minMagnitude && star.magnitude <= filters.maxMagnitude
      
      return matchesSearch && matchesCatalog && matchesMagnitude
    })
    
    setFilteredStars(filtered)
  }, [filters])

  const { data: catalogs, isLoading: catalogsLoading } = useQuery({
    queryKey: ['catalogs'],
    queryFn: ApiService.getCatalogs,
    staleTime: 5 * 60 * 1000
  })

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-br from-purple-600 to-purple-800 rounded-full flex items-center justify-center">
            <Database className="h-8 w-8 text-white" />
          </div>
          <h1 className="text-4xl font-light text-white mb-4">
            Star Catalog
          </h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Browse and search through our comprehensive database of stars from TESS, Kepler, and K2 missions
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-4 gap-8">
          {/* Filters Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="lg:col-span-1"
          >
            <div className="bg-gray-900/50 rounded-xl border border-gray-800 p-6 sticky top-8">
              <h2 className="text-lg font-semibold text-white mb-6 flex items-center">
                <Filter className="h-5 w-5 mr-2" />
                Filters
              </h2>
              
              <div className="space-y-6">
                {/* Search */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Search
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Star name or ID..."
                      value={filters.search}
                      onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
                      className="w-full px-3 py-2 pl-10 bg-black/50 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
                    />
                    <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-500" />
                  </div>
                </div>

                {/* Catalog Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Catalog
                  </label>
                  <select
                    value={filters.catalog}
                    onChange={(e) => setFilters(prev => ({ ...prev, catalog: e.target.value as any }))}
                    className="w-full px-3 py-2 bg-black/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
                  >
                    <option value="all">All Catalogs</option>
                    <option value="TIC">TIC (TESS)</option>
                    <option value="KIC">KIC (Kepler)</option>
                    <option value="EPIC">EPIC (K2)</option>
                  </select>
                </div>

                {/* Magnitude Range */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Magnitude Range
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="number"
                      placeholder="Min"
                      value={filters.minMagnitude}
                      onChange={(e) => setFilters(prev => ({ ...prev, minMagnitude: Number(e.target.value) }))}
                      className="px-3 py-2 bg-black/50 border border-gray-700 rounded text-white text-sm focus:outline-none focus:border-blue-500"
                    />
                    <input
                      type="number"
                      placeholder="Max"
                      value={filters.maxMagnitude}
                      onChange={(e) => setFilters(prev => ({ ...prev, maxMagnitude: Number(e.target.value) }))}
                      className="px-3 py-2 bg-black/50 border border-gray-700 rounded text-white text-sm focus:outline-none focus:border-blue-500"
                    />
                  </div>
                </div>

                {/* Stats */}
                <div className="pt-4 border-t border-gray-700">
                  <div className="text-sm text-gray-400">
                    <div className="flex justify-between mb-2">
                      <span>Total Stars:</span>
                      <span className="text-white">{mockStars.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Filtered:</span>
                      <span className="text-blue-400">{filteredStars.length}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Stars List */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="lg:col-span-3"
          >
            <div className="bg-gray-900/50 rounded-xl border border-gray-800 p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-white flex items-center">
                  <Star className="h-5 w-5 mr-2" />
                  Stars ({filteredStars.length})
                </h2>
              </div>
              
              {filteredStars.length > 0 ? (
                <div className="space-y-4">
                  {filteredStars.map((star, index) => (
                    <motion.div
                      key={star.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.05 }}
                      className="p-4 bg-black/30 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors group cursor-pointer"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <h3 className="text-lg font-medium text-white group-hover:text-blue-400 transition-colors">
                              {star.name}
                            </h3>
                            <span className="px-2 py-1 bg-blue-900/50 text-blue-300 text-xs rounded border border-blue-800">
                              {star.catalog}
                            </span>
                          </div>
                          <p className="text-sm text-gray-400 mb-3">{star.id}</p>
                          
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div>
                              <span className="text-gray-500">Magnitude:</span>
                              <span className="text-white ml-2">{star.magnitude}</span>
                            </div>
                            <div>
                              <span className="text-gray-500">Temperature:</span>
                              <span className="text-white ml-2">{star.temperature}K</span>
                            </div>
                            <div>
                              <span className="text-gray-500">Distance:</span>
                              <span className="text-white ml-2">{star.distance} ly</span>
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2 ml-4">
                          <button className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                            <Eye className="h-4 w-4" />
                          </button>
                          <ChevronRight className="h-5 w-5 text-gray-500 group-hover:text-gray-300 transition-colors" />
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <Database className="h-16 w-16 text-gray-600 mb-4" />
                  <h3 className="text-lg font-medium text-gray-400 mb-2">
                    No Stars Found
                  </h3>
                  <p className="text-gray-500 max-w-md">
                    No stars match your current filter criteria. Try adjusting your search parameters.
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
