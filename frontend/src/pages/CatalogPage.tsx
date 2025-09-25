import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useForm } from 'react-hook-form'
import { 
  Search, 
  Database, 
  Star, 
  Shuffle,
  ExternalLink,
  Filter,
  ChevronDown,
  Info
} from 'lucide-react'
import ApiService from '../services/api'
import { useAppStore } from '../store/useAppStore'
import LoadingSpinner from '../components/ui/LoadingSpinner'

interface CatalogSearchForm {
  query: string
  catalog: 'TIC' | 'KIC' | 'EPIC'
  limit: number
}

interface StarTarget {
  target_id: string
  catalog: string
  ra: number
  dec: number
  magnitude: number
  temperature?: number
  radius?: number
  mass?: number
  distance?: number
  stellar_type?: string
}

export default function CatalogPage() {
  const [searchResults, setSearchResults] = useState<StarTarget[]>([])
  const [randomTargets, setRandomTargets] = useState<StarTarget[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [catalogsData, setCatalogsData] = useState<any>(null)
  const [showFilters, setShowFilters] = useState(false)
  const { addToast } = useAppStore()

  const { register, handleSubmit, watch, setValue, formState: { errors } } = useForm<CatalogSearchForm>({
    defaultValues: {
      query: '',
      catalog: 'TIC',
      limit: 10
    }
  })

  const watchedCatalog = watch('catalog')

  // Load catalogs info
  useEffect(() => {
    ApiService.getCatalogs()
      .then(setCatalogsData)
      .catch(() => {
        addToast({
          type: 'error',
          title: 'Ошибка загрузки',
          message: 'Не удалось загрузить информацию о каталогах'
        })
      })
  }, [])

  // Load random targets on mount
  useEffect(() => {
    loadRandomTargets()
  }, [watchedCatalog])

  const loadRandomTargets = async () => {
    try {
      const result = await ApiService.getRandomTargets(watchedCatalog, 5, 14)
      setRandomTargets(result.targets)
    } catch (error) {
      console.error('Failed to load random targets:', error)
    }
  }

  const handleSearch = async (data: CatalogSearchForm) => {
    if (!data.query.trim()) {
      addToast({
        type: 'warning',
        title: 'Введите запрос',
        message: 'Укажите название звезды или идентификатор для поиска'
      })
      return
    }

    try {
      setIsLoading(true)
      const result = await ApiService.searchTargets(data.query, data.catalog, data.limit)
      setSearchResults(result.targets)
      
      if (result.targets.length === 0) {
        addToast({
          type: 'info',
          title: 'Ничего не найдено',
          message: 'Попробуйте изменить запрос или выбрать другой каталог'
        })
      } else {
        addToast({
          type: 'success',
          title: 'Поиск завершен',
          message: `Найдено ${result.targets.length} объектов`
        })
      }
    } catch (error: any) {
      addToast({
        type: 'error',
        title: 'Ошибка поиска',
        message: error.message || 'Произошла ошибка при поиске в каталоге'
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleAnalyzeTarget = (targetId: string) => {
    // Navigate to analysis page with target
    window.location.href = `/analysis/${targetId}`
  }

  const formatCoordinate = (value: number, isRA: boolean = false) => {
    if (isRA) {
      const hours = Math.floor(value / 15)
      const minutes = Math.floor((value / 15 - hours) * 60)
      const seconds = ((value / 15 - hours) * 60 - minutes) * 60
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toFixed(1).padStart(4, '0')}`
    } else {
      const degrees = Math.floor(Math.abs(value))
      const minutes = Math.floor((Math.abs(value) - degrees) * 60)
      const seconds = ((Math.abs(value) - degrees) * 60 - minutes) * 60
      const sign = value >= 0 ? '+' : '-'
      return `${sign}${degrees.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toFixed(1).padStart(4, '0')}`
    }
  }

  const StarCard = ({ star }: { star: StarTarget }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="nasa-card hover:animate-pulse-glow transition-all duration-300"
    >
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-lg font-bold text-white mb-1">{star.target_id}</h3>
          <p className="text-sm text-gray-400">{star.catalog} Catalog</p>
        </div>
        <div className="flex items-center space-x-2">
          <Star className="h-4 w-4 text-yellow-400" />
          <span className="text-sm font-mono text-yellow-400">
            {star.magnitude?.toFixed(2)}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
        <div>
          <p className="text-gray-400">RA</p>
          <p className="text-white font-mono">{formatCoordinate(star.ra, true)}</p>
        </div>
        <div>
          <p className="text-gray-400">Dec</p>
          <p className="text-white font-mono">{formatCoordinate(star.dec)}</p>
        </div>
        {star.temperature && (
          <div>
            <p className="text-gray-400">Температура</p>
            <p className="text-white font-mono">{star.temperature.toFixed(0)} K</p>
          </div>
        )}
        {star.stellar_type && (
          <div>
            <p className="text-gray-400">Спектральный класс</p>
            <p className="text-white font-mono">{star.stellar_type}</p>
          </div>
        )}
      </div>

      <button
        onClick={() => handleAnalyzeTarget(star.target_id)}
        className="w-full nasa-btn nasa-btn-success flex items-center justify-center space-x-2"
      >
        <Search className="h-4 w-4" />
        <span>Анализировать</span>
        <ExternalLink className="h-4 w-4" />
      </button>
    </motion.div>
  )

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="nasa-title">
            Каталог звезд
          </h1>
          <p className="nasa-subtitle mb-6">
            Поиск и исследование звездных объектов
          </p>
          <p className="nasa-text text-lg max-w-4xl mx-auto">
            Исследуйте каталоги TESS, Kepler и K2 для поиска интересных звездных объектов 
            и потенциальных кандидатов в экзопланеты
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-4 gap-8">
          {/* Search Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="nasa-card sticky top-24">
              <form onSubmit={handleSubmit(handleSearch)} className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    <Database className="inline h-4 w-4 mr-1" />
                    Каталог
                  </label>
                  <select
                    {...register('catalog')}
                    className="nasa-input"
                  >
                    <option value="TIC">TIC (TESS)</option>
                    <option value="KIC">KIC (Kepler)</option>
                    <option value="EPIC">EPIC (K2)</option>
                  </select>
                  {catalogsData?.description && (
                    <p className="text-xs text-gray-400 mt-1">
                      {catalogsData.description[watchedCatalog]}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    <Search className="inline h-4 w-4 mr-1" />
                    Поиск объекта
                  </label>
                  <input
                    {...register('query')}
                    type="text"
                    placeholder="TIC 441420236, HD 209458..."
                    className="nasa-input"
                  />
                </div>

                <div>
                  <button
                    type="button"
                    onClick={() => setShowFilters(!showFilters)}
                    className="flex items-center justify-between w-full p-3 bg-space-700/30 rounded-lg border border-space-600 hover:bg-space-700/50 transition-colors"
                  >
                    <div className="flex items-center space-x-2">
                      <Filter className="h-4 w-4" />
                      <span className="text-sm font-medium">Фильтры</span>
                    </div>
                    <ChevronDown className={`h-4 w-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
                  </button>

                  {showFilters && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="mt-4 p-4 bg-space-800/30 rounded-lg border border-space-600"
                    >
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Лимит результатов
                        </label>
                        <select
                          {...register('limit')}
                          className="nasa-input"
                        >
                          <option value={5}>5 объектов</option>
                          <option value={10}>10 объектов</option>
                          <option value={20}>20 объектов</option>
                          <option value={50}>50 объектов</option>
                        </select>
                      </div>
                    </motion.div>
                  )}
                </div>

                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full nasa-btn py-3 flex items-center justify-center space-x-2 disabled:opacity-50"
                >
                  {isLoading ? (
                    <LoadingSpinner size="sm" />
                  ) : (
                    <>
                      <Search className="h-5 w-5" />
                      <span>Поиск в каталоге</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={loadRandomTargets}
                  className="w-full nasa-btn nasa-btn-success py-3 flex items-center justify-center space-x-2"
                >
                  <Shuffle className="h-5 w-5" />
                  <span>Случайные объекты</span>
                </button>
              </form>

              {/* Info Panel */}
              <div className="mt-6 p-4 bg-primary-600/10 rounded-lg border border-primary-500/30">
                <div className="flex items-start space-x-2">
                  <Info className="h-4 w-4 text-primary-400 mt-0.5 flex-shrink-0" />
                  <div className="text-sm">
                    <p className="text-primary-300 font-medium mb-1">Совет</p>
                    <p className="text-gray-300">
                      Используйте полные идентификаторы (например, TIC 441420236) 
                      для точного поиска или частичные названия для широкого поиска.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Results Area */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-3"
          >
            {/* Search Results */}
            {searchResults.length > 0 && (
              <div className="mb-8">
                <h2 className="text-2xl font-bold text-white mb-6">
                  Результаты поиска ({searchResults.length})
                </h2>
                <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
                  {searchResults.map((star, index) => (
                    <StarCard key={`search-${star.target_id}-${index}`} star={star} />
                  ))}
                </div>
              </div>
            )}

            {/* Random Targets */}
            {randomTargets.length > 0 && (
              <div>
                <h2 className="text-2xl font-bold text-white mb-6">
                  Рекомендуемые объекты ({watchedCatalog})
                </h2>
                <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
                  {randomTargets.map((star, index) => (
                    <StarCard key={`random-${star.target_id}-${index}`} star={star} />
                  ))}
                </div>
              </div>
            )}

            {/* Empty State */}
            {!isLoading && searchResults.length === 0 && randomTargets.length === 0 && (
              <div className="nasa-card text-center py-12">
                <Database className="h-16 w-16 text-gray-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-400 mb-2">
                  Каталог готов к поиску
                </h3>
                <p className="text-gray-500">
                  Введите название звезды или используйте случайный выбор для начала исследования
                </p>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  )
}
