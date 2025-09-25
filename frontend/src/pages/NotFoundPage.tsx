import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Home, Search, ArrowLeft, Radar } from 'lucide-react'

export default function NotFoundPage() {
  return (
    <div className="min-h-screen flex items-center justify-center py-8 px-4">
      <div className="max-w-md mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          {/* Animated 404 */}
          <motion.div
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
            className="text-8xl font-bold text-gradient mb-4"
          >
            404
          </motion.div>
          
          {/* Floating telescope */}
          <motion.div
            animate={{ y: [-10, 10, -10] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            className="inline-block p-4 bg-cosmic-gradient rounded-full mb-6"
          >
            <Radar className="h-12 w-12 text-white" />
          </motion.div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-white mb-4">
            Страница не найдена
          </h1>
          <p className="text-lg text-gray-300 mb-2">
            Похоже, эта страница затерялась в космосе
          </p>
          <p className="text-gray-400">
            Возможно, она была поглощена черной дырой или просто не существует
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="space-y-4"
        >
          <div className="flex flex-col sm:flex-row gap-4">
            <Link
              to="/"
              className="btn-cosmic px-6 py-3 flex items-center justify-center space-x-2 rounded-lg"
            >
              <Home className="h-5 w-5" />
              <span>На главную</span>
            </Link>
            
            <Link
              to="/search"
              className="btn-secondary px-6 py-3 flex items-center justify-center space-x-2 rounded-lg"
            >
              <Search className="h-5 w-5" />
              <span>Поиск экзопланет</span>
            </Link>
          </div>
          
          <button
            onClick={() => window.history.back()}
            className="text-primary-400 hover:text-primary-300 flex items-center justify-center space-x-2 mx-auto"
          >
            <ArrowLeft className="h-4 w-4" />
            <span>Вернуться назад</span>
          </button>
        </motion.div>

        {/* Floating particles */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-white rounded-full opacity-30"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                y: [0, -100, 0],
                opacity: [0.3, 0.8, 0.3],
              }}
              transition={{
                duration: 3 + Math.random() * 2,
                repeat: Infinity,
                delay: Math.random() * 2,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
