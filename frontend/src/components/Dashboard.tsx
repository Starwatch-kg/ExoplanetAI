import React from 'react';
import { motion } from 'framer-motion';
import { Search, Brain, BarChart3, Zap, Star, Circle } from 'lucide-react';

interface DashboardProps {
  onStartSearch: () => void;
}

const Dashboard: React.FC<DashboardProps> = ({ onStartSearch }) => {
  const features = [
    {
      icon: <Brain className="w-8 h-8 text-cyan-400" />,
      title: "ИИ-анализ",
      description: "Глубокое обучение для детекции транзитов экзопланет"
    },
    {
      icon: <BarChart3 className="w-8 h-8 text-purple-400" />,
      title: "Визуализация",
      description: "Интерактивные графики и анимации космических явлений"
    },
    {
      icon: <Zap className="w-8 h-8 text-yellow-400" />,
      title: "Быстрый поиск",
      description: "Анализ тысяч звезд за считанные минуты"
    }
  ];

  return (
    <div className="min-h-screen flex flex-col">
      {/* Hero Section */}
      <section className="flex-1 flex items-center justify-center px-6 py-20">
        <div className="max-w-6xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-12"
          >
            <h1 className="text-6xl md:text-8xl font-bold mb-6">
              <span className="bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Exoplanet
              </span>
              <br />
              <span className="text-white">AI Search</span>
            </h1>
            <p className="text-xl text-slate-300 max-w-3xl mx-auto leading-relaxed">
              Революционная система поиска экзопланет с использованием искусственного интеллекта. 
              Анализируйте кривые блеска звезд и находите новые миры в нашей галактике.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mb-16"
          >
            <button
              onClick={onStartSearch}
              className="cosmic-button text-lg px-8 py-4 inline-flex items-center space-x-3 group"
            >
              <Search className="w-6 h-6 group-hover:rotate-12 transition-transform" />
              <span>Начать поиск экзопланет</span>
              <Star className="w-5 h-5 animate-pulse" />
            </button>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto"
          >
            {features.map((feature, index) => (
              <motion.div
                key={index}
                whileHover={{ scale: 1.05, y: -5 }}
                className="space-card text-center group"
              >
                <div className="mb-4 flex justify-center group-hover:scale-110 transition-transform">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-3 text-white">
                  {feature.title}
                </h3>
                <p className="text-slate-400 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.6 }}
        className="py-16 px-6"
      >
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { number: "5000+", label: "Проанализированных звезд" },
              { number: "150+", label: "Найденных кандидатов" },
              { number: "99.8%", label: "Точность алгоритма" },
              { number: "24/7", label: "Непрерывный мониторинг" }
            ].map((stat, index) => (
              <motion.div
                key={index}
                whileHover={{ scale: 1.05 }}
                className="text-center space-card"
              >
                <div className="text-3xl md:text-4xl font-bold text-cyan-400 mb-2">
                  {stat.number}
                </div>
                <div className="text-slate-400 text-sm">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      {/* Floating Planet Animation */}
      <div className="fixed top-1/4 right-10 w-32 h-32 opacity-20 pointer-events-none">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          className="w-full h-full relative"
        >
          <Circle className="w-full h-full text-cyan-400" />
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;
