import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { 
  Camera, 
  Upload, 
  Image as ImageIcon, 
  Brain,
  CheckCircle,
  AlertCircle,
  Loader,
  Trash2,
  BarChart3
} from 'lucide-react';

interface ClassificationResult {
  class: string;
  confidence: number;
  description: string;
}

interface UploadedImage {
  id: string;
  file: File;
  url: string;
  result?: ClassificationResult;
  status: 'uploaded' | 'processing' | 'completed' | 'error';
}

const WorkingImageClassification: React.FC = () => {
  const [images, setImages] = useState<UploadedImage[]>([]);
  const [selectedModel, setSelectedModel] = useState('exoplanet_cnn');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const models = [
    { id: 'exoplanet_cnn', name: 'ExoplanetCNN', description: 'Базовая CNN для детекции транзитов' },
    { id: 'exoplanet_resnet', name: 'ExoplanetResNet', description: 'ResNet с остаточными связями' },
    { id: 'exoplanet_densenet', name: 'ExoplanetDenseNet', description: 'DenseNet с плотными связями' },
    { id: 'exoplanet_attention', name: 'ExoplanetAttention', description: 'CNN с механизмом внимания' }
  ];

  const exampleClasses = [
    { name: 'Транзит экзопланеты', color: 'text-green-400', description: 'Обнаружен транзитный сигнал' },
    { name: 'Затменная двойная', color: 'text-blue-400', description: 'Система двойных звезд' },
    { name: 'Переменная звезда', color: 'text-yellow-400', description: 'Пульсирующая переменная' },
    { name: 'Шум', color: 'text-red-400', description: 'Инструментальная помеха' }
  ];

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  };

  const handleFiles = (files: FileList) => {
    const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    
    imageFiles.forEach(file => {
      const id = `img_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const url = URL.createObjectURL(file);
      
      const newImage: UploadedImage = {
        id,
        file,
        url,
        status: 'uploaded'
      };
      
      setImages(prev => [...prev, newImage]);
      
      // Автоматически запускаем классификацию
      setTimeout(() => {
        classifyImage(id);
      }, 500);
    });
  };

  const classifyImage = async (imageId: string) => {
    try {
      // Устанавливаем статус обработки
      setImages(prev => prev.map(img => 
        img.id === imageId ? { ...img, status: 'processing' } : img
      ));

      // Находим изображение
      const image = images.find(img => img.id === imageId);
      if (!image) return;

      // Реальная классификация через API
      const formData = new FormData();
      formData.append('image', image.file);
      formData.append('model', selectedModel);

      const response = await fetch('/api/cnn/classify', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setImages(prev => prev.map(img => 
          img.id === imageId ? { 
            ...img, 
            status: 'completed',
            result: result
          } : img
        ));
      } else {
        throw new Error('Ошибка классификации');
      }
    } catch (error) {
      console.error('Ошибка классификации:', error);
      
      // Fallback результат при ошибке API
      const fallbackResult = {
        class: 'Ошибка анализа',
        confidence: 0.0,
        description: 'Не удалось проанализировать изображение'
      };
      
      setImages(prev => prev.map(img => 
        img.id === imageId ? { 
          ...img, 
          status: 'error',
          result: fallbackResult
        } : img
      ));
    }
  };

  const removeImage = (imageId: string) => {
    setImages(prev => {
      const image = prev.find(img => img.id === imageId);
      if (image) {
        URL.revokeObjectURL(image.url);
      }
      return prev.filter(img => img.id !== imageId);
    });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'processing':
        return <Loader className="w-4 h-4 animate-spin text-blue-400" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      default:
        return <ImageIcon className="w-4 h-4 text-gray-400" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    if (confidence >= 0.4) return 'text-orange-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-lg">
      {/* Заголовок */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div className="card-header">
          <h2 className="card-title flex items-center gap-sm">
            <Camera className="w-6 h-6 text-primary" />
            Классификация изображений
          </h2>
          <div className="text-sm text-secondary">
            Анализ астрономических изображений с помощью CNN
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-lg">
        {/* Левая панель - Загрузка и настройки */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="space-y-md"
        >
          {/* Выбор модели */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title flex items-center gap-sm">
                <Brain className="w-5 h-5 text-purple-400" />
                Модель классификации
              </h3>
            </div>
            <div className="card-body">
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="form-select w-full"
              >
                {models.map(model => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
              <div className="text-xs text-tertiary mt-2">
                {models.find(m => m.id === selectedModel)?.description}
              </div>
            </div>
          </div>

          {/* Зона загрузки */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title flex items-center gap-sm">
                <Upload className="w-5 h-5 text-blue-400" />
                Загрузка изображений
              </h3>
            </div>
            <div className="card-body">
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  dragActive 
                    ? 'border-blue-400 bg-blue-400/10' 
                    : 'border-gray-600 hover:border-gray-500'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <Camera className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-300 mb-4">
                  Перетащите изображения сюда или
                </p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn btn-primary"
                >
                  Выберите файлы
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleFileInput}
                  className="hidden"
                />
              </div>
            </div>
          </div>

          {/* Примеры классов */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title flex items-center gap-sm">
                <BarChart3 className="w-5 h-5 text-green-400" />
                Классы объектов
              </h3>
            </div>
            <div className="card-body">
              <div className="space-y-sm">
                {exampleClasses.map((cls, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full bg-current ${cls.color}`} />
                    <div>
                      <div className={`text-sm font-medium ${cls.color}`}>
                        {cls.name}
                      </div>
                      <div className="text-xs text-gray-400">
                        {cls.description}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Правая панель - Результаты */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2"
        >
          <div className="card">
            <div className="card-header">
              <h3 className="card-title flex items-center gap-sm">
                <ImageIcon className="w-5 h-5 text-blue-400" />
                Результаты классификации ({images.length})
              </h3>
            </div>
            <div className="card-body">
              {images.length === 0 ? (
                <div className="text-center py-12">
                  <Camera className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400">
                    Загрузите изображения для начала классификации
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-md">
                  {images.map((image, index) => (
                    <motion.div
                      key={image.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-gray-800/50 rounded-lg p-md"
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(image.status)}
                          <span className="text-sm font-medium text-white">
                            {image.file.name}
                          </span>
                        </div>
                        <button
                          onClick={() => removeImage(image.id)}
                          className="text-red-400 hover:text-red-300 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>

                      <div className="mb-3">
                        <img
                          src={image.url}
                          alt="Uploaded"
                          className="w-full h-32 object-cover rounded-lg"
                        />
                      </div>

                      {image.result && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-white">
                              {image.result.class}
                            </span>
                            <span className={`text-sm font-bold ${getConfidenceColor(image.result.confidence)}`}>
                              {(image.result.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-700 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full transition-all duration-500 ${
                                image.result.confidence >= 0.8 ? 'bg-green-400' :
                                image.result.confidence >= 0.6 ? 'bg-yellow-400' :
                                image.result.confidence >= 0.4 ? 'bg-orange-400' : 'bg-red-400'
                              }`}
                              style={{ width: `${image.result.confidence * 100}%` }}
                            />
                          </div>
                          <p className="text-xs text-gray-400">
                            {image.result.description}
                          </p>
                        </div>
                      )}

                      {image.status === 'processing' && (
                        <div className="text-center py-4">
                          <Loader className="w-6 h-6 animate-spin text-blue-400 mx-auto mb-2" />
                          <p className="text-sm text-gray-400">Анализ изображения...</p>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default WorkingImageClassification;
