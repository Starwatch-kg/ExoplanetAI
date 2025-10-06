# 🚀 ExoplanetAI - Примеры использования улучшенного ИИ

## 🔬 Примеры API запросов

### 1. Умный анализ кривой блеска

```bash
# Анализ по названию цели
curl -X POST "http://localhost:8001/api/v1/ai/analyze_lightcurve" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "target_name": "TOI-715",
    "mission": "TESS",
    "auto_detect_cadence": true,
    "adaptive_detrending": true,
    "include_uncertainty": true,
    "explain_prediction": true
  }'
```

**Ответ:**
```json
{
  "target_name": "TOI-715",
  "predicted_class": "Confirmed",
  "confidence_score": 0.87,
  "uncertainty_bounds": [0.82, 0.92],
  "transit_probability": 0.91,
  "signal_characteristics": {
    "snr_estimate": 12.4,
    "data_points": 18743,
    "time_span_days": 27.4
  },
  "feature_importance": [0.23, 0.18, 0.15, 0.12, 0.08, ...],
  "decision_reasoning": [
    "Для класса Confirmed: ключевые признаки - transit_depth, snr, duration",
    "Высокая глубина транзита (0.8%) указывает на реальную планету",
    "SNR 12.4 значительно превышает пороговое значение"
  ],
  "recommendations": [
    "Высокая уверенность в классификации",
    "Рекомендуется follow-up наблюдения для подтверждения"
  ],
  "data_quality_metrics": {
    "photometric_precision": 0.001,
    "systematic_noise_level": 0.02,
    "data_completeness": 0.95,
    "instrumental_effects_score": 0.03
  },
  "processing_time_ms": 342.5,
  "model_version": "enhanced_v2.0"
}
```

### 2. Умный поиск экзопланет

```bash
# Поиск с ИИ ранжированием
curl -X POST "http://localhost:8001/api/v1/ai/smart_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "TOI",
    "filters": {
      "confidence_min": 0.7,
      "snr_min": 5.0,
      "data_quality_min": 0.8,
      "missions": ["TESS", "Kepler"],
      "planet_types": ["confirmed", "candidate"],
      "period_range": [1.0, 100.0]
    },
    "use_ai_ranking": true,
    "max_results": 10,
    "include_similar": true
  }'
```

**Ответ:**
```json
{
  "results": [
    {
      "name": "TOI-715.01",
      "predicted_class": "Confirmed",
      "confidence_score": 0.87,
      "ai_score": 0.92,
      "signal_characteristics": {
        "snr_estimate": 12.4,
        "data_points": 18743,
        "time_span_days": 27.4
      },
      "disposition": "CONFIRMED",
      "orbital_period": 19.3,
      "planet_radius": 1.55,
      "ai_ranking_applied": true
    }
  ],
  "total_found": 156,
  "ai_ranked": true,
  "recommendations": [
    "Найдено 45 подтвержденных экзопланет",
    "Найдено 111 кандидатов в экзопланеты"
  ],
  "search_time_ms": 1247.3,
  "filters_applied": {
    "confidence_min": 0.7,
    "snr_min": 5.0,
    "missions": ["TESS", "Kepler"]
  }
}
```

### 3. Пакетный анализ

```bash
# Анализ нескольких целей параллельно
curl -X POST "http://localhost:8001/api/v1/ai/batch_analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "targets": ["TOI-715", "Kepler-452b", "TRAPPIST-1e"],
    "analysis_params": {
      "mission": "TESS",
      "include_uncertainty": true,
      "explain_prediction": false
    },
    "parallel_limit": 3
  }'
```

## 🌐 Примеры фронтенд интеграции

### 1. React Hook для умного поиска

```typescript
// hooks/useSmartSearch.ts
import { useState, useCallback } from 'react';

interface SearchFilters {
  confidenceMin: number;
  snrMin: number;
  dataQualityMin: number;
  missions: string[];
  planetTypes: string[];
}

interface SearchResult {
  name: string;
  predicted_class: string;
  confidence_score: number;
  ai_score: number;
}

export const useSmartSearch = () => {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const search = useCallback(async (query: string, filters: SearchFilters) => {
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('/api/v1/ai/smart_search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          filters,
          use_ai_ranking: true,
          max_results: 20,
          include_similar: true
        })
      });

      if (!response.ok) {
        throw new Error(`Search failed: ${response.status}`);
      }

      const data = await response.json();
      setResults(data.results);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { results, isLoading, error, search };
};
```

### 2. Компонент анализа с загрузкой файла

```typescript
// components/FileAnalysis.tsx
import React, { useState } from 'react';
import { Upload, Brain } from 'lucide-react';

const FileAnalysis: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile) {
      setFile(uploadedFile);
    }
  };

  const analyzeFile = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    
    try {
      // Читаем файл
      const text = await file.text();
      const lines = text.split('\n').filter(line => line.trim());
      
      const data = lines.slice(1).map(line => {
        const values = line.split(/[,\s]+/).map(v => parseFloat(v.trim()));
        return { time: values[0], flux: values[1] };
      }).filter(row => !isNaN(row.time) && !isNaN(row.flux));

      // Отправляем на анализ
      const response = await fetch('/api/v1/ai/analyze_lightcurve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          time_data: data.map(d => d.time),
          flux_data: data.map(d => d.flux),
          mission: 'TESS',
          include_uncertainty: true,
          explain_prediction: true
        })
      });

      const result = await response.json();
      setAnalysis(result);
      
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
        <Brain className="w-6 h-6" />
        Анализ загруженного файла
      </h2>

      <div className="space-y-4">
        <div>
          <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-600 border-dashed rounded-lg cursor-pointer bg-gray-700 hover:bg-gray-600">
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <Upload className="w-8 h-8 mb-4 text-gray-400" />
              <p className="mb-2 text-sm text-gray-400">
                <span className="font-semibold">Нажмите для загрузки</span>
              </p>
              <p className="text-xs text-gray-500">CSV файлы (время, поток)</p>
            </div>
            <input type="file" accept=".csv,.txt" onChange={handleFileUpload} className="hidden" />
          </label>
        </div>

        {file && (
          <div className="text-sm text-green-400">
            ✓ Файл загружен: {file.name}
          </div>
        )}

        <button
          onClick={analyzeFile}
          disabled={!file || isAnalyzing}
          className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
        >
          {isAnalyzing ? 'Анализ...' : 'Анализировать'}
        </button>

        {analysis && (
          <div className="mt-6 bg-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-white mb-2">Результаты анализа</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-400">Класс:</span>
                <span className="text-white ml-2">{analysis.predicted_class}</span>
              </div>
              <div>
                <span className="text-gray-400">Уверенность:</span>
                <span className="text-white ml-2">{(analysis.confidence_score * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileAnalysis;
```

## 🐍 Python SDK для разработчиков

```python
# exoplanet_ai_sdk.py
import requests
import pandas as pd
from typing import Dict, List, Optional, Union
import numpy as np

class ExoplanetAI:
    """Python SDK для ExoplanetAI API"""
    
    def __init__(self, api_url: str = "http://localhost:8001", api_key: str = None):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def analyze_target(self, 
                      target_name: str,
                      mission: str = "TESS",
                      include_uncertainty: bool = True,
                      explain_prediction: bool = True) -> Dict:
        """Анализ цели по названию"""
        
        data = {
            "target_name": target_name,
            "mission": mission,
            "auto_detect_cadence": True,
            "adaptive_detrending": True,
            "include_uncertainty": include_uncertainty,
            "explain_prediction": explain_prediction
        }
        
        response = self.session.post(f"{self.api_url}/api/v1/ai/analyze_lightcurve", json=data)
        response.raise_for_status()
        return response.json()
    
    def analyze_lightcurve(self,
                          time_data: List[float],
                          flux_data: List[float],
                          flux_err_data: Optional[List[float]] = None,
                          mission: str = "TESS") -> Dict:
        """Анализ кривой блеска из массивов данных"""
        
        data = {
            "time_data": time_data,
            "flux_data": flux_data,
            "flux_err_data": flux_err_data,
            "mission": mission,
            "auto_detect_cadence": True,
            "adaptive_detrending": True,
            "include_uncertainty": True,
            "explain_prediction": True
        }
        
        response = self.session.post(f"{self.api_url}/api/v1/ai/analyze_lightcurve", json=data)
        response.raise_for_status()
        return response.json()
    
    def smart_search(self,
                    query: str,
                    confidence_min: float = 0.7,
                    snr_min: float = 5.0,
                    missions: List[str] = None,
                    max_results: int = 20) -> Dict:
        """Умный поиск экзопланет"""
        
        if missions is None:
            missions = ["TESS", "Kepler"]
        
        data = {
            "query": query,
            "filters": {
                "confidence_min": confidence_min,
                "snr_min": snr_min,
                "data_quality_min": 0.8,
                "missions": missions,
                "planet_types": ["confirmed", "candidate"]
            },
            "use_ai_ranking": True,
            "max_results": max_results,
            "include_similar": True
        }
        
        response = self.session.post(f"{self.api_url}/api/v1/ai/smart_search", json=data)
        response.raise_for_status()
        return response.json()
    
    def batch_analyze(self,
                     targets: List[str],
                     mission: str = "TESS",
                     parallel_limit: int = 5) -> Dict:
        """Пакетный анализ множественных целей"""
        
        data = {
            "targets": targets,
            "analysis_params": {
                "mission": mission,
                "include_uncertainty": True,
                "explain_prediction": False
            },
            "parallel_limit": parallel_limit
        }
        
        response = self.session.post(f"{self.api_url}/api/v1/ai/batch_analyze", json=data)
        response.raise_for_status()
        return response.json()
    
    def analyze_csv_file(self, file_path: str, target_name: str = "uploaded_data") -> Dict:
        """Анализ CSV файла с кривой блеска"""
        
        df = pd.read_csv(file_path)
        
        # Автоматическое определение колонок
        time_col = None
        flux_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'time' in col_lower or 'bjd' in col_lower or 'mjd' in col_lower:
                time_col = col
            elif 'flux' in col_lower or 'mag' in col_lower:
                flux_col = col
        
        if time_col is None or flux_col is None:
            time_col = df.columns[0]
            flux_col = df.columns[1]
        
        return self.analyze_lightcurve(
            time_data=df[time_col].tolist(),
            flux_data=df[flux_col].tolist()
        )

# Пример использования
if __name__ == "__main__":
    # Инициализация клиента
    client = ExoplanetAI(api_url="http://localhost:8001", api_key="your_api_key")
    
    # Анализ известной экзопланеты
    result = client.analyze_target("TOI-715")
    print(f"Класс: {result['predicted_class']}")
    print(f"Уверенность: {result['confidence_score']:.1%}")
    print(f"Рекомендации: {result['recommendations']}")
    
    # Умный поиск
    search_results = client.smart_search("TOI", confidence_min=0.8)
    print(f"Найдено: {len(search_results['results'])} объектов")
    
    # Пакетный анализ
    batch_results = client.batch_analyze(["TOI-715", "Kepler-452b", "TRAPPIST-1e"])
    print(f"Успешно проанализировано: {len(batch_results['successful_analyses'])} объектов")
    
    # Анализ файла
    file_result = client.analyze_csv_file("lightcurve_data.csv")
    print(f"Анализ файла: {file_result['predicted_class']}")
```

## 📊 Jupyter Notebook примеры

```python
# Пример анализа в Jupyter Notebook
import matplotlib.pyplot as plt
import numpy as np
from exoplanet_ai_sdk import ExoplanetAI

# Инициализация
client = ExoplanetAI()

# Анализ цели
result = client.analyze_target("TOI-715")

# Визуализация результатов
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# График важности признаков
feature_names = ['Transit Depth', 'SNR', 'Duration', 'Asymmetry', 'V-shape', 
                'U-shape', 'Box-shape', 'Frequency Power', 'Data Quality', 'Noise']
importance = result['feature_importance'][:10]

axes[0, 0].barh(feature_names, importance)
axes[0, 0].set_title('Feature Importance')
axes[0, 0].set_xlabel('Importance Score')

# Pie chart уверенности
confidence = result['confidence_score']
uncertainty = 1 - confidence
axes[0, 1].pie([confidence, uncertainty], labels=['Confident', 'Uncertain'], 
               autopct='%1.1f%%', startangle=90)
axes[0, 1].set_title(f'Prediction Confidence\n{result["predicted_class"]}')

# Метрики качества данных
quality_metrics = result['data_quality_metrics']
metrics_names = list(quality_metrics.keys())
metrics_values = list(quality_metrics.values())

axes[1, 0].bar(range(len(metrics_names)), metrics_values)
axes[1, 0].set_xticks(range(len(metrics_names)))
axes[1, 0].set_xticklabels(metrics_names, rotation=45)
axes[1, 0].set_title('Data Quality Metrics')
axes[1, 0].set_ylabel('Score')

# Характеристики сигнала
signal_chars = result['signal_characteristics']
char_names = list(signal_chars.keys())
char_values = list(signal_chars.values())

axes[1, 1].bar(char_names, char_values)
axes[1, 1].set_title('Signal Characteristics')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Вывод рекомендаций
print("🤖 AI Recommendations:")
for i, rec in enumerate(result['recommendations'], 1):
    print(f"{i}. {rec}")

print("\n🧠 AI Reasoning:")
for i, reason in enumerate(result['decision_reasoning'], 1):
    print(f"{i}. {reason}")
```

## 🔧 Настройка и развертывание

### Docker Compose для разработки

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  exoplanet-ai-backend:
    build: 
      context: ./backend
      dockerfile: Dockerfile.dev
    ports:
      - "8001:8001"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/exoplanet_ai
      - NASA_API_KEY=${NASA_API_KEY}
    volumes:
      - ./backend:/app
      - ./data:/app/data
    depends_on:
      - redis
      - postgres
    command: uvicorn main:app --host 0.0.0.0 --port 8001 --reload

  exoplanet-ai-frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8001
    volumes:
      - ./frontend:/app
      - /app/node_modules
    command: npm run dev

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=exoplanet_ai
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Запуск системы

```bash
# Клонирование и настройка
git clone https://github.com/your-repo/exoplanet-ai.git
cd exoplanet-ai

# Настройка переменных окружения
cp .env.example .env
# Отредактируйте .env файл

# Запуск с Docker Compose
docker-compose -f docker-compose.dev.yml up -d

# Или запуск вручную
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

cd ../frontend
npm install
npm run dev
```

Система будет доступна по адресам:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs
