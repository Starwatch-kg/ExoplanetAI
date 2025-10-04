# 🌌 AstroManas - High-Performance Exoplanet Detection System

<div align="center">

![AstroManas Logo](https://img.shields.io/badge/AstroManas-v3.0.0--enhanced-blue?style=for-the-badge&logo=nasa)
[![React](https://img.shields.io/badge/React-18.3.1-61DAFB?style=for-the-badge&logo=react)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6.3-3178C6?style=for-the-badge&logo=typescript)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-3.0+-003B57?style=for-the-badge&logo=sqlite)](https://sqlite.org/)
[![C++](https://img.shields.io/badge/C++-17+-00599C?style=for-the-badge&logo=cplusplus)](https://isocpp.org/)

**🚀 Production-ready AI system with revolutionary GPI method, C++ acceleration, and comprehensive database integration**

[🚀 Live Demo](#) • [📖 Documentation](#documentation) • [🛠️ Installation](#installation) • [🤝 Contributing](#contributing)

</div>

## 🚀 **Latest Major Update (v3.0.0-enhanced)**

### ✨ **Frontend Enhancements (v3.0)**
- 🏠 **Multi-Page Architecture** - Complete routing with dedicated pages for each feature
- 🎯 **Advanced Search Interface** - Method selection between BLS and GPI with parameter controls
- 📊 **Interactive Catalog** - Real-time exoplanet database browsing with filtering
- 📈 **Database Dashboard** - System metrics, search history, and performance monitoring
- 🎨 **Enhanced Navigation** - Professional header with active state highlighting
- 📱 **Mobile Responsive** - Optimized mobile experience with hamburger menu
- ⭐ **Cleaner Design** - Removed distracting star elements for better focus

### ⚡ **Backend Optimizations (v3.0)**
- 🗄️ **SQLite Database** - Complete database integration with real data storage
- ⚡ **C++ Acceleration** - High-performance modules for BLS and GPI analysis
- 🔧 **Synthetic Data Generation** - C++ powered GPI test data creation
- 📊 **Performance Metrics** - Comprehensive system monitoring and analytics
- 🔗 **Enhanced API** - New endpoints for database operations and performance comparison
- 🚀 **Real Data Processing** - Eliminated mock data, using actual NASA datasets
- 🏗️ **Improved Architecture** - Better separation of concerns and modular design

### 🧬 **Advanced GPI System (v3.0)**
- 🔬 **C++ Implementation** - Native high-performance gravitational phase analysis
- 🤖 **AI Enhancement** - Machine learning integration for improved accuracy
- 📊 **Real-time Metrics** - Live performance monitoring and confidence scoring
- 🔄 **Fallback System** - Automatic fallback to Python implementation if C++ fails
- 🎯 **Synthetic Data** - Advanced test data generation for development and testing
- ⚡ **Performance Boost** - Up to 10x faster processing with C++ acceleration

## ✨ Features

### 🎯 **Core Functionality**
- 🔍 **Real NASA TESS Data** - Direct integration with NASA's Transiting Exoplanet Survey Satellite
- 🤖 **AI-Powered Detection** - Advanced machine learning models for transit detection
- 📊 **Dual Analysis Methods** - Both BLS and revolutionary GPI detection algorithms
- ⚡ **C++ Acceleration** - High-performance native modules for critical computations
- 🗄️ **Database Integration** - SQLite-powered data persistence and analytics
- 📈 **Interactive Visualizations** - Beautiful light curve plots with Plotly.js

### 🎨 **Modern UI/UX**
- 🏠 **Multi-Page Application** - Dedicated pages for each major feature
- 🎯 **Method Selection** - Interactive choice between BLS and GPI analysis
- 📊 **Live Dashboard** - Real-time system metrics and database statistics
- 🔍 **Advanced Filtering** - Comprehensive exoplanet catalog with search and filters
- 🌓 **Dark/Light Theme** - Automatic system detection with manual toggle
- 📱 **Mobile Optimized** - Perfect responsive design for all devices
- 🎭 **Clean Aesthetics** - Professional design without distracting elements

### 🔬 **Scientific Components**
- **`<TargetInfo />`** - Stellar properties and observation statistics
- **`<LightCurveGraph />`** - Interactive time-series visualization
- **`<BLSDetails />`** - Detailed transit analysis with significance testing
- **`<ResultsPage />`** - Comprehensive results dashboard

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ and npm 9+
- **Python** 3.11+ with pip
- **C++ Compiler** (GCC 9+ or MSVC 2019+) for acceleration modules
- **FFTW3** library for signal processing
- **Git** for version control

### 1. Clone Repository
```bash
git clone https://github.com/Starwatch-kg/Exoplanet_AI.git
cd Exoplanet_AI
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```
Backend will start on `http://localhost:8001`

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend will start on `http://localhost:5176`

### 4. Quick Start Scripts
```bash
# Start backend
./scripts/start-backend.sh

# Start frontend (in another terminal)
./scripts/start-frontend.sh
```

## 🛠️ Technology Stack

### Frontend
- **React 18** with TypeScript
- **TailwindCSS** for styling
- **Plotly.js** for interactive charts
- **tsParticles** for animations
- **i18next** for internationalization
- **Framer Motion** for smooth animations

### Backend
- **FastAPI** for high-performance API
- **SQLite** with aiosqlite for database operations
- **C++** modules for performance-critical computations
- **lightkurve** for NASA data access
- **astroquery** for catalog queries
- **scikit-learn** for ML models
- **numpy/scipy** for scientific computing
- **FFTW3** for fast Fourier transforms

### Data Sources
- **NASA TESS** - Transiting Exoplanet Survey Satellite
- **NASA Kepler** - Legacy exoplanet mission data
- **MAST Archive** - Mikulski Archive for Space Telescopes

## 📊 Example Usage

### Search for Exoplanets
```python
# Example: Analyze TIC 441420236 (known exoplanet host)
target = "TIC 441420236"
catalog = "TIC"
mission = "TESS"

# The system will:
# 1. Download real TESS lightcurve data
# 2. Apply BLS algorithm for transit detection
# 3. Calculate statistical significance
# 4. Display interactive results
```

### API Endpoints
```bash
# Health check
GET /api/v1/health

# Search for exoplanets
POST /api/v1/search
{
  "target_name": "TIC 441420236",
  "catalog": "TIC",
  "mission": "TESS",
  "period_min": 1.0,
  "period_max": 10.0,
  "snr_threshold": 5.0
}

# Get NASA statistics
GET /api/v1/nasa-stats
```

## 🎯 Results Dashboard

The system provides comprehensive analysis results:

### Target Information
- **Stellar Properties**: Magnitude, coordinates, temperature
- **Data Quality**: Observation period, cadence, noise level
- **Coverage Statistics**: Data points, completeness percentage

### Light Curve Analysis
- **Interactive Plot**: Zoom, pan, export capabilities
- **Transit Highlighting**: Automatic detection of periodic signals
- **Quality Metrics**: SNR, significance, confidence levels

### BLS Detection Results
- **Orbital Period**: Best-fit period in days
- **Transit Depth**: Percentage flux decrease
- **Duration**: Transit length in hours
- **Statistical Significance**: σ-level detection confidence

## 🌟 Screenshots

<div align="center">

### Dark Theme Interface
![Dark Theme](https://via.placeholder.com/800x400/1a1a2e/ffffff?text=Dark+Theme+Interface)

### Light Curve Visualization
![Light Curve](https://via.placeholder.com/800x400/0f3460/ffffff?text=Interactive+Light+Curve)

### Results Dashboard
![Results](https://via.placeholder.com/800x400/533a7b/ffffff?text=Comprehensive+Results)

</div>

## 📈 Performance

- **Build Size**: 5.3MB JS, 40KB CSS (gzipped)
- **Load Time**: < 2s on modern browsers
- **Data Processing**: Real-time analysis of 10k+ data points
- **Responsiveness**: 60fps animations with WebGL acceleration

## 🔧 Development

### Project Structure
```
Exoplanet_AI/
├── backend/           # FastAPI backend
│   ├── api/          # API routes
│   ├── auth/         # Authentication
│   ├── core/         # Configuration and utilities
│   ├── ml/           # ML models
│   ├── services/     # Business logic
│   └── main.py       # Application entry point
├── frontend/         # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   ├── services/    # API services
│   │   └── types/       # TypeScript types
│   └── public/       # Static assets
├── config/           # Configuration files
└── scripts/          # Utility scripts
```

### Available Scripts

#### Frontend
```bash
npm run dev          # Development server
npm run build        # Production build
npm run preview      # Preview build
npm run lint         # ESLint check
npm run type-check   # TypeScript check
```

#### Backend
```bash
python main.py       # Start server
pytest              # Run tests
black .             # Code formatting
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA** for providing open access to TESS data
- **lightkurve** team for excellent Python tools
- **Plotly** for interactive visualization capabilities
- **React** and **FastAPI** communities

## 📞 Contact

- **Project Link**: [https://github.com/Starwatch-kg/Exoplanet_AI](https://github.com/Starwatch-kg/Exoplanet_AI)
- **Issues**: [GitHub Issues](https://github.com/Starwatch-kg/Exoplanet_AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Starwatch-kg/Exoplanet_AI/discussions)

---

<div align="center">

**Made with ❤️ for the astronomy community**

⭐ **Star this repo if you find it useful!** ⭐

</div>
