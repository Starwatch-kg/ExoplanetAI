"""
Optional dependencies handler for ExoplanetAI
Gracefully handles missing scientific libraries in production
"""

import logging
import warnings
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global flags for available libraries
SCIPY_AVAILABLE = False
SKLEARN_AVAILABLE = False
ASTROPY_AVAILABLE = False
LIGHTKURVE_AVAILABLE = False

try:
    import scipy
    from scipy import signal, stats, optimize, ndimage
    SCIPY_AVAILABLE = True
    logger.info("SciPy available")
except ImportError:
    logger.warning("SciPy not available - using fallback implementations")
    scipy = None

try:
    import sklearn
    SKLEARN_AVAILABLE = True
    logger.info("Scikit-learn available")
except ImportError:
    logger.warning("Scikit-learn not available - ML features disabled")
    sklearn = None

try:
    import astropy
    ASTROPY_AVAILABLE = True
    logger.info("Astropy available")
except ImportError:
    logger.warning("Astropy not available - astronomy features limited")
    astropy = None

try:
    import lightkurve
    LIGHTKURVE_AVAILABLE = True
    logger.info("Lightkurve available")
except ImportError:
    logger.warning("Lightkurve not available - using demo data only")
    lightkurve = None


def require_scipy(func_name: str = "function"):
    """Decorator to check if SciPy is available"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not SCIPY_AVAILABLE:
                logger.error(f"{func_name} requires SciPy but it's not available")
                raise ImportError(f"SciPy required for {func_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def safe_scipy_import(module_name: str, fallback=None):
    """Safely import SciPy modules with fallback"""
    if not SCIPY_AVAILABLE:
        logger.warning(f"SciPy module {module_name} not available, using fallback")
        return fallback
    
    try:
        if module_name == "signal":
            from scipy import signal
            return signal
        elif module_name == "stats":
            from scipy import stats
            return stats
        elif module_name == "optimize":
            from scipy import optimize
            return optimize
        elif module_name == "ndimage":
            from scipy import ndimage
            return ndimage
        else:
            return getattr(scipy, module_name, fallback)
    except ImportError:
        logger.warning(f"Failed to import scipy.{module_name}")
        return fallback


# Fallback implementations for basic functions
class FallbackStats:
    """Fallback implementations for basic statistical functions"""
    
    @staticmethod
    def skew(data):
        """Simple skewness calculation"""
        import numpy as np
        data = np.asarray(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def kurtosis(data):
        """Simple kurtosis calculation"""
        import numpy as np
        data = np.asarray(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    @staticmethod
    def linregress(x, y):
        """Simple linear regression"""
        import numpy as np
        x, y = np.asarray(x), np.asarray(y)
        n = len(x)
        if n < 2:
            return 0, 0, 0, 1, 0
        
        x_mean, y_mean = np.mean(x), np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        
        # Simple correlation coefficient
        r_value = np.corrcoef(x, y)[0, 1] if n > 1 else 0
        
        return slope, intercept, r_value, 0.05, 0  # p_value, std_err approximated


class FallbackSignal:
    """Fallback implementations for basic signal processing"""
    
    @staticmethod
    def find_peaks(data, height=None, distance=None):
        """Simple peak finding"""
        import numpy as np
        data = np.asarray(data)
        peaks = []
        
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                if height is None or data[i] >= height:
                    peaks.append(i)
        
        return np.array(peaks), {}
    
    @staticmethod
    def periodogram(data, fs=1.0):
        """Simple periodogram using numpy FFT"""
        import numpy as np
        data = np.asarray(data)
        n = len(data)
        
        # Simple FFT-based periodogram
        fft_data = np.fft.fft(data - np.mean(data))
        power = np.abs(fft_data) ** 2 / n
        freqs = np.fft.fftfreq(n, 1/fs)
        
        # Return positive frequencies only
        pos_mask = freqs > 0
        return freqs[pos_mask], power[pos_mask]


# Export safe imports
def get_stats():
    """Get stats module (SciPy or fallback)"""
    return safe_scipy_import("stats", FallbackStats())

def get_signal():
    """Get signal module (SciPy or fallback)"""
    return safe_scipy_import("signal", FallbackSignal())

def get_optimize():
    """Get optimize module (SciPy or fallback)"""
    return safe_scipy_import("optimize", None)

def get_ndimage():
    """Get ndimage module (SciPy or fallback)"""
    return safe_scipy_import("ndimage", None)
