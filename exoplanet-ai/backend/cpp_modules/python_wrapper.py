"""
Python wrapper for C++ acceleration modules
Provides Python interface to high-performance C++ implementations
"""

import ctypes
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import subprocess
import os
import sys

logger = logging.getLogger(__name__)

def _get_lib_extension() -> str:
    # FIX: [FINDING_001] Определяем расширение в зависимости от платформы
    if sys.platform.startswith("win"):
        return ".dll"
    if sys.platform.startswith("darwin"):
        return ".dylib"
    return ".so"

def load_cpp_library(cpp_dir: Path, lib_basename: str):
    # FIX: [FINDING_001] Безопасная загрузка C++ библиотеки
    lib_ext = _get_lib_extension()
    lib_path = (Path(cpp_dir) / f"{lib_basename}{lib_ext}").resolve()
    allowed_dir = Path(__file__).parent.resolve()
    if allowed_dir not in lib_path.parents:
        raise FileNotFoundError(f"Library {lib_path} is outside allowed directory")
    if not lib_path.exists():
        raise FileNotFoundError(f"Library not found: {lib_path}")
    # Используем ctypes для динамической загрузки
    try:
        return ctypes.CDLL(str(lib_path))
    except OSError as e:
        logger.exception("Failed to load native lib")
        raise

class CPPModuleManager:
    """Manager for C++ acceleration modules"""
    
    def __init__(self):
        self.gpi_generator = None
        self.search_accelerator = None
        self.modules_loaded = False
        self._compile_modules()
        self._load_modules()
    
    def _compile_modules(self):
        """Compile C++ modules if needed"""
        cpp_dir = Path(__file__).parent
        
        # Check if compiled libraries exist
        gpi_lib = cpp_dir / f"gpi_generator{_get_lib_extension()}"
        search_lib = cpp_dir / f"search_accelerator{_get_lib_extension()}"
        
        if not gpi_lib.exists() or not search_lib.exists():
            logger.info("C++ modules not found, checking for compiler...")
            try:
                # Use safe compile function
                safe_compile_cpp(
                    source_path=cpp_dir / "gpi_data_generator.cpp",
                    output_path=gpi_lib,
                    allowed_dir=cpp_dir
                )
                
                # Compile search accelerator (requires FFTW)
                safe_compile_cpp(
                    source_path=cpp_dir / "search_accelerator.cpp",
                    output_path=search_lib,
                    allowed_dir=cpp_dir,
                    extra_flags=["-lfftw3", "-lfftw3_threads", "-lm"]
                )
                
                logger.info("✅ C++ modules compiled successfully")
                
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                logger.warning(f"⚠️ C++ compilation failed: {e}")
                logger.info("Falling back to Python implementations")
                logger.info("To enable C++ acceleration, install g++ and FFTW library")
    
    def _load_modules(self):
        """Load compiled C++ modules"""
        cpp_dir = Path(__file__).parent
        
        try:
            # Load GPI generator
            try:
                self.gpi_lib = load_cpp_library(cpp_dir, "gpi_generator")
                self._setup_gpi_interface()
            except FileNotFoundError:
                logger.warning("GPI generator library not found, using Python fallback")
            
            # Load search accelerator
            try:
                self.search_lib = load_cpp_library(cpp_dir, "search_accelerator")
                self._setup_search_interface()
            except FileNotFoundError:
                logger.warning("Search accelerator library not found, using Python fallback")
            
            self.modules_loaded = True
            logger.info("✅ C++ modules loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load C++ modules: {e}")
            logger.info("Using Python fallback implementations")
    
    def _setup_gpi_interface(self):
        """Setup GPI generator C interface"""
        if not hasattr(self, 'gpi_lib') or self.gpi_lib is None:
            return
            
        try:
            # Function signatures
            self.gpi_lib.create_gpi_generator.argtypes = [ctypes.c_uint]
            self.gpi_lib.create_gpi_generator.restype = ctypes.c_void_p
            
            self.gpi_lib.destroy_gpi_generator.argtypes = [ctypes.c_void_p]
            self.gpi_lib.destroy_gpi_generator.restype = None
            
            self.gpi_lib.generate_gpi_data.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
            self.gpi_lib.generate_gpi_data.restype = ctypes.c_char_p
            
            self.gpi_lib.free_string.argtypes = [ctypes.c_char_p]
            self.gpi_lib.free_string.restype = None
        except Exception as e:
            logger.warning(f"Failed to setup GPI interface: {e}")
            self.gpi_lib = None
    
    def _setup_search_interface(self):
        """Setup search accelerator C interface"""
        if not hasattr(self, 'search_lib') or self.search_lib is None:
            return
            
        try:
            # Function signatures
            self.search_lib.create_search_accelerator.argtypes = [ctypes.c_int, ctypes.c_int]
            self.search_lib.create_search_accelerator.restype = ctypes.c_void_p
            
            self.search_lib.destroy_search_accelerator.argtypes = [ctypes.c_void_p]
            self.search_lib.destroy_search_accelerator.restype = None
            
            # BLS search
            self.search_lib.accelerated_bls_search.argtypes = [
                ctypes.c_void_p,  # accelerator
                ctypes.POINTER(ctypes.c_double),  # time
                ctypes.POINTER(ctypes.c_double),  # flux
                ctypes.POINTER(ctypes.c_double),  # flux_err
                ctypes.c_int,  # n_data
                ctypes.c_double,  # period_min
                ctypes.c_double,  # period_max
                ctypes.c_int,  # period_samples
                ctypes.POINTER(ctypes.c_double),  # result_period
                ctypes.POINTER(ctypes.c_double),  # result_snr
                ctypes.POINTER(ctypes.c_double)   # result_power
            ]
            
            # GPI analysis
            self.search_lib.accelerated_gpi_analysis.argtypes = [
                ctypes.c_void_p,  # accelerator
                ctypes.POINTER(ctypes.c_double),  # time
                ctypes.POINTER(ctypes.c_double),  # flux
                ctypes.POINTER(ctypes.c_double),  # flux_err
                ctypes.c_int,  # n_data
                ctypes.c_double,  # phase_sensitivity
                ctypes.POINTER(ctypes.c_double),  # result_confidence
                ctypes.POINTER(ctypes.c_double),  # result_period
                ctypes.POINTER(ctypes.c_double)   # result_snr
            ]
        except Exception as e:
            logger.warning(f"Failed to setup search interface: {e}")
            self.search_lib = None

def safe_compile_cpp(source_path: Path, output_path: Path, allowed_dir: Path, extra_flags: List[str] = None, timeout: int = 300):
    # FIX: [FINDING_002] Безопасная компиляция C++ через g++
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    allowed_dir = Path(allowed_dir).resolve()
    if allowed_dir not in source_path.parents:
        raise ValueError("Source path is outside allowed directory")
    
    # Build command with safe parameters
    cmd = ["g++", "-shared", "-fPIC", "-O3", "-fopenmp"]
    cmd.append(str(source_path))
    cmd.extend(["-o", str(output_path)])
    if extra_flags:
        cmd.extend(extra_flags)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, cwd=str(allowed_dir), timeout=timeout)
        logger.info("Compile stdout: %s", result.stdout.decode('utf-8', errors='ignore'))
    except subprocess.CalledProcessError as e:
        logger.error("Compile failed: %s", e.stderr.decode('utf-8', errors='ignore'))
        raise

class GPIDataGenerator:
    """High-performance GPI data generator"""
    
    def __init__(self, cpp_manager: CPPModuleManager):
        self.cpp_manager = cpp_manager
        self.generator_handle = None
        
        if cpp_manager.modules_loaded and hasattr(cpp_manager, 'gpi_lib'):
            self.generator_handle = cpp_manager.gpi_lib.create_gpi_generator(0)
    
    def generate_synthetic_data(self, num_points: int = 10000, observation_time: float = 365.0) -> Dict[str, Any]:
        """Generate synthetic GPI data"""
        
        if self.generator_handle and self.cpp_manager.modules_loaded:
            # Use C++ implementation
            try:
                result_ptr = self.cpp_manager.gpi_lib.generate_gpi_data(
                    self.generator_handle, num_points, observation_time
                )
                
                if result_ptr:
                    json_str = ctypes.string_at(result_ptr).decode('utf-8')
                    self.cpp_manager.gpi_lib.free_string(result_ptr)
                    return json.loads(json_str)
                
            except Exception as e:
                logger.warning(f"C++ GPI generation failed: {e}, falling back to Python")
        
        # Python fallback implementation
        return self._generate_python_fallback(num_points, observation_time)
    
    def _generate_python_fallback(self, num_points: int, observation_time: float) -> Dict[str, Any]:
        """Python fallback for GPI data generation"""
        
        # Generate realistic planetary system
        np.random.seed()
        
        system_params = {
            "star_mass": 0.5 + np.random.random() * 1.5,
            "planet_mass": 0.1 + np.random.random() * 20.0,
            "orbital_period": 1.0 + np.random.random() * 999.0,
            "distance_to_system": 10.0 + np.random.random() * 490.0
        }
        
        # Generate time series
        time_points = np.linspace(0, observation_time, num_points)
        
        # Calculate phase shifts (simplified model)
        orbital_phase = 2 * np.pi * time_points / system_params["orbital_period"]
        
        # Gravitational phase shift amplitude
        phase_amplitude = 1e-12 * system_params["planet_mass"] / (system_params["distance_to_system"]**2)
        
        # Generate phase shifts with orbital modulation
        phase_shifts = phase_amplitude * np.sin(orbital_phase)
        
        # Add noise
        noise_level = 1e-15 * system_params["distance_to_system"] / system_params["planet_mass"]
        noise = np.random.normal(0, noise_level, num_points)
        phase_shifts += noise
        
        # Calculate amplitude variations (gravitational lensing)
        amplitude_variations = 1e-6 * system_params["planet_mass"] * np.cos(orbital_phase) / (system_params["distance_to_system"]**2)
        
        # Calculate detection metrics
        snr = np.std(phase_shifts) / noise_level
        confidence = np.tanh(snr * np.log10(system_params["planet_mass"]) / 10.0)
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "system_parameters": system_params,
            "detection_metrics": {
                "snr": float(snr),
                "confidence": float(confidence),
                "data_points": num_points
            },
            "time_series": {
                "time": time_points.tolist(),
                "phase_shifts": phase_shifts.tolist(),
                "amplitude_variations": amplitude_variations.tolist()
            }
        }
    
    def __del__(self):
        if self.generator_handle and self.cpp_manager.modules_loaded:
            self.cpp_manager.gpi_lib.destroy_gpi_generator(self.generator_handle)

class SearchAccelerator:
    """High-performance search algorithms"""
    
    def __init__(self, cpp_manager: CPPModuleManager, max_fft_size: int = 65536, threads: int = 0):
        self.cpp_manager = cpp_manager
        self.accelerator_handle = None
        
        if cpp_manager.modules_loaded and hasattr(cpp_manager, 'search_lib'):
            self.accelerator_handle = cpp_manager.search_lib.create_search_accelerator(max_fft_size, threads)
    
    def accelerated_bls(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                       period_min: float = 0.5, period_max: float = 50.0, 
                       period_samples: int = 10000) -> Dict[str, Any]:
        """Accelerated BLS search"""
        
        if self.accelerator_handle and self.cpp_manager.modules_loaded:
            try:
                # Prepare arrays
                time_array = (ctypes.c_double * len(time))(*time)
                flux_array = (ctypes.c_double * len(flux))(*flux)
                flux_err_array = (ctypes.c_double * len(flux_err))(*flux_err)
                
                # Result variables
                result_period = ctypes.c_double()
                result_snr = ctypes.c_double()
                result_power = ctypes.c_double()
                
                # Call C++ function
                self.cpp_manager.search_lib.accelerated_bls_search(
                    self.accelerator_handle,
                    time_array, flux_array, flux_err_array, len(time),
                    period_min, period_max, period_samples,
                    ctypes.byref(result_period),
                    ctypes.byref(result_snr),
                    ctypes.byref(result_power)
                )
                
                return {
                    "period": result_period.value,
                    "snr": result_snr.value,
                    "power": result_power.value,
                    "method": "cpp_accelerated"
                }
                
            except Exception as e:
                logger.warning(f"C++ BLS failed: {e}, falling back to Python")
        
        # Python fallback
        return self._bls_python_fallback(time, flux, flux_err, period_min, period_max)
    
    def accelerated_gpi(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                       phase_sensitivity: float = 1e-12) -> Dict[str, Any]:
        """Accelerated GPI analysis"""
        
        if self.accelerator_handle and self.cpp_manager.modules_loaded:
            try:
                # Prepare arrays
                time_array = (ctypes.c_double * len(time))(*time)
                flux_array = (ctypes.c_double * len(flux))(*flux)
                flux_err_array = (ctypes.c_double * len(flux_err))(*flux_err)
                
                # Result variables
                result_confidence = ctypes.c_double()
                result_period = ctypes.c_double()
                result_snr = ctypes.c_double()
                
                # Call C++ function
                self.cpp_manager.search_lib.accelerated_gpi_analysis(
                    self.accelerator_handle,
                    time_array, flux_array, flux_err_array, len(time),
                    phase_sensitivity,
                    ctypes.byref(result_confidence),
                    ctypes.byref(result_period),
                    ctypes.byref(result_snr)
                )
                
                return {
                    "detection_confidence": result_confidence.value,
                    "orbital_period": result_period.value,
                    "snr": result_snr.value,
                    "method": "cpp_accelerated"
                }
                
            except Exception as e:
                logger.warning(f"C++ GPI failed: {e}, falling back to Python")
        
        # Python fallback
        return self._gpi_python_fallback(time, flux, flux_err, phase_sensitivity)
    
    def _bls_python_fallback(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                           period_min: float, period_max: float) -> Dict[str, Any]:
        """Python fallback BLS implementation"""
        
        # Simple BLS implementation
        periods = np.logspace(np.log10(period_min), np.log10(period_max), 1000)
        powers = []
        
        flux_normalized = (flux - np.mean(flux)) / np.mean(flux)
        
        for period in periods:
            # Phase fold
            phases = (time % period) / period
            
            # Sort by phase
            sort_idx = np.argsort(phases)
            sorted_phases = phases[sort_idx]
            sorted_flux = flux_normalized[sort_idx]
            
            # Simple transit search
            best_power = 0
            for transit_width in [0.01, 0.02, 0.05]:
                for center in np.linspace(0, 1, 20):
                    # Define transit window
                    in_transit = np.abs(sorted_phases - center) < transit_width/2
                    
                    if np.sum(in_transit) > 5 and np.sum(~in_transit) > 5:
                        in_transit_flux = sorted_flux[in_transit]
                        out_transit_flux = sorted_flux[~in_transit]
                        
                        depth = np.mean(out_transit_flux) - np.mean(in_transit_flux)
                        power = depth**2 * len(in_transit_flux) * len(out_transit_flux) / len(sorted_flux)
                        
                        if power > best_power:
                            best_power = power
            
            powers.append(best_power)
        
        powers = np.array(powers)
        best_idx = np.argmax(powers)
        best_period = periods[best_idx]
        best_power = powers[best_idx]
        
        # Calculate SNR
        snr = (best_power - np.mean(powers)) / np.std(powers)
        
        return {
            "period": float(best_period),
            "snr": float(snr),
            "power": float(best_power),
            "method": "python_fallback"
        }
    
    def _gpi_python_fallback(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                           phase_sensitivity: float) -> Dict[str, Any]:
        """Python fallback GPI implementation"""
        
        # Calculate phase shifts (simplified)
        flux_normalized = (flux - np.mean(flux)) / np.mean(flux)
        dt = np.diff(time)
        dflux = np.diff(flux_normalized)
        
        # Phase shifts
        phase_shifts = phase_sensitivity * dflux[:-1] / dt[:-1]
        
        # Estimate orbital period using FFT
        fft_result = np.fft.fft(phase_shifts)
        freqs = np.fft.fftfreq(len(phase_shifts), np.mean(dt))
        
        # Find dominant frequency
        power_spectrum = np.abs(fft_result)**2
        max_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_freq = freqs[max_freq_idx]
        
        if dominant_freq > 0:
            orbital_period = 1.0 / dominant_freq
        else:
            orbital_period = 10.0  # Default
        
        # Calculate confidence
        snr = np.std(phase_shifts) / np.mean(flux_err)
        confidence = np.tanh(snr / 5.0)
        
        return {
            "detection_confidence": float(max(0.0, min(1.0, confidence))),
            "orbital_period": float(max(0.1, min(1000.0, orbital_period))),
            "snr": float(snr),
            "method": "python_fallback"
        }
    
    def __del__(self):
        if self.accelerator_handle and self.cpp_manager.modules_loaded:
            self.cpp_manager.search_lib.destroy_search_accelerator(self.accelerator_handle)

# Global instances
cpp_manager = CPPModuleManager()
gpi_generator = GPIDataGenerator(cpp_manager)
search_accelerator = SearchAccelerator(cpp_manager)

def get_gpi_generator() -> GPIDataGenerator:
    """Get global GPI generator instance"""
    return gpi_generator

def get_search_accelerator() -> SearchAccelerator:
    """Get global search accelerator instance"""
    return search_accelerator
