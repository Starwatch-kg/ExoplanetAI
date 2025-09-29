"""
Enhanced BLS (Box Least Squares) Ensemble Algorithm for Exoplanet Detection.

This module implements an optimized BLS algorithm with ensemble methods,
parallel processing, and advanced signal processing techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from scipy import signal
from scipy.optimize import minimize_scalar
import warnings
from functools import lru_cache
import time
from contextlib import contextmanager

# Performance optimizations
warnings.filterwarnings('ignore')


@dataclass
class BLSParameters:
    """BLS algorithm parameters."""
    min_period: float = 0.5
    max_period: float = 50.0
    frequency_factor: float = 1.0
    duration: float = 0.1
    n_bins: int = 200
    min_transit_duration: float = 0.01
    max_transit_duration: float = 0.5
    snr_threshold: float = 7.0
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6


@dataclass
class BLSResult:
    """BLS analysis result."""
    period: float
    duration: float
    depth: float
    snr: float
    significance: float
    power: float
    transit_count: int
    false_positive_prob: float
    parameters: BLSParameters
    processing_time: float
    confidence_score: float


class BLSOptimizer:
    """Optimized BLS algorithm with ensemble methods."""

    def __init__(self, parameters: BLSParameters = None):
        """Initialize BLS optimizer with parameters."""
        self.params = parameters or BLSParameters()
        self._cache = {}

    @contextmanager
    def timer(self):
        """Context manager for timing operations."""
        start_time = time.time()
        yield
        self._last_execution_time = time.time() - start_time

    def preprocess_lightcurve(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess light curve data for BLS analysis.

        Args:
            time: Time array
            flux: Flux array

        Returns:
            Tuple of processed time and flux arrays
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(time) | np.isnan(flux))
        time = time[valid_mask]
        flux = flux[valid_mask]

        # Remove outliers using sigma clipping
        median_flux = np.median(flux)
        std_flux = np.std(flux)
        outlier_mask = np.abs(flux - median_flux) < 3 * std_flux
        time = time[outlier_mask]
        flux = flux[outlier_mask]

        # Normalize flux
        flux = (flux - np.median(flux)) / np.std(flux)

        # Sort by time
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        flux = flux[sort_idx]

        return time, flux

    def generate_frequency_grid(self) -> np.ndarray:
        """Generate optimized frequency grid for BLS."""
        # Use logarithmic spacing for better period coverage
        log_period_min = np.log10(self.params.min_period)
        log_period_max = np.log10(self.params.max_period)

        n_freq = int((log_period_max - log_period_min) * 100 * self.params.frequency_factor)
        log_periods = np.linspace(log_period_min, log_period_max, n_freq)
        periods = 10 ** log_periods
        frequencies = 1.0 / periods

        return frequencies

    def bls_single_frequency(self, time: np.ndarray, flux: np.ndarray,
                           frequency: float) -> Dict[str, float]:
        """
        Compute BLS spectrum for a single frequency.

        Args:
            time: Time array
            flux: Flux array
            frequency: Test frequency

        Returns:
            Dictionary with BLS metrics
        """
        period = 1.0 / frequency

        # Phase fold the data
        phase = (time * frequency) % 1.0
        phase = np.sort(phase)

        # Create transit model
        duration_phase = self.params.duration / period
        in_transit = (phase < duration_phase) | (phase > (1.0 - duration_phase))

        # Calculate BLS power
        n_transit = np.sum(in_transit)
        n_out = len(phase) - n_transit

        if n_transit == 0 or n_out == 0:
            return {
                'power': 0.0,
                'depth': 0.0,
                'snr': 0.0,
                'significance': 0.0
            }

        flux_folded = flux[np.argsort(time * frequency % 1.0)]
        flux_in = np.mean(flux_folded[in_transit])
        flux_out = np.mean(flux_folded[~in_transit])

        depth = flux_out - flux_in
        power = depth ** 2 * n_transit * n_out / (n_transit + n_out)

        # Calculate SNR
        residuals = flux_folded - np.mean(flux_folded)
        noise = np.std(residuals[~in_transit]) if n_out > 0 else np.std(residuals)
        snr = depth / noise if noise > 0 else 0.0

        # Calculate significance using chi-square
        chi2_transit = np.sum((flux_folded[in_transit] - flux_in) ** 2)
        chi2_out = np.sum((flux_folded[~in_transit] - flux_out) ** 2)
        significance = (chi2_out + chi2_transit) / len(flux_folded)

        return {
            'power': power,
            'depth': depth,
            'snr': snr,
            'significance': significance
        }

    def bls_parallel(self, time: np.ndarray, flux: np.ndarray,
                    frequencies: np.ndarray) -> List[Dict[str, float]]:
        """
        Run BLS analysis in parallel for multiple frequencies.

        Args:
            time: Time array
            flux: Flux array
            frequencies: Array of frequencies to test

        Returns:
            List of BLS results for each frequency
        """
        num_workers = min(mp.cpu_count(), len(frequencies))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Split frequencies among workers
            chunk_size = len(frequencies) // num_workers
            chunks = [
                frequencies[i:i + chunk_size]
                for i in range(0, len(frequencies), chunk_size)
            ]

            # Submit jobs
            futures = [
                executor.submit(self._bls_chunk, time, flux, chunk)
                for chunk in chunks
            ]

            # Collect results
            results = []
            for future in as_completed(futures):
                results.extend(future.result())

        return results

    def _bls_chunk(self, time: np.ndarray, flux: np.ndarray,
                  frequencies: np.ndarray) -> List[Dict[str, float]]:
        """Process a chunk of frequencies (for parallel execution)."""
        results = []
        for freq in frequencies:
            result = self.bls_single_frequency(time, flux, freq)
            result['frequency'] = freq
            result['period'] = 1.0 / freq
            results.append(result)
        return results

    def ensemble_bls(self, time: np.ndarray, flux: np.ndarray,
                    n_ensembles: int = 5) -> BLSResult:
        """
        Run BLS ensemble analysis for improved robustness.

        Args:
            time: Time array
            flux: Flux array
            n_ensembles: Number of ensemble iterations

        Returns:
            Best BLS result from ensemble
        """
        with self.timer():
            # Preprocess data
            time_clean, flux_clean = self.preprocess_lightcurve(time, flux)

            if len(time_clean) < 10:
                raise ValueError("Insufficient data points for BLS analysis")

            # Generate frequency grid
            frequencies = self.generate_frequency_grid()

            # Run multiple BLS analyses with different parameters
            ensemble_results = []

            for i in range(n_ensembles):
                # Vary parameters slightly for ensemble diversity
                params_var = BLSParameters(
                    min_period=self.params.min_period * (0.9 + 0.2 * np.random.random()),
                    max_period=self.params.max_period * (0.9 + 0.2 * np.random.random()),
                    duration=self.params.duration * (0.8 + 0.4 * np.random.random()),
                    frequency_factor=self.params.frequency_factor
                )

                # Update parameters temporarily
                old_params = self.params
                self.params = params_var

                try:
                    frequencies_var = self.generate_frequency_grid()
                    results = self.bls_parallel(time_clean, flux_clean, frequencies_var)
                    ensemble_results.extend(results)
                finally:
                    self.params = old_params

            if not ensemble_results:
                raise ValueError("No valid BLS results generated")

            # Find best result
            best_result = max(ensemble_results, key=lambda x: x['power'])

            # Calculate ensemble statistics
            powers = [r['power'] for r in ensemble_results]
            snrs = [r['snr'] for r in ensemble_results]

            # Calculate false positive probability
            mean_power = np.mean(powers)
            std_power = np.std(powers)
            fpp = 1.0 - (best_result['power'] - mean_power) / (3 * std_power) if std_power > 0 else 0.0

            # Create final result
            result = BLSResult(
                period=best_result['period'],
                duration=self.params.duration,
                depth=best_result['depth'],
                snr=best_result['snr'],
                significance=best_result['significance'],
                power=best_result['power'],
                transit_count=int(len(time_clean) * self.params.duration / best_result['period']),
                false_positive_prob=max(0.0, min(1.0, fpp)),
                parameters=self.params,
                processing_time=self._last_execution_time,
                confidence_score=min(100.0, best_result['snr'] * 10)
            )

        return result

    def optimize_transit_parameters(self, time: np.ndarray, flux: np.ndarray,
                                  initial_period: float) -> BLSResult:
        """
        Optimize transit parameters around a detected period.

        Args:
            time: Time array
            flux: Flux array
            initial_period: Initial period estimate

        Returns:
            Optimized BLS result
        """
        def objective(params):
            """Objective function for optimization."""
            period, duration = params

            # Create temporary parameters
            temp_params = BLSParameters(
                min_period=period * 0.5,
                max_period=period * 1.5,
                duration=duration,
                frequency_factor=self.params.frequency_factor
            )

            old_params = self.params
            self.params = temp_params

            try:
                result = self.bls_single_frequency(time, flux, 1.0 / period)
                return -result['power']  # Negative for minimization
            finally:
                self.params = old_params

        # Optimize parameters
        bounds = [
            (initial_period * 0.5, initial_period * 1.5),  # period bounds
            (0.01, 0.5)  # duration bounds
        ]

        result = minimize_scalar(
            objective,
            bounds=bounds,
            method='bounded',
            options={'maxiter': 50}
        )

        if result.success:
            # Run final BLS with optimized parameters
            opt_period, opt_duration = result.x
            temp_params = BLSParameters(
                min_period=opt_period * 0.8,
                max_period=opt_period * 1.2,
                duration=opt_duration,
                frequency_factor=self.params.frequency_factor
            )

            old_params = self.params
            self.params = temp_params

            try:
                frequencies = self.generate_frequency_grid()
                results = self.bls_parallel(time, flux, frequencies)
                best_result = max(results, key=lambda x: x['power'])

                return BLSResult(
                    period=best_result['period'],
                    duration=opt_duration,
                    depth=best_result['depth'],
                    snr=best_result['snr'],
                    significance=best_result['significance'],
                    power=best_result['power'],
                    transit_count=1,
                    false_positive_prob=0.1,
                    parameters=temp_params,
                    processing_time=result.execution_time,
                    confidence_score=min(100.0, best_result['snr'] * 15)
                )
            finally:
                self.params = old_params

        # Fallback to original result
        return self.ensemble_bls(time, flux)


def analyze_transit_with_bls(time: np.ndarray, flux: np.ndarray,
                           parameters: BLSParameters = None) -> BLSResult:
    """
    Main function to analyze transit data using BLS ensemble.

    Args:
        time: Time array in days
        flux: Normalized flux array
        parameters: BLS parameters (optional)

    Returns:
        BLS analysis result
    """
    optimizer = BLSOptimizer(parameters)
    return optimizer.ensemble_bls(time, flux)


def batch_analyze_transits(light_curves: List[Dict[str, np.ndarray]],
                         parameters: BLSParameters = None) -> List[BLSResult]:
    """
    Batch analyze multiple light curves.

    Args:
        light_curves: List of light curve dictionaries with 'time' and 'flux' keys
        parameters: BLS parameters (optional)

    Returns:
        List of BLS results
    """
    optimizer = BLSOptimizer(parameters)
    results = []

    for lc in light_curves:
        try:
            result = optimizer.ensemble_bls(lc['time'], lc['flux'])
            results.append(result)
        except Exception as e:
            print(f"Error analyzing light curve: {e}")
            continue

    return results


# Export for backward compatibility
__all__ = [
    'BLSParameters',
    'BLSResult',
    'BLSOptimizer',
    'analyze_transit_with_bls',
    'batch_analyze_transits'
]
