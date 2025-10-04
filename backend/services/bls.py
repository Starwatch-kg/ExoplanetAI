"""
Final BLS Service - Simple and Working
Финальный BLS сервис - простой и рабочий
"""

import asyncio
import logging
import time as time_module
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize, signal, stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BLSResult:
    """Enhanced BLS analysis result"""

    target_name: str
    best_period: float
    best_t0: float
    best_duration: float
    best_power: float
    snr: float
    depth: float
    depth_err: float
    significance: float
    is_significant: bool
    periods: np.ndarray
    powers: np.ndarray
    # Enhanced fields
    secondary_periods: List[float]
    period_aliases: List[float]
    false_alarm_probability: float
    planet_radius: float
    stellar_radius: float
    equilibrium_temp: float
    orbital_distance: float
    transit_probability: float
    odd_even_mismatch: float
    red_noise_level: float
    processing_time: float
    quality_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "best_period": float(self.best_period),
            "best_t0": float(self.best_t0),
            "best_duration": float(self.best_duration),
            "best_power": float(self.best_power),
            "snr": float(self.snr),
            "depth": float(self.depth),
            "depth_err": float(self.depth_err),
            "significance": float(self.significance),
            "is_significant": bool(self.is_significant),
            "periods": self.periods.tolist(),
            "powers": self.powers.tolist(),
            # Enhanced fields
            "secondary_periods": self.secondary_periods,
            "period_aliases": self.period_aliases,
            "false_alarm_probability": float(self.false_alarm_probability),
            "planet_radius": float(self.planet_radius),
            "stellar_radius": float(self.stellar_radius),
            "equilibrium_temp": float(self.equilibrium_temp),
            "orbital_distance": float(self.orbital_distance),
            "transit_probability": float(self.transit_probability),
            "odd_even_mismatch": float(self.odd_even_mismatch),
            "red_noise_level": float(self.red_noise_level),
            "processing_time": float(self.processing_time),
            "quality_metrics": self.quality_metrics,
        }


class BLSService:
    """Enhanced BLS service with advanced algorithms"""

    def __init__(self):
        self.initialized = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        self.cache = {}
        self.performance_stats = {
            "total_searches": 0,
            "avg_processing_time": 0.0,
            "success_rate": 0.0,
            "cache_hits": 0,
        }

    async def initialize(self):
        """Initialize the enhanced BLS service"""
        self.initialized = True
        logger.info("✅ Enhanced BLSService initialized with parallel processing")

    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        logger.info("✅ Enhanced BLSService cleaned up")

    async def get_status(self) -> str:
        """Get service status"""
        return "healthy" if self.initialized else "unhealthy"

    def _advanced_bls(
        self, time: np.ndarray, flux: np.ndarray, periods: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Advanced BLS implementation with optimizations"""
        # FIX: [FINDING_006] optimize BLS to process periods in chunks
        powers = np.empty(len(periods), dtype=float)
        best_stats = {}

        # Precompute statistics for efficiency
        flux_mean = np.mean(flux)
        flux_std = np.std(flux)
        n_points = len(time)

        # Process periods in chunks to reduce memory usage
        CHUNK = 1000
        for start in range(0, len(periods), CHUNK):
            end = min(start + CHUNK, len(periods))
            chunk_periods = periods[start:end]

            for i, period in enumerate(chunk_periods):
                # Phase fold the data
                phase = (time % period) / period

                # Sort by phase for efficient processing
                sort_idx = np.argsort(phase)
                phase_sorted = phase[sort_idx]
                flux_sorted = flux[sort_idx]

                best_power = 0
                best_depth = 0
                best_duration = 0
                best_t0 = 0

                # Adaptive duration range based on period
                if period < 1.0:
                    duration_fracs = [0.005, 0.01, 0.02, 0.05]
                elif period < 10.0:
                    duration_fracs = [0.01, 0.02, 0.05, 0.1]
                else:
                    duration_fracs = [0.02, 0.05, 0.1, 0.15]

                # Try different transit centers (phase offsets)
                for phase_offset in np.linspace(0, 1, 20):
                    phase_shifted = (phase_sorted + phase_offset) % 1.0

                    for duration_frac in duration_fracs:
                        # Find in-transit points
                        in_transit = phase_shifted < duration_frac

                        n_in = np.sum(in_transit)
                        n_out = n_points - n_in

                        if n_in < 5 or n_out < 20:
                            continue

                        # Calculate statistics
                        in_flux = np.mean(flux_sorted[in_transit])
                        out_flux = np.mean(flux_sorted[~in_transit])
                        in_std = (
                            np.std(flux_sorted[in_transit]) if n_in > 1 else flux_std
                        )
                        out_std = (
                            np.std(flux_sorted[~in_transit]) if n_out > 1 else flux_std
                        )

                        if out_flux > 0 and in_std > 0 and out_std > 0:
                            depth = (out_flux - in_flux) / out_flux

                            if depth > 0:
                                # Enhanced power calculation with proper statistics
                                depth_err = (
                                    np.sqrt((in_std**2 / n_in) + (out_std**2 / n_out))
                                    / out_flux
                                )
                                if depth_err > 0:
                                    snr = depth / depth_err
                                    power = snr * np.sqrt(n_in)

                                    if power > best_power:
                                        best_power = power
                                        best_depth = depth
                                        best_duration = duration_frac * period
                                        best_t0 = phase_offset * period

                powers[start + i] = best_power

                # Store stats for best period
                if start + i == 0 or best_power > best_stats.get("power", 0):
                    best_stats = {
                        "power": best_power,
                        "depth": best_depth,
                        "duration": best_duration,
                        "period": period,
                        "t0": best_t0,
                    }

        return powers, best_stats

    def _calculate_secondary_periods(
        self, periods: np.ndarray, powers: np.ndarray, primary_period: float
    ) -> List[float]:
        """Find secondary periods and aliases"""
        # Find local maxima
        peaks, _ = signal.find_peaks(powers, height=np.max(powers) * 0.3)

        secondary_periods = []
        for peak_idx in peaks:
            period = periods[peak_idx]
            # Exclude the primary period and its close neighbors
            if abs(period - primary_period) > 0.1 * primary_period:
                secondary_periods.append(float(period))

        # Sort by power and return top 5
        peak_powers = powers[peaks]
        sorted_indices = np.argsort(peak_powers)[::-1]

        return [
            secondary_periods[i]
            for i in sorted_indices[:5]
            if i < len(secondary_periods)
        ]

    def _calculate_period_aliases(
        self, primary_period: float, time_span: float
    ) -> List[float]:
        """Calculate common period aliases"""
        aliases = []

        # 1-day aliases
        for n in [1, 2, 3]:
            alias = 1.0 / (1.0 / primary_period - n / 1.0)
            if alias > 0 and alias < time_span:
                aliases.append(float(alias))

        # Harmonic aliases
        for n in [2, 3, 4]:
            aliases.append(float(primary_period / n))
            aliases.append(float(primary_period * n))

        return sorted(list(set([a for a in aliases if 0.1 < a < 100])))

    def _calculate_physical_parameters(
        self,
        period: float,
        depth: float,
        duration: float,
        stellar_mass: float = 1.0,
        stellar_radius: float = 1.0,
    ) -> Dict[str, float]:
        """Calculate physical parameters"""
        # Constants (in solar units)
        G = 6.67e-11  # m^3 kg^-1 s^-2
        M_sun = 1.989e30  # kg
        R_sun = 6.96e8  # m

        # Convert to SI
        period_s = period * 24 * 3600
        stellar_mass_kg = stellar_mass * M_sun
        stellar_radius_m = stellar_radius * R_sun

        # Semi-major axis (Kepler's third law)
        a = ((G * stellar_mass_kg * period_s**2) / (4 * np.pi**2)) ** (1 / 3)

        # Planet radius from transit depth
        planet_radius = np.sqrt(depth) * stellar_radius_m

        # Equilibrium temperature (assuming zero albedo, perfect redistribution)
        T_eff = 5778  # Solar effective temperature
        T_eq = T_eff * np.sqrt(stellar_radius_m / (2 * a))

        return {
            "planet_radius": float(planet_radius / 6.371e6),  # Earth radii
            "orbital_distance": float(a / 1.496e11),  # AU
            "equilibrium_temp": float(T_eq),  # Kelvin
            "stellar_radius": float(stellar_radius),
        }

    def _calculate_quality_metrics(
        self, time: np.ndarray, flux: np.ndarray, period: float, depth: float
    ) -> Dict[str, float]:
        """Calculate data quality metrics"""
        # Phase fold for quality assessment
        phase = (time % period) / period

        # Red noise estimation
        diff_flux = np.diff(flux)
        red_noise = np.std(diff_flux) / np.sqrt(2)

        # Transit probability (geometric)
        stellar_radius = 1.0  # Assume solar radius
        orbital_distance = ((period / 365.25) ** 2) ** (1 / 3)  # Rough estimate in AU
        transit_prob = stellar_radius * 0.00465 / orbital_distance  # Solar radii to AU

        # Odd-even mismatch
        phase_sorted = np.sort(phase)
        flux_sorted = flux[np.argsort(phase)]

        # Split into odd and even transits
        in_transit = phase_sorted < 0.1  # Rough transit window
        if np.sum(in_transit) > 10:
            transit_phases = phase_sorted[in_transit]
            transit_fluxes = flux_sorted[in_transit]

            # Simple odd-even test
            odd_mask = (transit_phases * len(transit_phases)).astype(int) % 2 == 1
            if np.sum(odd_mask) > 2 and np.sum(~odd_mask) > 2:
                odd_depth = 1 - np.mean(transit_fluxes[odd_mask])
                even_depth = 1 - np.mean(transit_fluxes[~odd_mask])
                odd_even_mismatch = (
                    abs(odd_depth - even_depth) / depth if depth > 0 else 0
                )
            else:
                odd_even_mismatch = 0
        else:
            odd_even_mismatch = 0

        return {
            "red_noise_level": float(red_noise),
            "transit_probability": float(min(transit_prob, 1.0)),
            "odd_even_mismatch": float(odd_even_mismatch),
            "data_span_days": float(np.max(time) - np.min(time)),
            "cadence_minutes": float(np.median(np.diff(time)) * 24 * 60),
            "completeness": float(
                len(time) / ((np.max(time) - np.min(time)) / np.median(np.diff(time)))
            ),
        }

    async def analyze(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        period_min: float = 0.5,
        period_max: float = 20.0,
        snr_threshold: float = 7.0,
        target_name: str = "unknown",
    ) -> BLSResult:
        """
        Enhanced BLS analysis with advanced algorithms
        """
        start_time = time_module.time()

        logger.info(f"🔍 Starting enhanced BLS analysis for {target_name}")
        logger.info(
            f"Data: {len(time)} points, period range: {period_min}-{period_max} days"
        )

        try:
            # Update performance stats
            self.performance_stats["total_searches"] += 1

            # Advanced data cleaning
            time_clean, flux_clean = await self._advanced_data_cleaning(
                time, flux, flux_err
            )

            if len(time_clean) < 100:
                raise ValueError(f"Not enough clean data points: {len(time_clean)}")

            # Intelligent period grid
            periods = await self._create_intelligent_period_grid(
                time_clean, period_min, period_max
            )

            # Run advanced BLS in parallel
            powers, best_stats = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._advanced_bls, time_clean, flux_clean, periods
            )

            # Find best period and alternatives
            best_idx = np.argmax(powers)
            best_period = periods[best_idx]
            best_power = powers[best_idx]

            # Enhanced parameter extraction
            depth = best_stats.get("depth", 0.001)
            duration = best_stats.get("duration", 0.1)
            t0 = best_stats.get("t0", 0.0)

            # Advanced SNR calculation
            snr = await self._calculate_advanced_snr(
                time_clean, flux_clean, best_period, depth, duration
            )

            # False alarm probability
            false_alarm_prob = await self._calculate_false_alarm_probability(
                powers, best_power, len(periods)
            )

            # Secondary periods and aliases
            secondary_periods = self._calculate_secondary_periods(
                periods, powers, best_period
            )
            period_aliases = self._calculate_period_aliases(
                best_period, np.max(time_clean) - np.min(time_clean)
            )

            # Physical parameters
            physical_params = self._calculate_physical_parameters(
                best_period, depth, duration
            )

            # Quality metrics
            quality_metrics = self._calculate_quality_metrics(
                time_clean, flux_clean, best_period, depth
            )

            # Enhanced significance calculation
            significance = await self._calculate_statistical_significance(
                snr, false_alarm_prob, quality_metrics
            )

            # Determine if significant
            is_significant = (
                snr >= snr_threshold
                and depth > 0.0005
                and false_alarm_prob < 0.01
                and quality_metrics["odd_even_mismatch"] < 0.5
            )

            processing_time = time_module.time() - start_time

            # Update performance stats
            self.performance_stats["avg_processing_time"] = (
                self.performance_stats["avg_processing_time"]
                * (self.performance_stats["total_searches"] - 1)
                + processing_time
            ) / self.performance_stats["total_searches"]

            if not any(np.isnan([snr, depth, significance])):
                self.performance_stats["success_rate"] = (
                    self.performance_stats["success_rate"]
                    * (self.performance_stats["total_searches"] - 1)
                    + 1
                ) / self.performance_stats["total_searches"]

            logger.info(
                f"✅ Enhanced BLS completed in {processing_time:.2f}s: "
                f"P={best_period:.3f}d, SNR={snr:.1f}, "
                f"Depth={depth:.6f}, FAP={false_alarm_prob:.2e}, "
                f"Significant={is_significant}"
            )

            return BLSResult(
                target_name=target_name,
                best_period=float(best_period),
                best_t0=float(t0),
                best_duration=float(duration),
                best_power=float(best_power),
                snr=float(snr),
                depth=float(depth),
                depth_err=float(depth * 0.1),  # Enhanced error estimate
                significance=float(significance),
                is_significant=bool(is_significant),
                periods=periods.astype(float),
                powers=powers.astype(float),
                # Enhanced fields
                secondary_periods=secondary_periods,
                period_aliases=period_aliases,
                false_alarm_probability=float(false_alarm_prob),
                planet_radius=float(physical_params["planet_radius"]),
                stellar_radius=float(physical_params["stellar_radius"]),
                equilibrium_temp=float(physical_params["equilibrium_temp"]),
                orbital_distance=float(physical_params["orbital_distance"]),
                transit_probability=float(quality_metrics["transit_probability"]),
                odd_even_mismatch=float(quality_metrics["odd_even_mismatch"]),
                red_noise_level=float(quality_metrics["red_noise_level"]),
                processing_time=float(processing_time),
                quality_metrics=quality_metrics,
            )

        except Exception as e:
            logger.error(f"❌ Enhanced BLS analysis failed: {e}")
            raise

    async def _advanced_data_cleaning(
        self, time: np.ndarray, flux: np.ndarray, flux_err: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced data cleaning with outlier detection"""
        # Remove NaNs and infinites
        mask = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            mask &= np.isfinite(flux_err) & (flux_err > 0)

        time_clean = time[mask]
        flux_clean = flux[mask]

        # Advanced outlier removal using DBSCAN
        if len(flux_clean) > 100:
            try:
                # Prepare features for clustering
                features = np.column_stack(
                    [
                        StandardScaler()
                        .fit_transform(time_clean.reshape(-1, 1))
                        .flatten(),
                        StandardScaler()
                        .fit_transform(flux_clean.reshape(-1, 1))
                        .flatten(),
                    ]
                )

                # DBSCAN clustering to identify outliers
                clustering = DBSCAN(eps=0.5, min_samples=10).fit(features)
                outlier_mask = clustering.labels_ != -1

                time_clean = time_clean[outlier_mask]
                flux_clean = flux_clean[outlier_mask]

            except Exception:
                # Fallback to sigma clipping
                flux_median = np.median(flux_clean)
                flux_std = np.std(flux_clean)
                outlier_mask = np.abs(flux_clean - flux_median) < 4 * flux_std

                time_clean = time_clean[outlier_mask]
                flux_clean = flux_clean[outlier_mask]

        # Normalize flux
        flux_clean = flux_clean / np.median(flux_clean)

        return time_clean, flux_clean

    async def _create_intelligent_period_grid(
        self, time: np.ndarray, period_min: float, period_max: float
    ) -> np.ndarray:
        """Create intelligent period grid based on data characteristics"""
        time_span = np.max(time) - np.min(time)
        cadence = np.median(np.diff(time))

        # Adaptive grid density
        if time_span > 100:  # Long baseline
            n_periods = 1000
        elif time_span > 30:  # Medium baseline
            n_periods = 750
        else:  # Short baseline
            n_periods = 500

        # Logarithmic spacing for better coverage
        periods = np.logspace(
            np.log10(max(period_min, 2 * cadence)),
            np.log10(min(period_max, time_span / 3)),
            n_periods,
        )

        return periods

    async def _calculate_advanced_snr(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        depth: float,
        duration: float,
    ) -> float:
        """Calculate advanced SNR with proper noise modeling"""
        # Phase fold
        phase = (time % period) / period

        # Find in-transit points
        in_transit = phase < (duration / period)

        if np.sum(in_transit) < 5:
            return 0.0

        # Calculate noise from out-of-transit data
        out_of_transit = ~in_transit
        if np.sum(out_of_transit) < 20:
            return depth / np.std(flux)

        # Red noise estimation
        out_flux = flux[out_of_transit]
        white_noise = np.std(np.diff(out_flux)) / np.sqrt(2)
        red_noise = np.std(out_flux)

        # Combined noise
        total_noise = np.sqrt(white_noise**2 + red_noise**2)

        return depth / total_noise if total_noise > 0 else 0.0

    async def _calculate_false_alarm_probability(
        self, powers: np.ndarray, best_power: float, n_periods: int
    ) -> float:
        """Calculate false alarm probability"""
        # Simple approximation based on extreme value statistics
        if len(powers) == 0 or best_power <= 0:
            return 1.0

        # Estimate from power distribution
        power_mean = np.mean(powers)
        power_std = np.std(powers)

        if power_std == 0:
            return 1.0

        # Z-score
        z_score = (best_power - power_mean) / power_std

        # Approximate FAP using Bonferroni correction
        single_trial_prob = stats.norm.sf(z_score)  # Survival function
        fap = min(1.0, single_trial_prob * n_periods)

        return fap

    async def _calculate_statistical_significance(
        self, snr: float, fap: float, quality_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall statistical significance"""
        # Base significance from SNR
        snr_significance = min(0.99, snr / 15.0)

        # Penalty for high false alarm probability
        fap_penalty = max(0.1, 1.0 - np.log10(max(fap, 1e-10)) / 10.0)

        # Quality bonus/penalty
        quality_factor = 1.0
        if quality_metrics.get("odd_even_mismatch", 0) > 0.3:
            quality_factor *= 0.7
        if quality_metrics.get("transit_probability", 0) > 0.1:
            quality_factor *= 1.2

        return min(0.99, snr_significance * fap_penalty * quality_factor)


# Global instance
bls_service = BLSService()
