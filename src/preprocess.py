# src/preprocess.py
#
# Author: Senior ML Engineer (AI Assistant)
# Date: 13.09.2025
#
# Description:
# This module contains the core data preprocessing and feature extraction pipeline for the 
# Helios Exoplanet Search project. It is designed to handle raw light curves (from FITS or CSV),
# perform robust detrending using Gaussian Processes, and run a cached, parallelized
# Box-Least Squares (BLS) search to extract initial transit parameters.

import logging
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from joblib import Memory, Parallel, delayed
from scipy.interpolate import interp1d

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)

# Setup a cache for expensive computations like BLS
# This will store results on disk and prevent re-computation
CACHE_DIR = "./.cache"
memory = Memory(CACHE_DIR, verbose=0)

# --- 1. Data Loading & Initial Cleaning ---

def load_light_curve(filepath: str) -> pd.DataFrame:
    """
    Loads light curve data from either a FITS file or a CSV file.
    
    Args:
        filepath (str): Path to the data file.

    Returns:
        pd.DataFrame: A DataFrame with 'time' and 'flux' columns.
    """
    logging.info(f"Loading light curve from: {filepath}")
    if filepath.endswith(".fits"):
        with fits.open(filepath, mode="readonly") as hdul:
            data = hdul[1].data
            time = data['TIME']
            flux = data['PDCSAP_FLUX'] # Pre-search Data Conditioning Simple Aperture Photometry
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        time = df['time'].values
        flux = df['flux'].values
    else:
        raise ValueError("Unsupported file format. Please use .fits or .csv")

    # Basic cleaning: remove NaNs and infinite values
    mask = np.isfinite(time) & np.isfinite(flux)
    return pd.DataFrame({'time': time[mask], 'flux': flux[mask]})

# --- 2. Advanced Detrending with Gaussian Processes ---

def gp_detrend(time: np.ndarray, flux: np.ndarray, kernel=None):
    """
    Performs robust detrending of a light curve using a Gaussian Process Regressor.
    This is superior to simple polynomial or spline fitting as it models the
    stochastic nature of stellar variability.

    Args:
        time (np.ndarray): Time values of the light curve.
        flux (np.ndarray): Flux values.
        kernel: A scikit-learn GP kernel. If None, a default Matern kernel is used.

    Returns:
        np.ndarray: The detrended (normalized) flux.
    """
    logging.info("Performing Gaussian Process detrending...")
    
    if kernel is None:
        # The Matern kernel is a good default for stellar variability.
        # It's a generalization of the RBF kernel and is less smooth, which is often
        # more realistic for physical processes.
        kernel = 1.0 * Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=0.1)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
    
    # Reshape time for scikit-learn
    time_reshaped = time.reshape(-1, 1)
    
    # Fit the GP to the data
    gp.fit(time_reshaped, flux)
    
    # Predict the stellar variability trend
    trend = gp.predict(time_reshaped)
    
    # Subtract the trend and normalize by the median
    detrended_flux = flux / trend
    normalized_flux = detrended_flux / np.median(detrended_flux)
    
    logging.info("GP detrending complete.")
    return normalized_flux

# --- 3. Hybrid BLS Feature Extraction ---

@memory.cache
def run_bls_search(time: np.ndarray, flux: np.ndarray, min_period=0.5, max_period=25, n_periods=10000):
    """
    Performs a Box-Least Squares (BLS) search to find transit-like signals.
    This function is cached to disk to avoid re-running on the same data.

    Args:
        time (np.ndarray): Time values.
        flux (np.ndarray): Detrended flux values.
        min_period (float): Minimum period to search.
        max_period (float): Maximum period to search.
        n_periods (int): Number of period grid points.

    Returns:
        dict: A dictionary containing the BLS results and statistics.
    """
    logging.info(f"Running BLS search from {min_period} to {max_period} days...")
    
    period_grid = np.exp(np.linspace(np.log(min_period), np.log(max_period), n_periods))
    bls = BoxLeastSquares(time, flux)
    power = bls.power(period_grid, duration=0.1)
    
    # Find the highest peak in the periodogram
    index = np.argmax(power.power)
    best_period = power.period[index]
    best_power = power.power[index]
    
    # Get statistics for the best-fit model
    stats = bls.compute_stats(best_period, duration=0.1)
    
    return {
        "period": best_period,
        "power": best_power,
        "depth": stats['depth'][0],
        "duration": stats['duration'][0],
        "snr": stats['depth_snr'][0],
        "periodogram": {"period": power.period, "power": power.power}
    }

def get_local_transit_view(time: np.ndarray, flux: np.ndarray, period: float, epoch: float, duration: float, n_points=256):
    """
    Extracts a "local view" of a transit, centered on the transit event and
    resampled to a fixed number of points. This is a crucial input for the
    representation model.

    Args:
        time, flux: The light curve data.
        period, epoch, duration: The parameters of the transit.
        n_points (int): The number of points in the output view.

    Returns:
        np.ndarray: The resampled flux of the local transit view.
    """
    # Phase-fold the light curve
    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    
    # Select a window around the transit (e.g., 3x the duration)
    window_mask = (phase > -1.5 * duration) & (phase < 1.5 * duration)
    
    if np.sum(window_mask) < 10: # Not enough points in the window
        return np.ones(n_points) # Return a flat line

    phase_window = phase[window_mask]
    flux_window = flux[window_mask]
    
    # Sort by phase
    sort_order = np.argsort(phase_window)
    phase_sorted = phase_window[sort_order]
    flux_sorted = flux_window[sort_order]
    
    # Interpolate to a fixed grid
    interp_func = interp1d(phase_sorted, flux_sorted, bounds_error=False, fill_value=1.0)
    new_phase_grid = np.linspace(-1.5 * duration, 1.5 * duration, n_points)
    
    return interp_func(new_phase_grid)

def full_feature_extraction_pipeline(filepath: str):
    """
    The main end-to-end function of this module. It takes a filepath and returns
    the features required for the downstream representation model.
    """
    logging.info(f"--- Starting Full Feature Extraction for {os.path.basename(filepath)} ---")
    
    # 1. Load data
    lc_df = load_light_curve(filepath)
    time, flux = lc_df['time'].values, lc_df['flux'].values
    
    # 2. Detrend
    detrended_flux = gp_detrend(time, flux)
    
    # 3. Run BLS to get primary features
    bls_results = run_bls_search(time, detrended_flux)
    
    # 4. Get local transit view based on BLS results
    # Note: BLS stats gives transit time relative to the start of the data
    epoch = bls_results.get('transit_time', time[0]) 
    local_view = get_local_transit_view(
        time, detrended_flux,
        period=bls_results['period'],
        epoch=epoch,
        duration=bls_results['duration']
    )
    
    # 5. Assemble the final feature dictionary
    features = {
        "bls_period": bls_results['period'],
        "bls_depth": bls_results['depth'],
        "bls_duration": bls_results['duration'],
        "bls_snr": bls_results['snr'],
        "local_view_flux": local_view,
        "original_filepath": filepath
    }
    
    logging.info(f"--- Feature Extraction Complete for {os.path.basename(filepath)} ---")
    return features

if __name__ == '__main__':
    # Example usage:
    # Create a dummy FITS file for demonstration
    from astropy.io import fits
    time_arr = np.linspace(0, 50, 5000)
    flux_arr = 1.0 - 0.01 * np.exp(-0.5 * ((time_arr - 25) / 0.1)**2) + 0.001 * np.random.randn(5000)
    
    primary_hdu = fits.PrimaryHDU()
    col1 = fits.Column(name='TIME', format='D', array=time_arr)
    col2 = fits.Column(name='PDCSAP_FLUX', format='D', array=flux_arr)
    hdu = fits.BinTableHDU.from_columns([col1, col2])
    
    dummy_fits_path = "dummy_tess_lc.fits"
    fits.HDUList([primary_hdu, hdu]).writeto(dummy_fits_path, overwrite=True)
    
    # Run the full pipeline on the dummy file
    extracted_features = full_feature_extraction_pipeline(dummy_fits_path)
    
    print("\n--- Extracted Features ---")
    for key, val in extracted_features.items():
        if isinstance(val, np.ndarray):
            print(f"{key}: array of shape {val.shape}")
        else:
            print(f"{key}: {val}")
