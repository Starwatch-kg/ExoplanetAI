import os
import logging
import pandas as pd
import numpy as np
import lightkurve as lk
import concurrent.futures
from tqdm import tqdm

# --- Configuration ---
OUTPUT_DIR = "data/benchmark_v1"
SECTORS = range(1, 61)  # TESS Sectors 1-60
MAX_WORKERS = 8  # Adjust based on your machine's capability

# --- Target Lists (Ground Truth) ---
# Using a small, representative sample for this script.
# In a real scenario, these lists would be much larger, sourced from NASA Exoplanet Archive, etc.
CONFIRMED_PLANETS = {
    "Pi Men c": 261136679,
    "LHS 3844 b": 38846515,
    "TOI-700 d": 307210830,
    "K2-138 b": 201332580, # A multi-planet system
}

ECLIPSING_BINARIES = {
    "ASASSN-V J060000.76-310027.8": 40079924, # Well-known EB
    "V1007 Sco": 219863539,
    "U Sge": 289899359,
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_target(target_name: str, tic_id: int, label: int, output_dir: str):
    """
    Searches, downloads, processes, and saves the light curve for a single TIC ID.

    Args:
        target_name (str): Common name for the target for logging.
        tic_id (int): The TESS Input Catalog ID.
        label (int): The label (1 for planet, 0 for binary/false positive).
        output_dir (str): The directory to save the processed file.
    """
    try:
        # 1. Search for available light curves for the given TIC ID and sectors
        search_result = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS', sector=SECTORS)
        
        if not search_result:
            logging.warning(f"No data found for {target_name} (TIC {tic_id}) in sectors 1-60.")
            return None

        # 2. Download all available light curve files
        lc_collection = search_result.download_all()
        
        # 3. Stitch them into a single light curve, handling potential errors
        if len(lc_collection) > 1:
            stitched_lc = lc_collection.stitch()
        else:
            stitched_lc = lc_collection[0]

        # 4. Perform cleaning and normalization
        processed_lc = stitched_lc.remove_nans().remove_outliers(sigma=5).normalize()

        # 5. Save to a CSV file
        df = processed_lc.to_pandas()[['time', 'flux']]
        filename = f"{tic_id}_{label}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        return filepath

    except Exception as e:
        logging.error(f"Failed to process {target_name} (TIC {tic_id}). Error: {e}")
        return None

def create_benchmark_dataset():
    """
    Main function to orchestrate the creation of the benchmark dataset.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Starting benchmark dataset creation. Output will be in '{OUTPUT_DIR}'")

    # Combine all targets into a single list for processing
    targets = []
    for name, tic in CONFIRMED_PLANETS.items():
        targets.append({"name": name, "tic": tic, "label": 1})
    for name, tic in ECLIPSING_BINARIES.items():
        targets.append({"name": name, "tic": tic, "label": 0})

    successful_downloads = 0
    
    # Use ThreadPoolExecutor for parallel downloading and processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future for each target
        future_to_target = {
            executor.submit(process_target, t['name'], t['tic'], t['label'], OUTPUT_DIR): t 
            for t in targets
        }
        
        # Use tqdm to create a progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_target), total=len(targets), desc="Processing Targets"):
            result = future.result()
            if result:
                successful_downloads += 1
                logging.info(f"Successfully processed and saved: {os.path.basename(result)}")

    logging.info("--- Benchmark Dataset Creation Complete ---")
    logging.info(f"Successfully downloaded and processed {successful_downloads}/{len(targets)} targets.")
    logging.info(f"Dataset saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    create_benchmark_dataset()
