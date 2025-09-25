import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from astropy.timeseries import BoxLeastSquares
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

# --- Configuration ---
BENCHMARK_DIR = "data/benchmark_v1"
MODEL_SAVE_PATH = "models/neural_periodogram_v1.pth"
N_EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 16
RESAMPLED_POINTS = 2000 # Fixed length for CNN input

# --- Ground Truth Periods (for evaluation) ---
# In a real scenario, this would be loaded from a comprehensive catalog.
GROUND_TRUTH_PERIODS = {
    261136679: 8.6,    # Pi Men c
    38846515: 0.44,   # LHS 3844 b
    307210830: 61.3,   # TOI-700 d (long period, will be missed by our grid)
    201332580: 8.99,   # K2-138 b
    40079924: 0.47,   # ASASSN-V J06... (EB)
    219863539: 1.44,   # V1007 Sco (EB)
    289899359: 3.38,   # U Sge (EB)
}

# --- Logging and Warnings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning) # Suppress minor astropy warnings

# --- 1. Neural Periodogram Model Definition ---
class SimpleCNNPeriodogram(nn.Module):
    """A simple 1D-CNN to score a phase-folded light curve."""
    def __init__(self, input_length=RESAMPLED_POINTS):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * (input_length // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.fc_layers(x)

# --- 2. Data Preparation and Model Training ---
def load_and_prepare_training_data(data_dir, true_periods):
    """Loads benchmark data and creates a training set for the neural model."""
    X, y = [], []
    logging.info("Preparing training data for Neural Periodogram...")
    
    for filename in tqdm(os.listdir(data_dir), desc="Loading Benchmark Data"):
        if not filename.endswith(".csv"):
            continue
        
        tic_id = int(filename.split('_')[0])
        if tic_id not in true_periods:
            continue

        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)
        lc = lk.LightCurve(time=df['time'], flux=df['flux'])
        
        true_period = true_periods[tic_id]
        
        # Positive sample: folded on the correct period
        try:
            folded_true = lc.fold(true_period).bin(bins=RESAMPLED_POINTS)
            X.append(folded_true.flux.value)
            y.append(1.0)
        except Exception:
            continue # Skip if folding fails

        # Negative sample: folded on a random, incorrect period
        random_period = true_period * (1 + np.random.uniform(0.2, 1.0))
        try:
            folded_false = lc.fold(random_period).bin(bins=RESAMPLED_POINTS)
            X.append(folded_false.flux.value)
            y.append(0.0)
        except Exception:
            continue

    return np.array(X), np.array(y)

def train_neural_model(X, y):
    """Trains the SimpleCNNPeriodogram model."""
    logging.info("Starting training of the Neural Periodogram...")
    X_tensor = torch.from_numpy(X).unsqueeze(1).float()
    y_tensor = torch.from_numpy(y).unsqueeze(1).float()
    
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = SimpleCNNPeriodogram()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(N_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        logging.info(f"Epoch {epoch+1}/{N_EPOCHS}, Val Loss: {val_loss/len(val_loader):.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logging.info(f"Trained model saved to {MODEL_SAVE_PATH}")
    return model

# --- 3. A/B Test Search Algorithm Implementations ---
def run_classic_bls(lc, period_grid):
    """(a) Pure Classic BLS Search."""
    bls = BoxLeastSquares(lc.time.value, lc.flux.value)
    results = bls.power(period_grid, duration=0.1)
    best_period = results.period[np.argmax(results.power)]
    return best_period

def run_neural_search(lc, model, period_grid):
    """(b) Pure Neural Periodogram Search."""
    model.eval()
    best_score, best_period = -np.inf, None
    with torch.no_grad():
        for period in period_grid:
            try:
                folded_flux = lc.fold(period).bin(bins=RESAMPLED_POINTS).flux.value
                flux_tensor = torch.from_numpy(folded_flux).unsqueeze(0).unsqueeze(0).float()
                score = model(flux_tensor).item()
                if score > best_score:
                    best_score = score
                    best_period = period
            except Exception:
                continue
    return best_period

def run_hybrid_search(lc, model, period_grid):
    """(c) Hybrid Search: BLS candidates + Neural refinement."""
    # Step 1: Get top 5 candidates from classic BLS
    bls = BoxLeastSquares(lc.time.value, lc.flux.value)
    results = bls.power(period_grid, duration=0.1)
    top_periods = results.period[np.argsort(-results.power)[:5]]
    
    # Step 2: Use neural network to score these 5 candidates
    best_period = run_neural_search(lc, model, top_periods)
    return best_period

# --- 4. Main Experiment Orchestration ---
def run_ab_test():
    """Main function to run the A/B test."""
    # Step 1: Train the neural model
    X, y = load_and_prepare_training_data(BENCHMARK_DIR, GROUND_TRUTH_PERIODS)
    model = train_neural_model(X, y)

    # Step 2: Run the A/B test on all benchmark data
    logging.info("\n--- Starting A/B Test of Search Algorithms ---")
    results = {"classic_bls": 0, "neural_search": 0, "hybrid_search": 0}
    total_targets = 0
    
    period_grid = np.linspace(0.3, 15, 5000) # Define a search grid

    for filename in tqdm(os.listdir(BENCHMARK_DIR), desc="A/B Testing"):
        if not filename.endswith(".csv"):
            continue
        
        tic_id = int(filename.split('_')[0])
        if tic_id not in GROUND_TRUTH_PERIODS:
            continue
        
        true_period = GROUND_TRUTH_PERIODS[tic_id]
        total_targets += 1

        filepath = os.path.join(BENCHMARK_DIR, filename)
        df = pd.read_csv(filepath)
        lc = lk.LightCurve(time=df['time'], flux=df['flux'])

        # Run algorithms
        p_bls = run_classic_bls(lc, period_grid)
        p_neural = run_neural_search(lc, model, period_grid)
        p_hybrid = run_hybrid_search(lc, model, period_grid)

        # Check if found period is close to the true period (within 5%)
        if np.isclose(p_bls, true_period, rtol=0.05):
            results["classic_bls"] += 1
        if np.isclose(p_neural, true_period, rtol=0.05):
            results["neural_search"] += 1
        if np.isclose(p_hybrid, true_period, rtol=0.05):
            results["hybrid_search"] += 1

    # Step 3: Report results
    logging.info("\n--- A/B Test Results ---")
    logging.info(f"Total targets evaluated: {total_targets}")
    for method, score in results.items():
        recovery_rate = (score / total_targets) * 100
        logging.info(f"Method: {method:<15} | Period Recovery Rate: {recovery_rate:.2f}% ({score}/{total_targets})")

if __name__ == "__main__":
    run_ab_test()