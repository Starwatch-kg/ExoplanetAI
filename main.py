# main.py — debug / full pipeline
import os
import sys
import numpy as np
import torch

from src.preprocess import generate_synthetic, normalize_array
from src.detect import make_windows_from_series, train_on_windows, sliding_prediction, extract_candidates
from src.visualize import plot_lightcurve

print("DEBUG: старт main.py", flush=True)

# ---- 1) Сгенерировать синтетику ----
num_samples = 120
length = 2000
transit_fraction = 0.5
print(f"DEBUG: генерируем синтетику: samples={num_samples}, length={length}, transit_frac={transit_fraction}", flush=True)
times, fluxes, labels = generate_synthetic(num_samples=num_samples, length=length, transit_fraction=transit_fraction)
print("DEBUG: синтетика сгенерирована", flush=True)

# ---- 2) Нормализация ----
fluxes_norm = [normalize_array(f) for f in fluxes]
print("DEBUG: нормализация завершена", flush=True)

# ---- 3) Формируем окна ----
window_size = 2000
stride_make = 400
X, Y = make_windows_from_series(fluxes_norm, labels, window_size=window_size, stride=stride_make)
print("DEBUG: windows shape:", getattr(X, "shape", None), "labels shape:", getattr(Y, "shape", None), flush=True)
print("DEBUG: positives in windows:", int(Y.sum()), flush=True)
if getattr(X, "size", 0) == 0:
    print("DEBUG: ОШИБКА — массив окон пуст. Увеличь transit_fraction или измени логику make_windows_from_series.", flush=True)
    sys.exit(1)

# ---- 4) Обучаем модель ----
device = 'cpu'
epochs = 4
batch_size = 16
lr = 1e-3
print(f"DEBUG: начинаем обучение: epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}", flush=True)
model = train_on_windows(X, Y, input_length=window_size, epochs=epochs, batch_size=batch_size, lr=lr, device=device)
print("DEBUG: обучение завершено", flush=True)

# ---- 5) Скользящее предсказание ----
idx_test = 0
t = times[idx_test]
flux = fluxes_norm[idx_test]
print(f"DEBUG: тестируем на кривой index={idx_test} (label={labels[idx_test]})", flush=True)

stride_pred = 100
probs = sliding_prediction(model, np.array(flux), window_size=window_size, stride=stride_pred, device=device)
print("DEBUG: probs min/max/mean:", float(probs.min()), float(probs.max()), float(probs.mean()), flush=True)

# ---- 6) Извлечение кандидатов ----
threshold = 0.5
cands = extract_candidates(probs, t, threshold=threshold, min_len=3)
print("DEBUG: найдено кандидатов:", len(cands), flush=True)
for i, c in enumerate(cands[:5]):
    print(f"  candidate {i}: start_time={c['start_time']:.3f}, end_time={c['end_time']:.3f}, mean_prob={c['mean_prob']:.3f}", flush=True)

# ---- 7) Сохранение/визуализация ----
out_png = "debug_output.png"
try:
    plot_lightcurve(t, flux, probs=probs, candidates=cands)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.plot(t, flux, label='flux (normalized)')
    p = np.array(probs)
    if p.max() - p.min() > 1e-9:
        p_plot = (p - p.min())/(p.max()-p.min()) * (flux.max()-flux.min()) + (flux.min() - 0.05*(flux.max()-flux.min()))
    else:
        p_plot = np.full_like(p, flux.min() - 0.05*(flux.max()-flux.min()))
    plt.plot(t, p_plot, color='red', alpha=0.8, label='probability (scaled)')
    for c in cands:
        plt.axvspan(c['start_time'], c['end_time'], color='orange', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    print("DEBUG: сохранён файл", out_png, "в", os.getcwd(), flush=True)
except Exception as e:
    print("DEBUG: ошибка при визуализации:", e, flush=True)

print("DEBUG: конец main.py", flush=True)
