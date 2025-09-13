import numpy as np

def inject_transit(flux, t, period, t0, depth, duration):
    out = flux.copy()
    phase = ((t - t0) % period)
    half = duration / 2.0
    mask = (phase >= (period - half)) | (phase <= half)
    out[mask] -= depth
    return out

def generate_synthetic(num_samples=200, length=2000, transit_fraction=0.4,
                       min_depth=0.005, max_depth=0.03):
    """
    Генерирует синтетические кривые и метки.
    Увеличен диапазон depth (min_depth..max_depth) чтобы транзиты были заметнее.
    """
    times = []
    fluxes = []
    labels = []
    t = np.linspace(0, 30, length)
    for i in range(num_samples):
        base = 1.0 + 0.001 * np.random.randn(length)
        if np.random.rand() < transit_fraction:
            period = np.random.uniform(1.0, 10.0)
            t0 = np.random.uniform(0, period)
            depth = np.random.uniform(min_depth, max_depth)  # усилили глубину
            duration = np.random.uniform(0.05, 0.5)
            flux = inject_transit(base, t, period, t0, depth, duration)
            labels.append(1)
        else:
            flux = base
            labels.append(0)
        times.append(t.copy())
        fluxes.append(flux)
    return times, fluxes, labels

def normalize_array(arr):
    a = np.array(arr, dtype=float)
    s = a.std()
    if s == 0:
        return a - a.mean()
    return (a - a.mean()) / s

def load_lightcurve_file(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data[:,0], data[:,1]
