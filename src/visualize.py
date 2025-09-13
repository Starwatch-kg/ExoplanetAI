# src/visualize.py — улучшённая версия, принимает probs и candidates
import matplotlib.pyplot as plt
import numpy as np

def plot_lightcurve(time, flux, probs=None, candidates=None):
    plt.figure(figsize=(12,4))
    plt.plot(time, flux, label='flux', linewidth=1)
    if probs is not None:
        p = np.array(probs)
        # безопасная нормализация probs для отрисовки рядом с flux
        if p.max() - p.min() > 1e-9:
            p_plot = (p - p.min())/(p.max()-p.min()) * (flux.max()-flux.min()) + (flux.min() - 0.05*(flux.max()-flux.min()))
        else:
            p_plot = np.full_like(p, flux.min() - 0.05*(flux.max()-flux.min()))
        plt.plot(time, p_plot, color='red', alpha=0.8, label='probability (scaled)')
    if candidates:
        for c in candidates:
            plt.axvspan(c['start_time'], c['end_time'], color='orange', alpha=0.3)
            plt.text((c['start_time']+c['end_time'])/2, flux.max(), f"{c['mean_prob']:.2f}",
                     horizontalalignment='center', verticalalignment='top', color='black')
    plt.xlabel('time')
    plt.ylabel('flux')
    plt.legend()
    plt.tight_layout()
    plt.show()
