# src/detect.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.model import ExoplanetNet

def make_windows_from_series(flux_list, labels, window_size=2000, stride=200):
    """
    Создаёт окна X и метки Y из списка длинных кривых flux_list.
    Для синтетики: если исходная кривая помечена label==1, то окно помечается как positive,
    если в окне есть точка ниже mean-3*std (простая эвристика).
    Возвращает X (numpy array shape (N, window_size)), Y (numpy array shape (N,))
    """
    X = []
    Y = []
    for flux, lab in zip(flux_list, labels):
        L = len(flux)
        # проходим по окнам
        for start in range(0, max(1, L - window_size + 1), stride):
            w = np.array(flux[start:start+window_size], dtype=float)
            if len(w) < window_size:
                # дополняем повторением края
                pad = window_size - len(w)
                w = np.pad(w, (0, pad), mode='edge')
            # Метка окна: простая эвристика для синтетики
            if lab == 1 and np.any(w < (w.mean() - 3 * w.std())):
                X.append(w)
                Y.append(1)
            else:
                X.append(w)
                Y.append(0)
    if len(X) == 0:
        return np.zeros((0, window_size)), np.zeros((0,), dtype=int)
    return np.stack(X), np.array(Y, dtype=int)

def train_on_windows(X, Y, input_length=2000, epochs=5, batch_size=16, lr=1e-3, device='cpu'):
    """
    Обучает ExoplanetNet на наборах окон X, Y.
    X: numpy array (N, window_size)
    Y: numpy array (N,)
    Возвращает обученную модель (на device).
    """
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,W)
    Y_t = torch.tensor(Y, dtype=torch.long)
    ds = TensorDataset(X_t, Y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = ExoplanetNet(input_length)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for e in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
        print(f"Epoch {e+1}/{epochs} - loss={running_loss:.5f}")
    return model

def sliding_prediction(model, flux, window_size=2000, stride=200, device='cpu'):
    """
    Скользящее предсказание по длинной кривой flux (numpy array).
    Возвращает массив вероятностей класса 1 длины len(flux).
    """
    model.to(device)
    model.eval()
    L = len(flux)
    scores = np.zeros(L, dtype=float)
    counts = np.zeros(L, dtype=float)
    with torch.no_grad():
        for start in range(0, max(1, L - window_size + 1), stride):
            window = np.array(flux[start:start+window_size], dtype=float)
            if len(window) < window_size:
                pad = window_size - len(window)
                window = np.pad(window, (0, pad), mode='edge')
            inp = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            out = model(inp)  # (1,2)
            prob1 = F.softmax(out, dim=1)[0,1].item()
            scores[start:start+window_size] += prob1
            counts[start:start+window_size] += 1.0
    counts[counts == 0] = 1.0
    probs_time = scores / counts
    return probs_time

def extract_candidates(probs, time, threshold=0.5, min_len=3):
    """
    Находит регионы, где probs > threshold и длина >= min_len.
    Возвращает список словарей с start_idx, end_idx, start_time, end_time, mean_prob.
    """
    above = probs > threshold
    candidates = []
    i = 0
    L = len(probs)
    while i < L:
        if above[i]:
            j = i
            while j < L and above[j]:
                j += 1
            if (j - i) >= min_len:
                candidates.append({
                    'start_idx': int(i),
                    'end_idx': int(j-1),
                    'start_time': float(time[i]),
                    'end_time': float(time[j-1]),
                    'mean_prob': float(probs[i:j].mean())
                })
            i = j
        else:
            i += 1
    return candidates
