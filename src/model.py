# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExoplanetNet(nn.Module):
    """
    1D-CNN для детекции транзитов.
    input_length: длина окна W, на которой модель обучается (например, 2000).
    Внимание: размер fc рассчитывается автоматически исходя из input_length и числа pooling-слоёв.
    """
    def __init__(self, input_length=2000):
        super().__init__()
        # свёртки + BatchNorm + Pool
        self.conv1 = nn.Conv1d(1, 16, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=11, padding=5)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)

        # сколько раз применили pool? здесь 3 раза -> деление длины на 8
        pools = 3
        reduced = max(1, input_length // (2 ** pools))
        self._reduced = reduced  # полезно для отладки

        self.fc1 = nn.Linear(64 * reduced, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # x shape: (B, 1, W)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
