# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Реализация ResNet-блока для 1D-CNN.
    """
    def __init__(self, in_channels, out_channels, kernel_size=11):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ExoplanetNet(nn.Module):
    """
    1D-CNN на основе ResNet-блоков для детекции транзитов.
    input_length: длина окна W (например, 2000).
    num_blocks: количество ResNet-блоков.
    base_filters: начальное количество фильтров.
    kernel_size: размер ядра свертки.
    dropout_rate: вероятность dropout.
    """
    def __init__(self, input_length=2000, num_blocks=3, base_filters=16, kernel_size=11, dropout_rate=0.5):
        super().__init__()
        self.in_channels = base_filters
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        
        layers = []
        for i in range(num_blocks):
            out_channels = self.in_channels * 2
            layers.append(ResidualBlock(self.in_channels, out_channels, kernel_size=kernel_size))
            layers.append(nn.MaxPool1d(2))
            self.in_channels = out_channels
        
        self.blocks = nn.Sequential(*layers)

        pools = num_blocks
        reduced = max(1, input_length // (2 ** pools))
        self._reduced = reduced

        self.fc1 = nn.Linear(self.in_channels * reduced, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # x shape: (B, 1, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
