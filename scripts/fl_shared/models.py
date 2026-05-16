from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedMNISTCNN(nn.Module):
    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MedMNISTCNNWide(nn.Module):
    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


MODEL_REGISTRY = {
    "cnn": MedMNISTCNN,
    "cnn_wide": MedMNISTCNNWide,
}


def available_model_arches() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def build_model(arch: str, num_channels: int, num_classes: int) -> nn.Module:
    arch_key = arch.lower()
    if arch_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model architecture '{arch}'. "
            f"Available: {available_model_arches()}"
        )
    return MODEL_REGISTRY[arch_key](num_channels=num_channels, num_classes=num_classes)

