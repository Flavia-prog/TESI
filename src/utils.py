import os
import random

import numpy as np
import torch


DEVICE = torch.device("cpu")


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_accuracy(model, dataloader, device=DEVICE):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
