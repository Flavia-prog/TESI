from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis.metrics.reconstruction import mse, ssim_windowed


def test_mse_zero_for_identical_tensors() -> None:
    x = torch.rand(2, 3, 28, 28)
    assert mse(x, x) == 0.0


def test_ssim_is_one_for_identical_tensors() -> None:
    x = torch.rand(2, 3, 28, 28)
    scores = ssim_windowed(x, x)
    assert torch.allclose(scores, torch.ones_like(scores), atol=1e-6)
