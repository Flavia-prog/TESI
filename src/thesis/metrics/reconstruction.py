from __future__ import annotations

import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    return float(torch.mean((reconstructed.float() - original.float()) ** 2).item())


def ssim_windowed(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Per-image windowed SSIM scores for tensors shaped [N, C, H, W]."""
    scores = []
    for image_index in range(original.size(0)):
        orig_img = original[image_index : image_index + 1]
        recon_img = reconstructed[image_index : image_index + 1]
        score = structural_similarity_index_measure(orig_img, recon_img, data_range=1.0)
        scores.append(score)
    return torch.stack(scores)


def lpips_distance(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Optional LPIPS metric.

    Requires torchmetrics LPIPS functional support at runtime.
    """
    try:
        from torchmetrics.functional.image.lpips import (
            learned_perceptual_image_patch_similarity,
        )
    except Exception as exc:  # pragma: no cover - depends on local install extras
        raise RuntimeError("LPIPS metric is unavailable in this environment.") from exc

    scores = []
    for image_index in range(original.size(0)):
        orig_img = original[image_index : image_index + 1]
        recon_img = reconstructed[image_index : image_index + 1]
        score = learned_perceptual_image_patch_similarity(orig_img, recon_img, normalize=False)
        scores.append(score)
    return torch.stack(scores)


def compute_best_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> tuple[float | None, float | None]:
    """Returns (best_mse, best_ssim)."""
    if original.ndim != 4 or reconstructed.ndim != 4:
        return None, None

    original_dn = denormalize(original.float())
    reconstructed_dn = denormalize(reconstructed.float())

    n_orig = original_dn.size(0)
    n_recon = reconstructed_dn.size(0)

    if n_orig <= 0 or n_recon <= 0:
        return None, None

    if n_recon == n_orig:
        candidates = reconstructed_dn.unsqueeze(0)
    elif n_recon % n_orig == 0:
        candidates = reconstructed_dn.view(n_recon // n_orig, n_orig, *original_dn.shape[1:])
    else:
        return None, None

    mse_values = []
    ssim_values = []

    for candidate in candidates:
        mse_values.append(mse(original_dn, candidate))
        ssim_values.append(float(ssim_windowed(original_dn, candidate).mean().item()))

    return float(min(mse_values)), float(max(ssim_values))
