from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis.data.splits import dirichlet_split_indices, iid_split_indices


def test_iid_split_covers_all_samples_once() -> None:
    n_samples = 100
    splits = iid_split_indices(n_samples=n_samples, num_clients=5, seed=42)
    merged = np.concatenate(splits)

    assert merged.size == n_samples
    assert np.unique(merged).size == n_samples


def test_dirichlet_split_covers_all_samples_once() -> None:
    rng = np.random.default_rng(123)
    labels = rng.integers(low=0, high=4, size=200)

    splits = dirichlet_split_indices(
        labels=labels,
        num_clients=5,
        alpha=0.5,
        seed=42,
        num_classes=4,
    )

    merged = np.concatenate(splits)

    assert merged.size == labels.size
    assert np.unique(merged).size == labels.size
    assert all(len(client_indices) > 0 for client_indices in splits)
