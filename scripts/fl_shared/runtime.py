from __future__ import annotations

import platform
import random
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")

    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_yaml_config(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    with Path(config_path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        raise ValueError("YAML config must be a dictionary at the top level.")

    return loaded


def merge_config(
    defaults: dict[str, Any],
    yaml_config: dict[str, Any],
    cli_values: dict[str, Any],
) -> dict[str, Any]:
    resolved = defaults.copy()

    for key in defaults:
        if key in yaml_config and yaml_config[key] is not None:
            resolved[key] = yaml_config[key]
        if key in cli_values and cli_values[key] is not None:
            resolved[key] = cli_values[key]

    return resolved


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def clean_state_dict_for_plain_model(state_dict: dict[str, Any]) -> dict[str, Any]:
    if all(key.startswith("server_model.") for key in state_dict.keys()):
        return {
            key.replace("server_model.", "", 1): value
            for key, value in state_dict.items()
        }
    return state_dict


def _version_or_na(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def collect_provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "aijack": _version_or_na("aijack"),
            "torch": _version_or_na("torch"),
            "torchvision": _version_or_na("torchvision"),
            "medmnist": _version_or_na("medmnist"),
            "torchmetrics": _version_or_na("torchmetrics"),
            "numpy": _version_or_na("numpy"),
            "pandas": _version_or_na("pandas"),
            "scikit-learn": _version_or_na("scikit-learn"),
        },
    }
    if extra:
        provenance["extra"] = extra
    return provenance

