from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def clean_state_dict_for_plain_model(state_dict: dict[str, Any]) -> dict[str, Any]:
    if state_dict and all(key.startswith("server_model.") for key in state_dict.keys()):
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
