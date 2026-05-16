from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from thesis.models import available_models
from thesis.utils import load_yaml


@dataclass
class TrainConfig:
    experiment_name: str | None = None
    dataset: str = "bloodmnist"
    model_name: str = "small_cnn"
    num_clients: int = 5
    num_rounds: int = 20
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    split_type: str = "iid"
    alpha: float = 0.5
    seed: int = 42
    device: str = "auto"
    data_dir: str = "./data"
    results_root: str = "results"


@dataclass
class DPTrainConfig(TrainConfig):
    dp_enabled: bool = True
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0
    dp_noise_std: float | None = None
    delta: float = 1e-5


@dataclass
class AttackConfig:
    client_id: int = 0
    sample_index: int = 0
    attack_batch_size: int = 1
    attack_iters: int = 300
    num_trials: int = 3
    attack_lr: float = 1.0
    distance: str = "l2"
    device: str = "auto"
    dataset: str | None = None
    model_name: str | None = None


@dataclass
class SweepConfig:
    experiment_dirs: list[str] | None = None
    client_ids: list[int] | None = None
    sample_indices: list[int] | None = None
    attack_batch_sizes: list[int] | None = None
    attack_iters: list[int] | None = None
    num_trials: list[int] | None = None
    attack_lrs: list[float] | None = None
    distances: list[str] | None = None
    dataset: str | None = None
    model_name: str | None = None
    device: str = "cpu"
    jobs: int = 1
    rerun_existing: bool = False
    max_runs: int | None = None
    output_root: str = "results/attack_parameter_impact"
    sweep_name: str | None = None


DEFAULT_SWEEP_EXPERIMENT_DIRS = [
    "results/iid_baseline",
    "results/noniid_alpha_1",
    "results/noniid_alpha_05",
    "results/noniid_alpha_01",
]


def _update_dataclass(instance, values: dict[str, Any]):
    valid = {f.name for f in fields(instance)}
    for key, value in values.items():
        if key in valid and value is not None:
            setattr(instance, key, value)
    return instance


def train_config_from_yaml(path: str | None, overrides: dict[str, Any] | None = None) -> TrainConfig:
    cfg = TrainConfig()
    if path:
        cfg = _update_dataclass(cfg, load_yaml(Path(path)))
    if overrides:
        cfg = _update_dataclass(cfg, overrides)

    cfg.model_name = str(cfg.model_name).lower()
    if cfg.model_name not in available_models():
        raise ValueError(f"Unsupported model_name '{cfg.model_name}'.")

    return cfg


def dp_train_config_from_yaml(
    path: str | None,
    overrides: dict[str, Any] | None = None,
) -> DPTrainConfig:
    cfg = DPTrainConfig()
    if path:
        cfg = _update_dataclass(cfg, load_yaml(Path(path)))
    if overrides:
        cfg = _update_dataclass(cfg, overrides)

    cfg.model_name = str(cfg.model_name).lower()
    if cfg.model_name not in available_models():
        raise ValueError(f"Unsupported model_name '{cfg.model_name}'.")

    return cfg


def attack_config_from_yaml(path: str | None, overrides: dict[str, Any] | None = None) -> AttackConfig:
    cfg = AttackConfig()
    if path:
        cfg = _update_dataclass(cfg, load_yaml(Path(path)))
    if overrides:
        cfg = _update_dataclass(cfg, overrides)

    if cfg.model_name is not None:
        cfg.model_name = str(cfg.model_name).lower()
        if cfg.model_name not in available_models():
            raise ValueError(f"Unsupported model_name '{cfg.model_name}'.")

    return cfg


def sweep_config_from_yaml(path: str | None, overrides: dict[str, Any] | None = None) -> SweepConfig:
    cfg = SweepConfig(
        experiment_dirs=list(DEFAULT_SWEEP_EXPERIMENT_DIRS),
        client_ids=[0],
        sample_indices=[0, 25, 50],
        attack_batch_sizes=[1],
        attack_iters=[1000],
        num_trials=[5],
        attack_lrs=[0.1],
        distances=["cossim"],
    )

    if path:
        cfg = _update_dataclass(cfg, load_yaml(Path(path)))
    if overrides:
        cfg = _update_dataclass(cfg, overrides)

    if cfg.model_name is not None:
        cfg.model_name = str(cfg.model_name).lower()
        if cfg.model_name not in available_models():
            raise ValueError(f"Unsupported model_name '{cfg.model_name}'.")

    return cfg


def as_config_dict(cfg: Any) -> dict[str, Any]:
    return asdict(cfg)
