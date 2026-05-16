from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from aijack.attack.inversion import GradientInversionAttackServerManager
from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from thesis.data import build_client_indices, load_dataset
from thesis.experiments.config import AttackConfig
from thesis.models import get_model
from thesis.utils import (
    clean_state_dict_for_plain_model,
    collect_provenance,
    ensure_dir,
    load_yaml,
    resolve_device,
    save_json,
    set_seed,
    timestamp,
)


# Local denormalize to keep this module self-contained.
def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def _find_reconstruction_tensor(obj: Any) -> torch.Tensor | None:
    if torch.is_tensor(obj):
        if obj.ndim == 3:
            return obj.unsqueeze(0)
        if obj.ndim == 4:
            return obj
        if obj.ndim == 5:
            n_trials, batch_size, channels, height, width = obj.shape
            return obj.reshape(n_trials * batch_size, channels, height, width)
        return None

    if isinstance(obj, dict):
        for value in obj.values():
            found = _find_reconstruction_tensor(value)
            if found is not None:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_reconstruction_tensor(item)
            if found is not None:
                return found

    return None


def _build_manager(cfg: AttackConfig, device: torch.device, image_shape: tuple[int, int, int]):
    kwargs = {
        "num_trial_per_communication": int(cfg.num_trials),
        "log_interval": 10,
        "num_iteration": int(cfg.attack_iters),
        "distancename": str(cfg.distance),
        "device": device,
        "lr": float(cfg.attack_lr),
    }

    try:
        manager = GradientInversionAttackServerManager(image_shape, **kwargs)
        return manager, kwargs
    except TypeError:
        kwargs_without_device = {
            "num_trial_per_communication": int(cfg.num_trials),
            "log_interval": 10,
            "num_iteration": int(cfg.attack_iters),
            "distancename": str(cfg.distance),
            "lr": float(cfg.attack_lr),
        }
        manager = GradientInversionAttackServerManager(image_shape, **kwargs_without_device)
        return manager, kwargs_without_device


def _run_attack_api(server, client, dataloader: DataLoader, lr: float, device: torch.device):
    local_dataloaders = [dataloader]
    local_optimizers = [torch.optim.SGD(client.parameters(), lr=lr)]

    def criterion(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = labels.view(-1).long()
        return F.nll_loss(outputs, targets)

    api_kwargs = dict(
        server=server,
        clients=[client],
        criterion=criterion,
        local_optimizers=local_optimizers,
        local_dataloaders=local_dataloaders,
        num_communication=1,
        local_epoch=1,
        use_gradients=True,
    )

    try:
        api = FedAVGAPI(device=device, **api_kwargs)
    except TypeError:
        api = FedAVGAPI(**api_kwargs)

    api.run()


def _load_experiment(experiment_dir: Path) -> tuple[dict[str, Any], Path]:
    config_path = experiment_dir / "config.yaml"
    model_path = experiment_dir / "final_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    config = load_yaml(config_path)
    required_keys = ["seed", "num_clients", "batch_size", "split_type", "data_dir", "lr"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required keys in config.yaml: {missing}")

    return config, model_path


def _build_attack_dataloader(
    train_dataset,
    client_indices: list,
    client_id: int,
    sample_index: int,
    attack_batch_size: int,
) -> tuple[DataLoader, Any]:
    if client_id < 0 or client_id >= len(client_indices):
        raise ValueError(f"Invalid client_id {client_id}. Expected [0, {len(client_indices) - 1}].")
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0.")
    if attack_batch_size <= 0:
        raise ValueError("attack_batch_size must be > 0.")

    chosen_client_indices = client_indices[client_id]
    max_start = len(chosen_client_indices) - attack_batch_size

    if sample_index > max_start:
        raise ValueError(
            f"Invalid sample_index {sample_index} for attack_batch_size={attack_batch_size}. "
            f"Allowed range is [0, {max_start}]."
        )

    selected_indices = chosen_client_indices[sample_index : sample_index + attack_batch_size]
    subset = Subset(train_dataset, selected_indices.tolist())
    loader = DataLoader(subset, batch_size=attack_batch_size, shuffle=False, num_workers=0)
    return loader, selected_indices


def run_attack(
    experiment_dir: str | Path,
    cfg: AttackConfig,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    experiment_path = Path(experiment_dir)
    training_config, model_path = _load_experiment(experiment_path)

    dataset = cfg.dataset or training_config.get("dataset", "bloodmnist")
    model_name = cfg.model_name or training_config.get("model_name") or training_config.get("model_arch", "small_cnn")

    set_seed(int(training_config["seed"]))
    device = resolve_device(str(cfg.device))

    train_dataset, _info, n_channels, num_classes = load_dataset(dataset, "train", str(training_config["data_dir"]))

    split_indices = build_client_indices(
        train_dataset=train_dataset,
        split_type=str(training_config["split_type"]),
        num_clients=int(training_config["num_clients"]),
        seed=int(training_config["seed"]),
        alpha=float(training_config.get("alpha", 0.0)),
        num_classes=num_classes,
    )

    attack_loader, selected_indices = _build_attack_dataloader(
        train_dataset=train_dataset,
        client_indices=split_indices,
        client_id=int(cfg.client_id),
        sample_index=int(cfg.sample_index),
        attack_batch_size=int(cfg.attack_batch_size),
    )

    images, _labels = next(iter(attack_loader))
    images = images.to(device)
    image_shape = tuple(int(x) for x in images.shape[1:])

    raw_state_dict = torch.load(model_path, map_location=device)
    state_dict = clean_state_dict_for_plain_model(raw_state_dict)

    client_model = get_model(model_name, n_channels, num_classes).to(device)
    client_model.load_state_dict(state_dict)
    client = FedAVGClient(client_model, user_id=int(cfg.client_id))

    server_model = get_model(model_name, n_channels, num_classes).to(device)
    server_model.load_state_dict(state_dict)

    manager, manager_kwargs = _build_manager(cfg, device, image_shape)
    AttackFedAVGServer = manager.attach(FedAVGServer)
    server = AttackFedAVGServer([client], server_model)

    attack_run_dir = (
        ensure_dir(Path(output_dir))
        if output_dir is not None
        else ensure_dir(experiment_path / "attacks" / f"gradient_inversion_{timestamp()}")
    )

    attack_status = "ok"
    attack_error = None

    try:
        _run_attack_api(
            server=server,
            client=client,
            dataloader=attack_loader,
            lr=float(training_config["lr"]),
            device=device,
        )
    except Exception as error:
        attack_status = "failed"
        attack_error = f"{type(error).__name__}: {error}"

    reconstructions = getattr(server, "attack_results", None)
    reconstructed_tensor = _find_reconstruction_tensor(reconstructions)

    original_images_cpu = images.detach().cpu()
    torch.save(original_images_cpu, attack_run_dir / "original_images.pt")
    save_image(
        denormalize(original_images_cpu),
        attack_run_dir / "original_images.png",
        nrow=min(8, original_images_cpu.size(0)),
    )

    reconstruction_mse = None
    num_reconstructions = 0

    if reconstructed_tensor is not None:
        reconstructed_cpu = reconstructed_tensor.detach().cpu()
        num_reconstructions = int(reconstructed_cpu.size(0))

        torch.save(reconstructed_cpu, attack_run_dir / "reconstructed_images.pt")
        save_image(
            denormalize(reconstructed_cpu),
            attack_run_dir / "reconstructed_images.png",
            nrow=min(8, reconstructed_cpu.size(0)),
        )

        if reconstructed_cpu.shape == original_images_cpu.shape:
            reconstruction_mse = float(torch.mean((reconstructed_cpu - original_images_cpu) ** 2).item())
    elif attack_status == "ok":
        attack_status = "no_reconstruction"

    metrics = {
        "experiment_dir": str(experiment_path.resolve()),
        "dataset": dataset,
        "model_name": model_name,
        **asdict(cfg),
        "image_shape": list(image_shape),
        "device": str(device),
        "split_type": training_config["split_type"],
        "alpha": float(training_config.get("alpha", 0.0)),
        "seed": int(training_config["seed"]),
        "model_path": str(model_path.resolve()),
        "attack_status": attack_status,
        "attack_error": attack_error,
        "number_of_reconstructions": num_reconstructions,
        "reconstruction_mse": reconstruction_mse,
        "original_shape": list(original_images_cpu.shape),
        "reconstructed_shape": (
            list(reconstructed_tensor.detach().cpu().shape) if reconstructed_tensor is not None else None
        ),
        "selected_dataset_indices": selected_indices.tolist(),
        "manager_kwargs_used": {
            key: str(value) if isinstance(value, torch.device) else value
            for key, value in manager_kwargs.items()
        },
        "output_dir": str(attack_run_dir.resolve()),
    }

    save_json(attack_run_dir / "attack_metrics.json", metrics)
    save_json(
        attack_run_dir / "provenance.json",
        collect_provenance(
            extra={
                "module": "thesis.attacks.gradient_inversion",
                "dataset": dataset,
                "model_name": model_name,
            }
        ),
    )

    return metrics
