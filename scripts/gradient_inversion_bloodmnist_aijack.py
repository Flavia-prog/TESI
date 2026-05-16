from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from aijack.collaborative.fedavg import FedAVGClient, FedAVGServer
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from fl_shared.attack_core import build_manager, find_reconstruction_tensor, run_attack_api
from fl_shared.medmnist_data import build_client_indices, load_medmnist_train, resolve_dataset_name
from fl_shared.models import available_model_arches, build_model
from fl_shared.runtime import (
    clean_state_dict_for_plain_model,
    collect_provenance,
    denormalize,
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIJack gradient inversion attack on trained MedMNIST FedAvg results"
    )
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset override. Default is loaded from experiment config.",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default=None,
        choices=available_model_arches(),
        help="Model architecture override. Default is loaded from experiment config.",
    )
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument("--attack-iters", type=int, default=300)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--attack-lr", type=float, default=1.0)
    parser.add_argument("--distance", type=str, choices=["l2", "cossim"], default="l2")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def load_config(experiment_dir: Path) -> tuple[dict[str, Any], Path]:
    config_path = experiment_dir / "config.yaml"
    model_path = experiment_dir / "final_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    required_keys = ["seed", "num_clients", "batch_size", "split_type", "data_dir", "lr"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required keys in config.yaml: {missing}")

    return config, model_path


def resolve_model_arch(config: dict[str, Any], model_arch_override: str | None) -> str:
    model_arch = model_arch_override or config.get("model_arch", "cnn")
    return str(model_arch).lower()


def resolve_dataset(config: dict[str, Any], dataset_override: str | None) -> str:
    dataset = dataset_override or config.get("dataset", "bloodmnist")
    return resolve_dataset_name(dataset)


def build_attack_dataloader(
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


def run_gradient_inversion_attack(
    experiment_dir: Path,
    client_id: int,
    sample_index: int,
    attack_batch_size: int,
    attack_iters: int,
    num_trials: int,
    attack_lr: float,
    distance: str,
    device_arg: str,
    output_dir: Path | None = None,
    dataset_override: str | None = None,
    model_arch_override: str | None = None,
) -> dict[str, Any]:
    config, model_path = load_config(experiment_dir)
    dataset = resolve_dataset(config, dataset_override)
    model_arch = resolve_model_arch(config, model_arch_override)

    set_seed(int(config["seed"]))
    device = resolve_device(device_arg)

    train_dataset, n_channels, num_classes = load_medmnist_train(dataset, str(config["data_dir"]))

    split_indices = build_client_indices(
        train_dataset=train_dataset,
        split_type=str(config["split_type"]),
        num_clients=int(config["num_clients"]),
        seed=int(config["seed"]),
        alpha=float(config.get("alpha", 0.0)),
        num_classes=num_classes,
    )

    attack_loader, selected_indices = build_attack_dataloader(
        train_dataset=train_dataset,
        client_indices=split_indices,
        client_id=client_id,
        sample_index=sample_index,
        attack_batch_size=attack_batch_size,
    )

    images, _labels = next(iter(attack_loader))
    images = images.to(device)
    image_shape = tuple(int(x) for x in images.shape[1:])

    raw_state_dict = torch.load(model_path, map_location=device)
    state_dict = clean_state_dict_for_plain_model(raw_state_dict)

    client_model = build_model(model_arch, n_channels, num_classes).to(device)
    client_model.load_state_dict(state_dict)
    client = FedAVGClient(client_model, user_id=client_id)

    server_model = build_model(model_arch, n_channels, num_classes).to(device)
    server_model.load_state_dict(state_dict)

    manager, manager_kwargs = build_manager(
        num_trials=num_trials,
        attack_iters=attack_iters,
        distance=distance,
        attack_lr=attack_lr,
        device=device,
        image_shape=image_shape,
    )

    AttackFedAVGServer = manager.attach(FedAVGServer)
    server = AttackFedAVGServer([client], server_model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    effective_output_dir = (
        output_dir
        if output_dir is not None
        else experiment_dir / "attacks" / f"gradient_inversion_{timestamp}"
    )
    effective_output_dir.mkdir(parents=True, exist_ok=True)

    attack_status = "ok"
    attack_error = None

    try:
        run_attack_api(
            server=server,
            client=client,
            dataloader=attack_loader,
            lr=float(config["lr"]),
            device=device,
        )
    except Exception as error:
        attack_status = "failed"
        attack_error = f"{type(error).__name__}: {error}"

    reconstructions = getattr(server, "attack_results", None)
    reconstructed_tensor = find_reconstruction_tensor(reconstructions)

    original_images_cpu = images.detach().cpu()

    torch.save(original_images_cpu, effective_output_dir / "original_images.pt")
    save_image(
        denormalize(original_images_cpu),
        effective_output_dir / "original_images.png",
        nrow=min(8, original_images_cpu.size(0)),
    )

    reconstruction_mse = None
    num_reconstructions = 0

    if reconstructed_tensor is not None:
        reconstructed_cpu = reconstructed_tensor.detach().cpu()
        num_reconstructions = int(reconstructed_cpu.size(0))

        torch.save(reconstructed_cpu, effective_output_dir / "reconstructed_images.pt")
        save_image(
            denormalize(reconstructed_cpu),
            effective_output_dir / "reconstructed_images.png",
            nrow=min(8, reconstructed_cpu.size(0)),
        )

        if reconstructed_cpu.shape == original_images_cpu.shape:
            reconstruction_mse = float(torch.mean((reconstructed_cpu - original_images_cpu) ** 2).item())
    elif attack_status == "ok":
        attack_status = "no_reconstruction"

    metrics = {
        "experiment_dir": str(experiment_dir.resolve()),
        "dataset": dataset,
        "model_arch": model_arch,
        "client_id": int(client_id),
        "sample_index": int(sample_index),
        "attack_batch_size": int(attack_batch_size),
        "attack_iters": int(attack_iters),
        "num_trials": int(num_trials),
        "attack_lr": float(attack_lr),
        "distance": distance,
        "image_shape": list(image_shape),
        "device": str(device),
        "split_type": config["split_type"],
        "alpha": float(config.get("alpha", 0.0)),
        "seed": int(config["seed"]),
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
        "output_dir": str(effective_output_dir.resolve()),
    }

    with (effective_output_dir / "attack_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    provenance = collect_provenance(
        extra={
            "script": "gradient_inversion_bloodmnist_aijack.py",
            "dataset": dataset,
            "model_arch": model_arch,
        }
    )
    with (effective_output_dir / "provenance.json").open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    return metrics


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    metrics = run_gradient_inversion_attack(
        experiment_dir=experiment_dir,
        client_id=args.client_id,
        sample_index=args.sample_index,
        attack_batch_size=args.attack_batch_size,
        attack_iters=args.attack_iters,
        num_trials=args.num_trials,
        attack_lr=args.attack_lr,
        distance=args.distance,
        device_arg=args.device,
        output_dir=output_dir,
        dataset_override=args.dataset,
        model_arch_override=args.model_arch,
    )

    output_dir_resolved = Path(metrics["output_dir"])
    print(f"Saved original images to: {output_dir_resolved / 'original_images.png'}")
    if metrics.get("reconstructed_shape") is not None:
        print(f"Saved reconstructed images to: {output_dir_resolved / 'reconstructed_images.png'}")
    else:
        print("No reconstruction tensor was produced.")
    print(f"Saved attack metrics to: {output_dir_resolved / 'attack_metrics.json'}")


if __name__ == "__main__":
    main()
