import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from aijack.attack.inversion import GradientInversionAttackServerManager
from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
from medmnist import INFO
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIJack gradient inversion attack on trained MedMNIST FedAvg results"
    )
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name. Default from config.yaml")
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


def load_config(experiment_dir: Path) -> tuple[dict, Path]:
    config_path = experiment_dir / "config.yaml"
    model_path = experiment_dir / "final_model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    required_keys = ["seed", "num_clients", "batch_size", "split_type", "data_dir", "lr"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required keys in config.yaml: {missing}")

    return config, model_path


def resolve_dataset(config: dict, dataset_override: str | None) -> str:
    dataset = dataset_override or config.get("dataset", "bloodmnist")
    dataset = str(dataset).lower()
    if dataset not in INFO:
        raise ValueError(f"Unsupported MedMNIST dataset: {dataset}")
    if INFO[dataset].get("task") != "multi-class":
        raise ValueError(f"Only multi-class tasks are supported, got {INFO[dataset].get('task')} for {dataset}")
    return dataset


def load_medmnist_train(dataset: str, data_dir: str):
    info = INFO[dataset]
    data_class = getattr(medmnist, info["python_class"])
    n_channels = int(info.get("n_channels", 1))
    num_classes = len(info["label"])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels),
        ]
    )

    train_dataset = data_class(split="train", transform=transform, download=True, root=data_dir)
    return train_dataset, n_channels, num_classes


def iid_split_indices(n_samples: int, num_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return np.array_split(indices, num_clients)


def dirichlet_split_indices(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    num_classes: int,
) -> list[np.ndarray]:
    if alpha <= 0:
        raise ValueError("alpha must be > 0 for Dirichlet splitting.")

    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]

        if class_indices.size == 0:
            continue

        rng.shuffle(class_indices)

        proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=float))
        split_points = (np.cumsum(proportions)[:-1] * class_indices.size).astype(int)
        class_splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(class_splits):
            if split.size > 0:
                client_indices[client_id].extend(split.tolist())

    split_indices = []
    for indices in client_indices:
        arr = np.array(indices, dtype=int)
        rng.shuffle(arr)
        split_indices.append(arr)

    if any(len(indices) == 0 for indices in split_indices):
        raise ValueError("Dirichlet split produced at least one empty client.")

    concatenated = np.concatenate(split_indices) if split_indices else np.array([], dtype=int)
    if np.unique(concatenated).size != labels.size:
        raise ValueError("Dirichlet split produced duplicated or missing samples.")

    return split_indices


def build_client_indices(train_dataset, config: dict, num_classes: int) -> list[np.ndarray]:
    labels = np.array(train_dataset.labels).reshape(-1)
    split_type = config["split_type"]

    if split_type == "iid":
        return iid_split_indices(
            n_samples=len(train_dataset),
            num_clients=int(config["num_clients"]),
            seed=int(config["seed"]),
        )

    if split_type == "dirichlet":
        alpha = float(config.get("alpha", 0.0))
        return dirichlet_split_indices(
            labels=labels,
            num_clients=int(config["num_clients"]),
            alpha=alpha,
            seed=int(config["seed"]),
            num_classes=num_classes,
        )

    raise ValueError(f"Unsupported split_type: {split_type}")


def build_attack_dataloader(
    train_dataset,
    client_indices: list[np.ndarray],
    client_id: int,
    sample_index: int,
    attack_batch_size: int,
) -> tuple[DataLoader, np.ndarray]:
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
            f"Invalid sample_index {sample_index} for attack_batch_size={attack_batch_size}. Allowed range is [0, {max_start}]."
        )

    selected_indices = chosen_client_indices[sample_index : sample_index + attack_batch_size]
    subset = Subset(train_dataset, selected_indices.tolist())

    loader = DataLoader(subset, batch_size=attack_batch_size, shuffle=False, num_workers=0)
    return loader, selected_indices


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def clean_state_dict_for_plain_model(state_dict: dict) -> dict:
    if all(key.startswith("server_model.") for key in state_dict.keys()):
        return {key.replace("server_model.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def build_manager(
    args: argparse.Namespace,
    device: torch.device,
    image_shape: tuple[int, int, int],
):
    kwargs = {
        "num_trial_per_communication": args.num_trials,
        "log_interval": 10,
        "num_iteration": args.attack_iters,
        "distancename": args.distance,
        "device": device,
        "lr": args.attack_lr,
    }

    try:
        manager = GradientInversionAttackServerManager(image_shape, **kwargs)
        return manager, kwargs
    except TypeError:
        kwargs_without_device = {
            "num_trial_per_communication": args.num_trials,
            "log_interval": 10,
            "num_iteration": args.attack_iters,
            "distancename": args.distance,
            "lr": args.attack_lr,
        }
        manager = GradientInversionAttackServerManager(image_shape, **kwargs_without_device)
        return manager, kwargs_without_device


def run_attack_api(
    server,
    client,
    dataloader: DataLoader,
    lr: float,
    device: torch.device,
):
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


def find_reconstruction_tensor(obj: Any) -> torch.Tensor | None:
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
            found = find_reconstruction_tensor(value)
            if found is not None:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = find_reconstruction_tensor(item)
            if found is not None:
                return found

    return None


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)

    config, model_path = load_config(experiment_dir)
    dataset = resolve_dataset(config, args.dataset)

    set_seed(int(config["seed"]))
    device = resolve_device(args.device)

    train_dataset, n_channels, num_classes = load_medmnist_train(dataset, config["data_dir"])

    split_indices = build_client_indices(
        train_dataset=train_dataset,
        config=config,
        num_classes=num_classes,
    )

    attack_loader, selected_indices = build_attack_dataloader(
        train_dataset=train_dataset,
        client_indices=split_indices,
        client_id=args.client_id,
        sample_index=args.sample_index,
        attack_batch_size=args.attack_batch_size,
    )

    images, labels = next(iter(attack_loader))
    images = images.to(device)
    labels = labels.view(-1).long().to(device)

    image_shape = tuple(int(x) for x in images.shape[1:])

    raw_state_dict = torch.load(model_path, map_location=device)
    state_dict = clean_state_dict_for_plain_model(raw_state_dict)

    client_model = MedMNISTCNN(num_channels=n_channels, num_classes=num_classes).to(device)
    client_model.load_state_dict(state_dict)
    client = FedAVGClient(client_model, user_id=args.client_id)

    server_model = MedMNISTCNN(num_channels=n_channels, num_classes=num_classes).to(device)
    server_model.load_state_dict(state_dict)

    manager, manager_kwargs = build_manager(args, device, image_shape)

    AttackFedAVGServer = manager.attach(FedAVGServer)
    server = AttackFedAVGServer([client], server_model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "attacks" / f"gradient_inversion_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_status = "ok"
    attack_error = None

    try:
        run_attack_api(server=server, client=client, dataloader=attack_loader, lr=float(config["lr"]), device=device)
    except Exception as error:
        attack_status = "failed"
        attack_error = f"{type(error).__name__}: {error}"

    reconstructions = getattr(server, "attack_results", None)
    reconstructed_tensor = find_reconstruction_tensor(reconstructions)

    original_images_cpu = images.detach().cpu()

    torch.save(original_images_cpu, output_dir / "original_images.pt")
    save_image(denormalize(original_images_cpu), output_dir / "original_images.png", nrow=min(8, original_images_cpu.size(0)))

    reconstruction_mse = None
    num_reconstructions = 0

    if reconstructed_tensor is not None:
        reconstructed_cpu = reconstructed_tensor.detach().cpu()
        num_reconstructions = int(reconstructed_cpu.size(0))

        torch.save(reconstructed_cpu, output_dir / "reconstructed_images.pt")
        save_image(denormalize(reconstructed_cpu), output_dir / "reconstructed_images.png", nrow=min(8, reconstructed_cpu.size(0)))

        if reconstructed_cpu.shape == original_images_cpu.shape:
            reconstruction_mse = float(torch.mean((reconstructed_cpu - original_images_cpu) ** 2).item())
    else:
        if attack_status == "ok":
            attack_status = "no_reconstruction"

    metrics = {
        "experiment_dir": str(experiment_dir.resolve()),
        "dataset": dataset,
        "client_id": int(args.client_id),
        "sample_index": int(args.sample_index),
        "attack_batch_size": int(args.attack_batch_size),
        "attack_iters": int(args.attack_iters),
        "num_trials": int(args.num_trials),
        "attack_lr": float(args.attack_lr),
        "distance": args.distance,
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
        "output_dir": str(output_dir.resolve()),
    }

    with (output_dir / "attack_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved original images to: {output_dir / 'original_images.png'}")
    if reconstructed_tensor is not None:
        print(f"Saved reconstructed images to: {output_dir / 'reconstructed_images.png'}")
    else:
        print("No reconstruction tensor was produced.")
    print(f"Saved attack metrics to: {output_dir / 'attack_metrics.json'}")


if __name__ == "__main__":
    main()
