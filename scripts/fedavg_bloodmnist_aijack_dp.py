import argparse
import copy
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
from aijack.defense import GeneralMomentAccountant
from aijack.defense.dp.manager import DPSGDClientManager, DPSGDManager
from medmnist import INFO
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


NUM_CLASSES = 8


class BloodMNISTCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
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
    parser = argparse.ArgumentParser(description="AIJack FedAvg + AIJack DP on BloodMNIST")

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)

    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)

    parser.add_argument("--optimizer", type=str, choices=["SGD"], default=None)

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
    )

    parser.add_argument(
        "--split-type",
        type=str,
        choices=["iid", "dirichlet"],
        default=None,
    )

    parser.add_argument("--dp-enabled", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--clip-norm", type=float, default=None)
    parser.add_argument("--noise-multiplier", type=float, default=None)
    parser.add_argument("--dp-noise-std", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)

    return parser.parse_args()


def parse_bool_arg(value: str | bool | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def load_yaml_config(config_path: str | None) -> dict:
    if config_path is None:
        return {}

    with Path(config_path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    if not isinstance(loaded, dict):
        raise ValueError("YAML config must be a dictionary at the top level.")

    return loaded


def resolve_config(args: argparse.Namespace) -> dict:
    defaults = {
        "num_clients": 5,
        "num_rounds": 20,
        "local_epochs": 1,
        "batch_size": 64,
        "lr": 0.01,
        "alpha": 0.5,
        "seed": 42,
        "data_dir": "./data",
        "device": "auto",
        "split_type": "iid",
        "optimizer": "SGD",
        "dp_enabled": False,
        "clip_norm": 1.0,
        "noise_multiplier": 0.0,
        "dp_noise_std": None,
        "delta": 1e-5,
    }

    yaml_config = load_yaml_config(args.config)

    cli_values = {
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "device": args.device,
        "split_type": args.split_type,
        "optimizer": args.optimizer,
        "dp_enabled": parse_bool_arg(args.dp_enabled),
        "clip_norm": args.clip_norm,
        "noise_multiplier": args.noise_multiplier,
        "dp_noise_std": args.dp_noise_std,
        "delta": args.delta,
    }

    resolved = defaults.copy()

    for key in defaults:
        if key in yaml_config and yaml_config[key] is not None:
            resolved[key] = yaml_config[key]
        if cli_values[key] is not None:
            resolved[key] = cli_values[key]

    resolved["dp_enabled"] = parse_bool_arg(resolved["dp_enabled"])

    if resolved["optimizer"] != "SGD":
        raise ValueError("Only optimizer=SGD is supported in this script.")

    if float(resolved["clip_norm"]) <= 0:
        raise ValueError("clip_norm must be > 0.")

    if float(resolved["noise_multiplier"]) < 0:
        raise ValueError("noise_multiplier must be >= 0.")

    if resolved["dp_noise_std"] is not None and float(resolved["dp_noise_std"]) < 0:
        raise ValueError("dp_noise_std must be >= 0 when provided.")

    if float(resolved["delta"]) <= 0 or float(resolved["delta"]) >= 1:
        raise ValueError("delta must be between 0 and 1.")

    if resolved["dp_noise_std"] is not None:
        effective_noise_multiplier = float(resolved["dp_noise_std"]) / float(
            resolved["clip_norm"]
        )
    else:
        effective_noise_multiplier = float(resolved["noise_multiplier"])

    if resolved["dp_enabled"]:
        actual_noise_std = float(resolved["clip_norm"]) * effective_noise_multiplier
    else:
        effective_noise_multiplier = 0.0
        actual_noise_std = 0.0

    resolved["effective_noise_multiplier"] = float(effective_noise_multiplier)
    resolved["actual_noise_std"] = float(actual_noise_std)

    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = yaml_config.get("experiment_name")
    if not experiment_name:
        experiment_name = f"fedavg_bloodmnist_dp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    resolved["experiment_name"] = experiment_name

    return resolved


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


def load_bloodmnist(data_dir: str):
    info = INFO["bloodmnist"]
    data_class = getattr(medmnist, info["python_class"])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    train_dataset = data_class(
        split="train",
        transform=transform,
        download=True,
        root=data_dir,
    )

    val_dataset = data_class(
        split="val",
        transform=transform,
        download=True,
        root=data_dir,
    )

    test_dataset = data_class(
        split="test",
        transform=transform,
        download=True,
        root=data_dir,
    )

    return train_dataset, val_dataset, test_dataset, info


def iid_split_indices(n_samples: int, num_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return np.array_split(indices, num_clients)


def dirichlet_split_indices(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    if alpha <= 0:
        raise ValueError("alpha must be > 0 for Dirichlet splitting.")

    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(NUM_CLASSES):
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

    lengths = [len(indices) for indices in split_indices]
    if any(length == 0 for length in lengths):
        raise ValueError(
            "Dirichlet split produced at least one empty client. "
            "Increase alpha or reduce num_clients."
        )

    concatenated = np.concatenate(split_indices) if split_indices else np.array([], dtype=int)
    if concatenated.size != labels.size:
        raise ValueError("Dirichlet split lost samples. Check split implementation.")

    if np.unique(concatenated).size != labels.size:
        raise ValueError("Dirichlet split duplicated samples. Check split implementation.")

    return split_indices


def create_client_dataloaders(
    train_dataset,
    num_clients: int,
    batch_size: int,
    seed: int,
    split_type: str,
    alpha: float,
):
    labels = np.array(train_dataset.labels).reshape(-1)
    if split_type == "iid":
        split_indices = iid_split_indices(
            n_samples=len(train_dataset),
            num_clients=num_clients,
            seed=seed,
        )
    elif split_type == "dirichlet":
        split_indices = dirichlet_split_indices(
            labels=labels,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported split_type: {split_type}")

    loaders = []
    distribution_rows = []

    for client_id, client_indices in enumerate(split_indices):
        subset = Subset(train_dataset, client_indices.tolist())

        generator = torch.Generator()
        generator.manual_seed(seed + client_id)

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        loaders.append(loader)

        local_labels = labels[client_indices]
        unique, counts = np.unique(local_labels, return_counts=True)
        class_count_map = {int(k): int(v) for k, v in zip(unique, counts)}

        row = {
            "client_id": client_id,
            "num_samples": int(len(client_indices)),
        }

        for class_id in range(NUM_CLASSES):
            row[f"class_{class_id}_count"] = class_count_map.get(class_id, 0)

        distribution_rows.append(row)

    return loaders, pd.DataFrame(distribution_rows)


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()

    total_loss = 0.0
    total_seen = 0
    non_finite_batches = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            targets = labels.view(-1).long().to(device)

            outputs = model(images)
            loss = F.nll_loss(outputs, targets)

            if not torch.isfinite(loss):
                non_finite_batches += 1
                continue

            total_loss += loss.item() * images.size(0)
            total_seen += images.size(0)

            preds = outputs.argmax(dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    if total_seen == 0:
        avg_loss = float("inf")
        accuracy = 0.0
        macro_f1 = 0.0
    else:
        avg_loss = total_loss / total_seen
        accuracy = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(
            all_targets,
            all_preds,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "targets": np.array(all_targets),
        "preds": np.array(all_preds),
        "non_finite_batches": non_finite_batches,
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    short_names = [
        "basophil",
        "eosinophil",
        "erythroblast",
        "imm. granulocytes",
        "lymphocyte",
        "monocyte",
        "neutrophil",
        "platelet",
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=short_names,
        yticklabels=short_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="AIJack FedAvg + DP BloodMNIST Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def get_state_dict_from_model(model: nn.Module):
    return copy.deepcopy(
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    )


def move_state_dict_to_device(state_dict, device: torch.device):
    return {k: v.to(device) for k, v in state_dict.items()}


def clean_state_dict_for_plain_model(state_dict: dict) -> dict:
    if all(key.startswith("server_model.") for key in state_dict.keys()):
        return {
            key.replace("server_model.", "", 1): value
            for key, value in state_dict.items()
        }
    return state_dict


def compute_epsilon_with_aijack(
    noise_multiplier: float,
    sampling_rate: float,
    iterations: int,
    delta: float,
) -> float | None:
    """
    Compute epsilon using AIJack's GeneralMomentAccountant.

    If DP is disabled or sigma is zero, epsilon is not finite/meaningful,
    so None is returned.
    """
    if noise_multiplier <= 0:
        return None

    accountant = GeneralMomentAccountant(
        noise_type="Gaussian",
        search="greedy",
        precision=0.001,
        orders=list(range(2, 64)),
        bound_type="rdp_tight_upperbound",
        backend="python",
    )

    accountant.reset_step_info()
    accountant.add_step_info(
        {"sigma": float(noise_multiplier)},
        float(sampling_rate),
        int(iterations),
    )

    return float(accountant.get_epsilon(float(delta)))


def make_optimizer_grad_safe(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
    Compatibility patch for AIJack's DP optimizer.

    Some AIJack DP internals assume every parameter has a non-None grad tensor.
    With small/non-IID clients, some lots can reach step_for_lot() with
    param.grad still equal to None.

    This wrapper creates missing zero gradients:
    - when the optimizer is created
    - after zero_grad()
    - immediately before step_for_lot()
    """

    def ensure_grad_tensors_exist() -> None:
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    param.grad = torch.zeros_like(param.data)

    ensure_grad_tensors_exist()

    original_zero_grad = optimizer.zero_grad

    def zero_grad_keep_tensors(*args, **kwargs):
        original_zero_grad()
        ensure_grad_tensors_exist()

    optimizer.zero_grad = zero_grad_keep_tensors

    if hasattr(optimizer, "step_for_lot"):
        original_step_for_lot = optimizer.step_for_lot

        def step_for_lot_grad_safe(*args, **kwargs):
            ensure_grad_tensors_exist()
            result = original_step_for_lot(*args, **kwargs)
            ensure_grad_tensors_exist()
            return result

        optimizer.step_for_lot = step_for_lot_grad_safe

    return optimizer


def build_clients_optimizers(
    config: dict,
    local_dataloaders: list[DataLoader],
    device: torch.device,
):
    clients = []
    local_optimizers = []

    for client_id, local_loader in enumerate(local_dataloaders):
        client_model = BloodMNISTCNN(num_classes=NUM_CLASSES).to(device)

        if bool(config["dp_enabled"]):
            local_dataset = local_loader.dataset
            local_dataset_size = len(local_dataset)

            if local_dataset_size <= 0:
                raise ValueError(f"Client {client_id} has an empty local dataset.")

            # AIJack DP-SGD can fail when lot_size/batch_size is larger than
            # a very small non-IID client's local dataset. Use client-safe
            # lot and batch sizes while preserving the global config value
            # for ordinary clients.
            client_lot_size = min(int(config["batch_size"]), local_dataset_size)
            client_batch_size = min(int(config["batch_size"]), client_lot_size)

            iterations = max(
                1,
                int(math.ceil(local_dataset_size / client_lot_size)),
            )

            accountant = GeneralMomentAccountant(
                noise_type="Gaussian",
                backend="python",
            )

            privacy_manager = DPSGDManager(
                accountant,
                torch.optim.SGD,
                l2_norm_clip=float(config["clip_norm"]),
                dataset=local_dataset,
                lot_size=client_lot_size,
                batch_size=client_batch_size,
                iterations=iterations,
            )

            dp_client_manager = DPSGDClientManager(
                privacy_manager,
                float(config["effective_noise_multiplier"]),
            )

            DPSGDFedAVGClient, DPOptimizerWrapper = dp_client_manager.attach(FedAVGClient)

            client = DPSGDFedAVGClient(
                client_model,
                user_id=client_id,
            )

            optimizer = DPOptimizerWrapper(
                client.parameters(),
                lr=float(config["lr"]),
            )

            optimizer = make_optimizer_grad_safe(optimizer)
        else:
            client = FedAVGClient(
                client_model,
                user_id=client_id,
            )

            optimizer = torch.optim.SGD(
                client.parameters(),
                lr=float(config["lr"]),
            )

        clients.append(client)
        local_optimizers.append(optimizer)

    return clients, local_optimizers


def build_fedavg_api(
    server,
    clients,
    criterion,
    local_optimizers,
    local_dataloaders,
    num_communication,
    local_epoch,
    custom_action,
    device,
):
    kwargs = dict(
        server=server,
        clients=clients,
        criterion=criterion,
        local_optimizers=local_optimizers,
        local_dataloaders=local_dataloaders,
        num_communication=num_communication,
        local_epoch=local_epoch,
        custom_action=custom_action,
    )

    try:
        return FedAVGAPI(device=device, **kwargs)
    except TypeError:
        return FedAVGAPI(**kwargs)


def main() -> None:
    args = parse_args()
    config = resolve_config(args)

    set_seed(config["seed"])

    device = resolve_device(config["device"])

    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)

    results_dir = results_root / config["experiment_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, info = load_bloodmnist(config["data_dir"])

    class_names = [info["label"][str(i)] for i in range(len(info["label"]))]

    local_dataloaders, client_distribution_df = create_client_dataloaders(
        train_dataset=train_dataset,
        num_clients=config["num_clients"],
        batch_size=config["batch_size"],
        seed=config["seed"],
        split_type=config["split_type"],
        alpha=config["alpha"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    clients, local_optimizers = build_clients_optimizers(
        config=config,
        local_dataloaders=local_dataloaders,
        device=device,
    )

    server = FedAVGServer(
        clients,
        BloodMNISTCNN(num_classes=NUM_CLASSES).to(device),
    )

    history_rows = []

    best_val_loss = float("inf")
    best_round = 0
    best_state_dict = None

    def criterion(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = labels.view(-1).long()
        return F.nll_loss(outputs, targets)

    def custom_action(api: FedAVGAPI) -> None:
        nonlocal best_val_loss, best_round, best_state_dict

        round_idx = len(history_rows) + 1

        val_metrics = evaluate_model(
            model=api.server,
            data_loader=val_loader,
            device=device,
        )

        history_rows.append(
            {
                "round": round_idx,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "non_finite_batches": val_metrics["non_finite_batches"],
            }
        )

        print(
            f"Round {round_idx:03d}/{config['num_rounds']} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_acc: {val_metrics['accuracy']:.4f} | "
            f"val_macro_f1: {val_metrics['macro_f1']:.4f} | "
            f"non_finite_batches: {val_metrics['non_finite_batches']}"
        )

        if math.isfinite(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_round = round_idx
            best_state_dict = get_state_dict_from_model(api.server)

    fedavg_api = build_fedavg_api(
        server=server,
        clients=clients,
        criterion=criterion,
        local_optimizers=local_optimizers,
        local_dataloaders=local_dataloaders,
        num_communication=config["num_rounds"],
        local_epoch=config["local_epochs"],
        custom_action=custom_action,
        device=device,
    )

    print(f"Using device: {device}")
    print(
        f"Starting AIJack FedAvg + DP BloodMNIST | "
        f"clients={config['num_clients']}, "
        f"rounds={config['num_rounds']}, "
        f"local_epochs={config['local_epochs']}, "
        f"batch_size={config['batch_size']}, "
        f"lr={config['lr']}, "
        f"split={config['split_type']}, "
        f"alpha={config['alpha']}, "
        f"optimizer={config['optimizer']}, "
        f"dp_enabled={config['dp_enabled']}, "
        f"clip_norm={config['clip_norm']}, "
        f"noise_multiplier={config['noise_multiplier']}, "
        f"dp_noise_std={config['dp_noise_std']}, "
        f"effective_noise_multiplier={config['effective_noise_multiplier']}, "
        f"actual_noise_std={config['actual_noise_std']}, "
        f"delta={config['delta']}"
    )

    fedavg_api.run()

    if best_state_dict is not None:
        server.load_state_dict(move_state_dict_to_device(best_state_dict, device))
    else:
        print(
            "Warning: no finite validation loss was found. "
            "Evaluating the final server model instead."
        )

    test_metrics = evaluate_model(
        model=server,
        data_loader=test_loader,
        device=device,
    )

    clients_per_round = int(config["num_clients"])
    sampling_rate = clients_per_round / int(config["num_clients"])
    delta = float(config["delta"])

    epsilon = compute_epsilon_with_aijack(
        noise_multiplier=float(config["effective_noise_multiplier"]),
        sampling_rate=float(sampling_rate),
        iterations=int(config["num_rounds"]),
        delta=delta,
    )

    final_config = {
        "experiment_name": config["experiment_name"],
        "dataset": "bloodmnist",
        "framework": "aijack",
        "algorithm": "fedavg_dp",
        "num_clients": int(config["num_clients"]),
        "num_rounds": int(config["num_rounds"]),
        "local_epochs": int(config["local_epochs"]),
        "batch_size": int(config["batch_size"]),
        "lr": float(config["lr"]),
        "alpha": float(config["alpha"]),
        "optimizer": "SGD",
        "seed": int(config["seed"]),
        "split_type": config["split_type"],
        "device": str(device),
        "data_dir": config["data_dir"],
        "dp_enabled": bool(config["dp_enabled"]),
        "clip_norm": float(config["clip_norm"]),
        "noise_multiplier": float(config["noise_multiplier"]),
        "dp_noise_std": (
            float(config["dp_noise_std"]) if config["dp_noise_std"] is not None else None
        ),
        "effective_noise_multiplier": float(config["effective_noise_multiplier"]),
        "actual_noise_std": float(config["actual_noise_std"]),
        "delta": delta,
        "sampling_rate": float(sampling_rate),
        "clients_per_round": int(clients_per_round),
        "epsilon": epsilon,
        "dp_impl": "aijack.defense.dp.manager.DPSGDManager + DPSGDClientManager",
        "privacy_accountant": "AIJack GeneralMomentAccountant",
        "accountant_noise_type": "Gaussian",
        "accountant_bound_type": "rdp_tight_upperbound",
    }

    config_path = results_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(final_config, f, sort_keys=False)

    per_class_f1 = f1_score(
        test_metrics["targets"],
        test_metrics["preds"],
        labels=list(range(NUM_CLASSES)),
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(
        test_metrics["targets"],
        test_metrics["preds"],
        labels=list(range(NUM_CLASSES)),
    )

    history_path = results_dir / "history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    client_distribution_path = results_dir / "client_distributions.csv"
    client_distribution_df.to_csv(client_distribution_path, index=False)

    test_row = {
        "seed": config["seed"],
        "num_clients": config["num_clients"],
        "num_rounds": config["num_rounds"],
        "local_epochs": config["local_epochs"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "alpha": config["alpha"],
        "device": str(device),
        "split_type": config["split_type"],
        "optimizer": "SGD",
        "dp_enabled": bool(config["dp_enabled"]),
        "clip_norm": float(config["clip_norm"]),
        "noise_multiplier": float(config["noise_multiplier"]),
        "effective_noise_multiplier": float(config["effective_noise_multiplier"]),
        "actual_noise_std": float(config["actual_noise_std"]),
        "delta": delta,
        "sampling_rate": float(sampling_rate),
        "clients_per_round": int(clients_per_round),
        "epsilon": epsilon,
        "privacy_accountant": "AIJack GeneralMomentAccountant",
        "best_round": best_round,
        "best_val_loss": float(best_val_loss),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_non_finite_batches": test_metrics["non_finite_batches"],
    }

    for class_id, score in enumerate(per_class_f1):
        test_row[f"test_f1_class_{class_id}"] = float(score)

    test_metrics_path = results_dir / "test_metrics.csv"
    pd.DataFrame([test_row]).to_csv(test_metrics_path, index=False)

    model_path = results_dir / "final_model.pt"
    plain_state_dict = clean_state_dict_for_plain_model(server.state_dict())
    torch.save(plain_state_dict, model_path)

    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    print("\nFinal test evaluation:")
    print(f"Best round: {best_round}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test non-finite batches: {test_metrics['non_finite_batches']}")
    print(f"DP epsilon: {epsilon}")
    print(f"DP delta: {delta}")
    print(f"DP sampling rate: {sampling_rate}")
    print("Privacy accountant: AIJack GeneralMomentAccountant")

    for class_id, score in enumerate(per_class_f1):
        print(f"Test F1 class {class_id}: {score:.4f} ({class_names[class_id]})")

    print(f"\nSaved config to: {config_path}")
    print(f"Saved history to: {history_path}")
    print(f"Saved test metrics to: {test_metrics_path}")
    print(f"Saved client distributions to: {client_distribution_path}")
    print(f"Saved final model to: {model_path}")
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()