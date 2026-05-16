from __future__ import annotations

import medmnist
import numpy as np
import pandas as pd
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def resolve_dataset_name(dataset: str) -> str:
    name = str(dataset).strip().lower()
    if name not in INFO:
        raise ValueError(f"Unsupported MedMNIST dataset: {name}")

    if INFO[name].get("task") != "multi-class":
        raise ValueError(
            f"Only multi-class MedMNIST tasks are currently supported. Got: {INFO[name].get('task')}"
        )

    return name


def get_dataset_meta(dataset: str) -> tuple[dict, int, int]:
    dataset_name = resolve_dataset_name(dataset)
    info = INFO[dataset_name]
    n_channels = int(info.get("n_channels", 1))
    num_classes = len(info["label"])
    return info, n_channels, num_classes


def build_transform(n_channels: int):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels),
        ]
    )


def load_medmnist_splits(dataset: str, data_dir: str):
    dataset_name = resolve_dataset_name(dataset)
    info, n_channels, num_classes = get_dataset_meta(dataset_name)
    data_class = getattr(medmnist, info["python_class"])
    transform = build_transform(n_channels)

    train_dataset = data_class(split="train", transform=transform, download=True, root=data_dir)
    val_dataset = data_class(split="val", transform=transform, download=True, root=data_dir)
    test_dataset = data_class(split="test", transform=transform, download=True, root=data_dir)

    return train_dataset, val_dataset, test_dataset, info, n_channels, num_classes


def load_medmnist_train(dataset: str, data_dir: str):
    dataset_name = resolve_dataset_name(dataset)
    info, n_channels, num_classes = get_dataset_meta(dataset_name)
    data_class = getattr(medmnist, info["python_class"])
    transform = build_transform(n_channels)
    train_dataset = data_class(split="train", transform=transform, download=True, root=data_dir)
    return train_dataset, n_channels, num_classes


def iid_split_indices(n_samples: int, num_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return [arr.astype(int) for arr in np.array_split(indices, num_clients)]


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


def build_client_indices(
    train_dataset,
    split_type: str,
    num_clients: int,
    seed: int,
    alpha: float,
    num_classes: int,
) -> list[np.ndarray]:
    labels = np.array(train_dataset.labels).reshape(-1)
    split_type_normalized = split_type.lower()

    if split_type_normalized == "iid":
        return iid_split_indices(
            n_samples=len(train_dataset),
            num_clients=num_clients,
            seed=seed,
        )

    if split_type_normalized == "dirichlet":
        return dirichlet_split_indices(
            labels=labels,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            num_classes=num_classes,
        )

    raise ValueError(f"Unsupported split_type: {split_type}")


def create_client_dataloaders(
    train_dataset,
    num_clients: int,
    batch_size: int,
    seed: int,
    split_type: str,
    alpha: float,
    num_classes: int,
):
    labels = np.array(train_dataset.labels).reshape(-1)
    split_indices = build_client_indices(
        train_dataset=train_dataset,
        split_type=split_type,
        num_clients=num_clients,
        seed=seed,
        alpha=alpha,
        num_classes=num_classes,
    )

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
        for class_id in range(num_classes):
            row[f"class_{class_id}_count"] = class_count_map.get(class_id, 0)
        distribution_rows.append(row)

    return loaders, pd.DataFrame(distribution_rows)
