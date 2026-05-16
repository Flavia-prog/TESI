from __future__ import annotations

import copy
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
from aijack.defense import GeneralMomentAccountant
from aijack.defense.dp.manager import DPSGDClientManager, DPSGDManager
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from thesis.data import create_client_dataloaders, get_class_names, load_dataset_splits
from thesis.experiments.config import DPTrainConfig
from thesis.models import get_model
from thesis.utils import (
    clean_state_dict_for_plain_model,
    collect_provenance,
    ensure_dir,
    resolve_device,
    save_json,
    save_yaml,
    set_seed,
    timestamp,
)


def _evaluate_model(model, data_loader: DataLoader, device: torch.device, num_classes: int):
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
            labels=list(range(num_classes)),
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


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
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


def _state_dict_cpu(model) -> dict[str, torch.Tensor]:
    return copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})


def _move_state_dict_to_device(state_dict, device: torch.device):
    return {k: v.to(device) for k, v in state_dict.items()}


def _make_optimizer_grad_safe(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
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


def _compute_epsilon(noise_multiplier: float, sampling_rate: float, iterations: int, delta: float) -> float | None:
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


def train_fedavg_dp(cfg: DPTrainConfig) -> dict[str, Any]:
    cfg_dict = asdict(cfg)

    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))

    experiment_name = cfg.experiment_name or f"fedavg_{cfg.dataset}_dp_{timestamp()}"

    results_root = ensure_dir(Path(cfg.results_root))
    results_dir = ensure_dir(results_root / experiment_name)

    train_dataset, val_dataset, test_dataset, info, n_channels, num_classes = load_dataset_splits(
        cfg.dataset,
        cfg.data_dir,
    )

    class_names = get_class_names(info)

    local_dataloaders, client_distribution_df = create_client_dataloaders(
        train_dataset=train_dataset,
        num_clients=int(cfg.num_clients),
        batch_size=int(cfg.batch_size),
        seed=int(cfg.seed),
        split_type=str(cfg.split_type),
        alpha=float(cfg.alpha),
        num_classes=num_classes,
    )

    val_loader = DataLoader(val_dataset, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    if cfg.dp_noise_std is not None:
        effective_noise_multiplier = float(cfg.dp_noise_std) / float(cfg.clip_norm)
    else:
        effective_noise_multiplier = float(cfg.noise_multiplier)

    if cfg.dp_enabled:
        actual_noise_std = float(cfg.clip_norm) * effective_noise_multiplier
    else:
        effective_noise_multiplier = 0.0
        actual_noise_std = 0.0

    clients = []
    local_optimizers = []

    for client_id, local_loader in enumerate(local_dataloaders):
        client_model = get_model(cfg.model_name, n_channels, num_classes).to(device)

        if cfg.dp_enabled:
            local_dataset = local_loader.dataset
            local_dataset_size = len(local_dataset)
            if local_dataset_size <= 0:
                raise ValueError(f"Client {client_id} has an empty local dataset.")

            client_lot_size = min(int(cfg.batch_size), local_dataset_size)
            client_batch_size = min(int(cfg.batch_size), client_lot_size)
            iterations = max(1, int(math.ceil(local_dataset_size / client_lot_size)))

            accountant = GeneralMomentAccountant(noise_type="Gaussian", backend="python")
            privacy_manager = DPSGDManager(
                accountant,
                torch.optim.SGD,
                l2_norm_clip=float(cfg.clip_norm),
                dataset=local_dataset,
                lot_size=client_lot_size,
                batch_size=client_batch_size,
                iterations=iterations,
            )

            dp_client_manager = DPSGDClientManager(privacy_manager, float(effective_noise_multiplier))
            DPSGDFedAVGClient, DPOptimizerWrapper = dp_client_manager.attach(FedAVGClient)

            client = DPSGDFedAVGClient(client_model, user_id=client_id)
            optimizer = DPOptimizerWrapper(client.parameters(), lr=float(cfg.lr))
            optimizer = _make_optimizer_grad_safe(optimizer)
        else:
            client = FedAVGClient(client_model, user_id=client_id)
            optimizer = torch.optim.SGD(client.parameters(), lr=float(cfg.lr))

        clients.append(client)
        local_optimizers.append(optimizer)

    server = FedAVGServer(clients, get_model(cfg.model_name, n_channels, num_classes).to(device))

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
        val_metrics = _evaluate_model(api.server, val_loader, device, num_classes)
        history_rows.append(
            {
                "round": round_idx,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "non_finite_batches": val_metrics["non_finite_batches"],
            }
        )

        if math.isfinite(val_metrics["loss"]) and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_round = round_idx
            best_state_dict = _state_dict_cpu(api.server)

    api_kwargs = dict(
        server=server,
        clients=clients,
        criterion=criterion,
        local_optimizers=local_optimizers,
        local_dataloaders=local_dataloaders,
        num_communication=int(cfg.num_rounds),
        local_epoch=int(cfg.local_epochs),
        custom_action=custom_action,
    )
    try:
        fedavg_api = FedAVGAPI(device=device, **api_kwargs)
    except TypeError:
        fedavg_api = FedAVGAPI(**api_kwargs)

    fedavg_api.run()

    if best_state_dict is not None:
        server.load_state_dict(_move_state_dict_to_device(best_state_dict, device))

    test_metrics = _evaluate_model(server, test_loader, device, num_classes)

    clients_per_round = int(cfg.num_clients)
    sampling_rate = clients_per_round / int(cfg.num_clients)
    epsilon = _compute_epsilon(
        noise_multiplier=float(effective_noise_multiplier),
        sampling_rate=float(sampling_rate),
        iterations=int(cfg.num_rounds),
        delta=float(cfg.delta),
    )

    final_config = {
        **cfg_dict,
        "experiment_name": experiment_name,
        "framework": "aijack",
        "algorithm": "fedavg_dp",
        "optimizer": "SGD",
        "device": str(device),
        "effective_noise_multiplier": float(effective_noise_multiplier),
        "actual_noise_std": float(actual_noise_std),
        "sampling_rate": float(sampling_rate),
        "clients_per_round": int(clients_per_round),
        "epsilon": epsilon,
    }

    config_path = results_dir / "config.yaml"
    save_yaml(config_path, final_config)

    per_class_f1 = f1_score(
        test_metrics["targets"],
        test_metrics["preds"],
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(
        test_metrics["targets"],
        test_metrics["preds"],
        labels=list(range(num_classes)),
    )

    history_path = results_dir / "history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    client_distribution_path = results_dir / "client_distributions.csv"
    client_distribution_df.to_csv(client_distribution_path, index=False)

    test_row = {
        "seed": cfg.seed,
        "dataset": cfg.dataset,
        "model_name": cfg.model_name,
        "num_clients": cfg.num_clients,
        "num_rounds": cfg.num_rounds,
        "local_epochs": cfg.local_epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "alpha": cfg.alpha,
        "device": str(device),
        "split_type": cfg.split_type,
        "optimizer": "SGD",
        "dp_enabled": bool(cfg.dp_enabled),
        "clip_norm": float(cfg.clip_norm),
        "noise_multiplier": float(cfg.noise_multiplier),
        "effective_noise_multiplier": float(effective_noise_multiplier),
        "actual_noise_std": float(actual_noise_std),
        "delta": float(cfg.delta),
        "sampling_rate": float(sampling_rate),
        "clients_per_round": int(clients_per_round),
        "epsilon": epsilon,
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
    _plot_confusion_matrix(
        cm,
        class_names,
        cm_path,
        title=f"AIJack FedAvg+DP {cfg.dataset} ({cfg.model_name}) Confusion Matrix",
    )

    provenance_path = results_dir / "provenance.json"
    save_json(
        provenance_path,
        collect_provenance(
            extra={
                "module": "thesis.federated.train_dp",
                "dataset": cfg.dataset,
                "model_name": cfg.model_name,
            }
        ),
    )

    return {
        "experiment_name": experiment_name,
        "results_dir": str(results_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "history_path": str(history_path.resolve()),
        "test_metrics_path": str(test_metrics_path.resolve()),
        "client_distribution_path": str(client_distribution_path.resolve()),
        "model_path": str(model_path.resolve()),
        "confusion_matrix_path": str(cm_path.resolve()),
        "provenance_path": str(provenance_path.resolve()),
        "epsilon": epsilon,
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
    }
