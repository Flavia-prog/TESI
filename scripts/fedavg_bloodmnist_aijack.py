from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from fl_shared.medmnist_data import create_client_dataloaders, load_medmnist_splits, resolve_dataset_name
from fl_shared.models import available_model_arches, build_model
from fl_shared.runtime import collect_provenance, load_yaml_config, merge_config, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIJack FedAvg baseline on a MedMNIST dataset")

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--model-arch",
        type=str,
        choices=available_model_arches(),
        default=None,
    )
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)

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

    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    defaults = {
        "dataset": "bloodmnist",
        "model_arch": "cnn",
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
    }

    yaml_config = load_yaml_config(args.config)

    cli_values = {
        "dataset": args.dataset,
        "model_arch": args.model_arch,
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
    }

    resolved = merge_config(defaults, yaml_config, cli_values)
    resolved["dataset"] = resolve_dataset_name(str(resolved["dataset"]))
    resolved["model_arch"] = str(resolved["model_arch"]).lower()

    experiment_name = args.experiment_name
    if experiment_name is None:
        experiment_name = yaml_config.get("experiment_name")
    if not experiment_name:
        experiment_name = f"fedavg_{resolved['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    resolved["experiment_name"] = experiment_name
    return resolved


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device, num_classes: int):
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


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path, title: str) -> None:
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


def get_state_dict_from_model(model: nn.Module):
    return copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})


def move_state_dict_to_device(state_dict, device: torch.device):
    return {k: v.to(device) for k, v in state_dict.items()}


def main() -> None:
    args = parse_args()
    config = resolve_config(args)

    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))

    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)

    results_dir = results_root / str(config["experiment_name"])
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, info, n_channels, num_classes = load_medmnist_splits(
        str(config["dataset"]),
        str(config["data_dir"]),
    )

    class_names = [str(info["label"][str(i)]) for i in range(num_classes)]

    local_dataloaders, client_distribution_df = create_client_dataloaders(
        train_dataset=train_dataset,
        num_clients=int(config["num_clients"]),
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]),
        split_type=str(config["split_type"]),
        alpha=float(config["alpha"]),
        num_classes=num_classes,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    clients = [
        FedAVGClient(
            build_model(str(config["model_arch"]), n_channels, num_classes).to(device),
            user_id=client_id,
        )
        for client_id in range(int(config["num_clients"]))
    ]

    local_optimizers = [
        torch.optim.SGD(client.parameters(), lr=float(config["lr"]))
        for client in clients
    ]

    server = FedAVGServer(
        clients,
        build_model(str(config["model_arch"]), n_channels, num_classes).to(device),
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
            num_classes=num_classes,
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

    fedavg_api = FedAVGAPI(
        server,
        clients,
        criterion,
        local_optimizers,
        local_dataloaders,
        num_communication=int(config["num_rounds"]),
        local_epoch=int(config["local_epochs"]),
        custom_action=custom_action,
    )

    print(f"Using device: {device}")
    print(
        f"Starting AIJack FedAvg {config['dataset']} ({config['model_arch']}) | "
        f"clients={config['num_clients']}, "
        f"rounds={config['num_rounds']}, "
        f"local_epochs={config['local_epochs']}, "
        f"batch_size={config['batch_size']}, "
        f"lr={config['lr']}, "
        f"split={config['split_type']}, "
        f"alpha={config['alpha']}, "
        f"optimizer=SGD"
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
        num_classes=num_classes,
    )

    final_config = {
        "experiment_name": config["experiment_name"],
        "dataset": config["dataset"],
        "model_arch": config["model_arch"],
        "framework": "aijack",
        "algorithm": "fedavg",
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
    }

    config_path = results_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(final_config, f, sort_keys=False)

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
        "seed": config["seed"],
        "dataset": config["dataset"],
        "model_arch": config["model_arch"],
        "num_clients": config["num_clients"],
        "num_rounds": config["num_rounds"],
        "local_epochs": config["local_epochs"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "alpha": config["alpha"],
        "device": str(device),
        "split_type": config["split_type"],
        "optimizer": "SGD",
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
    torch.save(server.state_dict(), model_path)

    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        cm,
        class_names,
        cm_path,
        title=f"AIJack FedAvg {config['dataset']} ({config['model_arch']}) Confusion Matrix",
    )

    provenance_path = results_dir / "provenance.json"
    with provenance_path.open("w", encoding="utf-8") as f:
        provenance_payload = collect_provenance(
            extra={
                "script": "fedavg_bloodmnist_aijack.py",
                "dataset": config["dataset"],
                "model_arch": config["model_arch"],
            }
        )
        json.dump(provenance_payload, f, indent=2)

    print("\nFinal test evaluation:")
    print(f"Best round: {best_round}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test non-finite batches: {test_metrics['non_finite_batches']}")

    for class_id, score in enumerate(per_class_f1):
        print(f"Test F1 class {class_id}: {score:.4f} ({class_names[class_id]})")

    print(f"\nSaved config to: {config_path}")
    print(f"Saved history to: {history_path}")
    print(f"Saved test metrics to: {test_metrics_path}")
    print(f"Saved client distributions to: {client_distribution_path}")
    print(f"Saved final model to: {model_path}")
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved provenance to: {provenance_path}")


if __name__ == "__main__":
    main()
