import argparse
import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
from medmnist import INFO
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


class BloodMNISTCNN(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIJack FedAvg baseline on BloodMNIST")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--split-type", type=str, choices=["iid"], default="iid")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = data_class(split="train", transform=transform, download=True, root=data_dir)
    val_dataset = data_class(split="val", transform=transform, download=True, root=data_dir)
    test_dataset = data_class(split="test", transform=transform, download=True, root=data_dir)
    return train_dataset, val_dataset, test_dataset, info


def iid_split_indices(n_samples: int, num_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)
    return np.array_split(shuffled, num_clients)


def create_client_dataloaders(
    train_dataset,
    num_clients: int,
    batch_size: int,
    seed: int,
):
    split_indices = iid_split_indices(len(train_dataset), num_clients, seed)
    loaders = []
    distributions = []

    for client_id, client_indices in enumerate(split_indices):
        subset = Subset(train_dataset, client_indices.tolist())

        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed + client_id)

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=loader_generator,
        )
        loaders.append(loader)

        labels = np.array(train_dataset.labels).reshape(-1)
        local_labels = labels[client_indices]
        unique, counts = np.unique(local_labels, return_counts=True)
        class_count_map = {int(k): int(v) for k, v in zip(unique, counts)}

        row = {"client_id": client_id, "num_samples": int(len(client_indices))}
        for class_id in range(8):
            row[f"class_{class_id}_count"] = class_count_map.get(class_id, 0)
        distributions.append(row)

    return loaders, pd.DataFrame(distributions)


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            targets = labels.squeeze(1).long().to(device)

            outputs = model(images)
            loss = F.nll_loss(outputs, targets)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(data_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "targets": np.array(all_targets),
        "preds": np.array(all_preds),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="FedAvg BloodMNIST Confusion Matrix",
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    device_str = str(device)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, info = load_bloodmnist(args.data_dir)
    class_names = [info["label"][str(i)] for i in range(len(info["label"]))]

    local_dataloaders, client_dist_df = create_client_dataloaders(
        train_dataset=train_dataset,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    global_model = BloodMNISTCNN(num_classes=8).to(device)

    clients = [
        FedAVGClient(
            model=BloodMNISTCNN(num_classes=8).to(device),
            user_id=i,
            lr=args.lr,
            server_side_update=True,
            device=device_str,
        )
        for i in range(args.num_clients)
    ]

    local_optimizers = [torch.optim.Adam(client.parameters(), lr=args.lr) for client in clients]

    server = FedAVGServer(
        clients=clients,
        global_model=global_model,
        lr=args.lr,
        optimizer_type="adam",
        server_side_update=True,
        device=device_str,
    )

    history_rows = []
    best_val_loss = float("inf")
    best_round = -1
    best_state_dict = copy.deepcopy(server.server_model.state_dict())

    def criterion(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # MedMNIST returns labels with shape [batch, 1], so we squeeze to [batch].
        return F.nll_loss(outputs, labels.squeeze(1).long())

    def custom_action(api: FedAVGAPI) -> None:
        nonlocal best_val_loss, best_round, best_state_dict
        round_idx = len(history_rows) + 1

        val_metrics = evaluate_model(api.server.server_model, val_loader, device)
        history_rows.append(
            {
                "round": round_idx,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        print(
            f"Round {round_idx:03d}/{args.num_rounds} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_acc: {val_metrics['accuracy']:.4f} | "
            f"val_macro_f1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_round = round_idx
            best_state_dict = copy.deepcopy(api.server.server_model.state_dict())

    fedavg_api = FedAVGAPI(
        server=server,
        clients=clients,
        criterion=criterion,
        local_optimizers=local_optimizers,
        local_dataloaders=local_dataloaders,
        num_communication=args.num_rounds,
        local_epoch=args.local_epochs,
        use_gradients=True,
        custom_action=custom_action,
        device=device_str,
    )

    print(f"Using device: {device_str}")
    print(
        f"Starting FedAvg with {args.num_clients} clients, "
        f"{args.num_rounds} rounds, local_epochs={args.local_epochs}"
    )

    fedavg_api.run()

    # Evaluate and save the best global model from validation tracking.
    server.server_model.load_state_dict(best_state_dict)
    test_metrics = evaluate_model(server.server_model, test_loader, device)

    per_class_f1 = f1_score(test_metrics["targets"], test_metrics["preds"], average=None)
    cm = confusion_matrix(test_metrics["targets"], test_metrics["preds"])

    history_path = results_dir / "fedavg_bloodmnist_aijack_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    dist_path = results_dir / "fedavg_bloodmnist_aijack_client_distributions.csv"
    client_dist_df.to_csv(dist_path, index=False)

    test_row = {
        "seed": args.seed,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": device_str,
        "split_type": args.split_type,
        "best_round": best_round,
        "best_val_loss": float(best_val_loss),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }

    for class_id, score in enumerate(per_class_f1):
        test_row[f"test_f1_class_{class_id}"] = float(score)

    test_metrics_path = results_dir / "fedavg_bloodmnist_aijack_test_metrics.csv"
    pd.DataFrame([test_row]).to_csv(test_metrics_path, index=False)

    model_path = results_dir / "fedavg_bloodmnist_aijack_final_model.pt"
    torch.save(server.server_model.state_dict(), model_path)

    cm_path = results_dir / "fedavg_bloodmnist_aijack_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    print("\nFinal test evaluation (best validation round model):")
    print(f"Best round: {best_round}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    for class_id, score in enumerate(per_class_f1):
        print(f"Test F1 class {class_id}: {score:.4f} ({class_names[class_id]})")

    print(f"\nSaved history to: {history_path}")
    print(f"Saved test metrics to: {test_metrics_path}")
    print(f"Saved client distributions to: {dist_path}")
    print(f"Saved final model to: {model_path}")
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()
