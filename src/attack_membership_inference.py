from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import DEVICE, ensure_dir


def collect_confidence_scores(model, member_dataset, non_member_dataset, batch_size: int = 256):
    model.eval()
    member_loader = DataLoader(member_dataset, batch_size=batch_size, shuffle=False)
    non_member_loader = DataLoader(non_member_dataset, batch_size=batch_size, shuffle=False)

    records = []
    with torch.no_grad():
        for x, _ in member_loader:
            logits = model(x.to(DEVICE))
            probs = torch.softmax(logits, dim=1)
            conf = probs.max(dim=1).values.detach().cpu().numpy()
            for c in conf:
                records.append({"confidence": float(c), "label": 1})

        for x, _ in non_member_loader:
            logits = model(x.to(DEVICE))
            probs = torch.softmax(logits, dim=1)
            conf = probs.max(dim=1).values.detach().cpu().numpy()
            for c in conf:
                records.append({"confidence": float(c), "label": 0})

    return pd.DataFrame(records)


def search_best_threshold(scores_df: pd.DataFrame):
    y_true = scores_df["label"].to_numpy(dtype=np.int32)
    conf = scores_df["confidence"].to_numpy(dtype=np.float32)
    thresholds = np.unique(conf)

    best = {
        "threshold": 0.5,
        "accuracy": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    for threshold in thresholds:
        y_pred = (conf >= threshold).astype(np.int32)
        metrics = compute_classification_metrics(y_true, y_pred)
        if metrics["accuracy"] > best["accuracy"]:
            best = {"threshold": float(threshold), **metrics}

    return best


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = max(tp + tn + fp + fn, 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_confidence_histogram(scores_df: pd.DataFrame, output_path: Path):
    member_scores = scores_df.loc[scores_df["label"] == 1, "confidence"].to_numpy()
    non_member_scores = scores_df.loc[scores_df["label"] == 0, "confidence"].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(member_scores, bins=30, alpha=0.6, label="member", density=True)
    plt.hist(non_member_scores, bins=30, alpha=0.6, label="non-member", density=True)
    plt.xlabel("Max softmax confidence")
    plt.ylabel("Density")
    plt.title("Membership Inference Confidence Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_membership_inference_attack(
    model,
    member_dataset,
    non_member_dataset,
    output_dir: str = "results/membership_inference",
):
    ensure_dir(output_dir)
    out_dir = Path(output_dir)

    scores_df = collect_confidence_scores(model, member_dataset, non_member_dataset)
    scores_df.to_csv(out_dir / "mia_scores.csv", index=False)

    best = search_best_threshold(scores_df)
    summary_df = pd.DataFrame(
        [
            {
                "best_threshold": best["threshold"],
                "attack_accuracy": best["accuracy"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"],
                "num_member_samples": int((scores_df["label"] == 1).sum()),
                "num_non_member_samples": int((scores_df["label"] == 0).sum()),
            }
        ]
    )
    summary_df.to_csv(out_dir / "mia_summary.csv", index=False)
    save_confidence_histogram(scores_df, out_dir / "confidence_histogram.png")
    return summary_df.iloc[0].to_dict()
