import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer

from src.attack_gradient_inversion import run_gradient_inversion_demo
from src.data import get_mnist, get_train_subset, make_loaders, split_iid
from src.model import SmallCNN
from src.utils import DEVICE, ensure_dir, evaluate_accuracy, set_seed

RESULTS_DIR = Path("results")
GRADIENT_INVERSION_DIR = RESULTS_DIR / "gradient_inversion"


def save_batch_size_vs_mse_plot(summary_df, output_path: str):
    width = 900
    height = 500
    pad = 60
    canvas = np.ones((height, width), dtype=np.float32)

    xs = summary_df["batch_size"].to_numpy(dtype=np.float32)
    series = {
        "avg": summary_df["avg_mse"].to_numpy(dtype=np.float32),
        "best": summary_df["best_mse"].to_numpy(dtype=np.float32),
        "worst": summary_df["worst_mse"].to_numpy(dtype=np.float32),
    }
    y_min = float(min(v.min() for v in series.values()))
    y_max = float(max(v.max() for v in series.values()))
    y_span = max(y_max - y_min, 1e-6)

    def to_px(x_val, y_val):
        x_norm = (x_val - xs.min()) / max(xs.max() - xs.min(), 1e-6)
        y_norm = (y_val - y_min) / y_span
        px = int(pad + x_norm * (width - 2 * pad))
        py = int(height - pad - y_norm * (height - 2 * pad))
        return px, py

    def draw_point(x, y, value):
        r = 4
        y0 = max(0, y - r)
        y1 = min(height, y + r + 1)
        x0 = max(0, x - r)
        x1 = min(width, x + r + 1)
        canvas[y0:y1, x0:x1] = value

    def draw_line(x0, y0, x1, y1, value):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for t in np.linspace(0.0, 1.0, n):
            x = int(round(x0 + t * (x1 - x0)))
            y = int(round(y0 + t * (y1 - y0)))
            if 0 <= x < width and 0 <= y < height:
                canvas[y, x] = value

    for x in range(pad, width - pad):
        canvas[height - pad, x] = 0.85
    for y in range(pad, height - pad):
        canvas[y, pad] = 0.85

    shades = {"avg": 0.15, "best": 0.4, "worst": 0.65}
    for key, yvals in series.items():
        points = [to_px(xs[i], yvals[i]) for i in range(len(xs))]
        for i in range(len(points) - 1):
            draw_line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], shades[key])
        for x, y in points:
            draw_point(x, y, shades[key])

    from matplotlib import pyplot as plt

    plt.imsave(output_path, canvas, cmap="gray")


def save_batch_size_vs_single_metric_plot(summary_df, metric_col: str, output_path: str):
    width = 900
    height = 500
    pad = 60
    canvas = np.ones((height, width), dtype=np.float32)

    xs = summary_df["batch_size"].to_numpy(dtype=np.float32)
    ys = summary_df[metric_col].to_numpy(dtype=np.float32)
    y_min = float(ys.min())
    y_max = float(ys.max())
    y_span = max(y_max - y_min, 1e-6)

    def to_px(x_val, y_val):
        x_norm = (x_val - xs.min()) / max(xs.max() - xs.min(), 1e-6)
        y_norm = (y_val - y_min) / y_span
        px = int(pad + x_norm * (width - 2 * pad))
        py = int(height - pad - y_norm * (height - 2 * pad))
        return px, py

    def draw_point(x, y, value=0.2):
        r = 5
        y0 = max(0, y - r)
        y1 = min(height, y + r + 1)
        x0 = max(0, x - r)
        x1 = min(width, x + r + 1)
        canvas[y0:y1, x0:x1] = value

    def draw_line(x0, y0, x1, y1, value=0.2):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for t in np.linspace(0.0, 1.0, n):
            x = int(round(x0 + t * (x1 - x0)))
            y = int(round(y0 + t * (y1 - y0)))
            if 0 <= x < width and 0 <= y < height:
                canvas[y, x] = value

    for x in range(pad, width - pad):
        canvas[height - pad, x] = 0.85
    for y in range(pad, height - pad):
        canvas[y, pad] = 0.85

    points = [to_px(xs[i], ys[i]) for i in range(len(xs))]
    for i in range(len(points) - 1):
        draw_line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
    for x, y in points:
        draw_point(x, y)

    from matplotlib import pyplot as plt

    plt.imsave(output_path, canvas, cmap="gray")


def recompute_batch_size_summary_from_existing_results(
    root_dir: str = "results/gradient_inversion", batch_sizes=(1, 4, 8)
):
    root = Path(root_dir)
    baseline_df = pd.read_csv(RESULTS_DIR / "baseline_metrics.csv")
    final_acc_map = (
        baseline_df.sort_values(["batch_size", "round"])
        .groupby("batch_size", as_index=False)
        .last()[["batch_size", "test_accuracy"]]
        .set_index("batch_size")["test_accuracy"]
        .to_dict()
    )

    rows = []
    for batch_size in batch_sizes:
        metrics_path = root / f"batch_size_{batch_size}" / "attack_metrics.csv"
        attack_df = pd.read_csv(metrics_path)
        mse = attack_df["mse"]
        rows.append(
            {
                "batch_size": int(batch_size),
                "avg_mse": float(mse.mean()),
                "median_mse": float(mse.median()),
                "std_mse": float(mse.std(ddof=0)),
                "best_mse": float(mse.min()),
                "worst_mse": float(mse.max()),
                "final_test_accuracy": float(final_acc_map[int(batch_size)]),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("batch_size")
    summary_path = root / "batch_size_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    save_batch_size_vs_mse_plot(summary_df, root / "batch_size_vs_mse.png")
    save_batch_size_vs_single_metric_plot(
        summary_df, "median_mse", root / "batch_size_vs_median_mse.png"
    )
    return summary_df


def run_baseline_and_attack():
    set_seed(0)

    train_dataset, test_dataset = get_mnist("./data")
    train_subset = get_train_subset(train_dataset, max_samples=2000, seed=0)
    client_datasets = split_iid(train_subset, num_clients=2, seed=0)
    _, test_loader = make_loaders(client_datasets, test_dataset, batch_size=64)

    ensure_dir(str(RESULTS_DIR))
    ensure_dir(str(GRADIENT_INVERSION_DIR))

    batch_sizes = [1, 4, 8]
    summary_rows = []
    baseline_rows = []

    for batch_size in batch_sizes:
        print(f"\n=== Running batch size {batch_size} ===")
        client_loaders, _ = make_loaders(client_datasets, test_dataset, batch_size=batch_size)

        global_model = SmallCNN().to(DEVICE)
        clients = [
            FedAVGClient(copy.deepcopy(global_model), user_id=i, lr=0.1, device=DEVICE)
            for i in range(2)
        ]
        server = FedAVGServer(clients, copy.deepcopy(global_model), device=DEVICE)
        criterion = torch.nn.CrossEntropyLoss()
        local_optimizers = [torch.optim.SGD(client.parameters(), lr=0.1) for client in clients]

        final_acc = 0.0
        for rnd in range(1, 4):
            api = FedAVGAPI(
                server=server,
                clients=clients,
                criterion=criterion,
                local_optimizers=local_optimizers,
                local_dataloaders=client_loaders,
                num_communication=1,
                local_epoch=1,
                use_gradients=True,
                device=DEVICE,
            )
            api.run()

            final_acc = evaluate_accuracy(server.server_model, test_loader, DEVICE)
            print(f"Batch {batch_size} - Round {rnd} test accuracy: {final_acc:.4f}")
            baseline_rows.append(
                {"batch_size": batch_size, "round": rnd, "test_accuracy": final_acc}
            )

        attack_output_dir = GRADIENT_INVERSION_DIR / f"batch_size_{batch_size}"
        attack_summary = run_gradient_inversion_demo(
            clients[0],
            client_datasets[0],
            output_dir=str(attack_output_dir),
            num_attacks=10,
            attack_batch_size=batch_size,
            attack_iterations=60,
        )

        summary_rows.append(
            {
                "batch_size": batch_size,
                "avg_mse": attack_summary["avg_mse"],
                "best_mse": attack_summary["best_mse"],
                "worst_mse": attack_summary["worst_mse"],
                "final_test_accuracy": final_acc,
            }
        )
        print(
            f"Batch {batch_size} summary: avg_mse={attack_summary['avg_mse']:.6f}, "
            f"best_mse={attack_summary['best_mse']:.6f}, "
            f"worst_mse={attack_summary['worst_mse']:.6f}, "
            f"final_test_accuracy={final_acc:.4f}"
        )

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("batch_size")
    summary_csv = GRADIENT_INVERSION_DIR / "batch_size_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_df = recompute_batch_size_summary_from_existing_results(
        str(GRADIENT_INVERSION_DIR), batch_sizes=batch_sizes
    )

    print("\n=== Final Batch Size Summary ===")
    for _, row in summary_df.iterrows():
        print(
            f"batch_size={int(row['batch_size'])} | avg_mse={row['avg_mse']:.6f} | "
            f"median_mse={row['median_mse']:.6f} | std_mse={row['std_mse']:.6f} | "
            f"best_mse={row['best_mse']:.6f} | worst_mse={row['worst_mse']:.6f} | "
            f"final_test_accuracy={row['final_test_accuracy']:.4f}"
        )


if __name__ == "__main__":
    run_baseline_and_attack()
