"""
Run Block A from BLOODMNIST_FRONTIER_EXPERIMENT_MATRIX.md.

Block A evaluates the existing BloodMNIST sigma frontier:

- splits: IID, Dirichlet alpha 1.0, 0.5, 0.1
- sigmas: 0, 0.25, 0.5, 0.75, 1.0, 2.0
- fixed attacker from configs/current/attack_protocols/bloodmnist_fixed_attacker_v1.yaml

The script delegates attack execution to run_bloodmnist_fixed_attacker_eval.py,
then creates Block-A-specific plots and a short report from the resulting
model_privacy_utility_table.csv.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FIXED_EVAL_SCRIPT = SCRIPT_DIR / "run_bloodmnist_fixed_attacker_eval.py"

DEFAULT_PROTOCOL = (
    REPO_ROOT / "configs/current/attack_protocols/bloodmnist_fixed_attacker_v1.yaml"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results/current/privacy_utility/bloodmnist_block_a_frontier_v1"
)

BLOCK_A_SPLITS = ["iid", "noniid_alpha_1", "noniid_alpha_05", "noniid_alpha_01"]
BLOCK_A_SIGMAS = [0.25, 0.5, 0.75, 1.0, 2.0]

SPLIT_ORDER = ["iid", "noniid_alpha_1", "noniid_alpha_05", "noniid_alpha_01"]
SPLIT_COLORS = {
    "iid": "#1f77b4",
    "noniid_alpha_1": "#2ca02c",
    "noniid_alpha_05": "#ff7f0e",
    "noniid_alpha_01": "#d62728",
}
SPLIT_LABELS = {
    "iid": "IID",
    "noniid_alpha_1": "Dirichlet alpha=1.0",
    "noniid_alpha_05": "Dirichlet alpha=0.5",
    "noniid_alpha_01": "Dirichlet alpha=0.1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BloodMNIST Block A fixed-attacker sigma frontier."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="Override protocol device.")
    parser.add_argument("--jobs", type=int, default=None, help="Override protocol jobs.")
    parser.add_argument("--rerun-attacks", action="store_true")
    parser.add_argument("--skip-attacks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing-models",
        action="store_true",
        help="Continue even if expected Block A training artifacts are missing.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Protocol not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in protocol YAML: {path}")
    return loaded


def one_value(protocol: dict[str, Any], key: str) -> Any:
    values = protocol.get(key)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"Protocol field {key!r} must be a one-element list.")
    return values[0]


def protocol_values(protocol: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "client_ids": protocol.get("client_ids", [0, 1, 2]),
        "sample_indices": protocol.get("sample_indices", [0, 25, 50]),
        "attack_batch_size": int(one_value(protocol, "attack_batch_sizes")),
        "attack_iters": int(one_value(protocol, "attack_iters")),
        "num_trials": int(one_value(protocol, "num_trials")),
        "attack_lr": float(one_value(protocol, "attack_lrs")),
        "distance": str(one_value(protocol, "distances")),
        "device": args.device or protocol.get("device", "cpu"),
        "jobs": int(args.jobs if args.jobs is not None else protocol.get("jobs", 1)),
    }
    if values["distance"] not in {"l2", "cossim"}:
        raise ValueError(f"Unsupported distance: {values['distance']}")
    return values


def sigma_tag(sigma: float) -> str:
    text = f"{sigma:g}"
    return re.sub(r"[^0-9A-Za-z]+", "", text)


def expected_training_dirs() -> list[Path]:
    baseline_root = REPO_ROOT / "results/current/training/bloodmnist/baselines"
    dp_root = REPO_ROOT / "results/current/training/bloodmnist/dp_matrix"
    prefixes = {
        "iid": "iid",
        "noniid_alpha_1": "noniid_alpha_1",
        "noniid_alpha_05": "noniid_alpha_05",
        "noniid_alpha_01": "noniid_alpha_01",
    }
    baseline_names = {
        "iid": "iid_baseline",
        "noniid_alpha_1": "noniid_alpha_1",
        "noniid_alpha_05": "noniid_alpha_05",
        "noniid_alpha_01": "noniid_alpha_01",
    }

    dirs: list[Path] = []
    for split in BLOCK_A_SPLITS:
        dirs.append(baseline_root / baseline_names[split])
        for sigma in BLOCK_A_SIGMAS:
            dirs.append(dp_root / f"{prefixes[split]}_dp_sigma_{sigma_tag(sigma)}")
    return dirs


def validate_training_artifacts(allow_missing: bool) -> None:
    required_files = ["config.yaml", "test_metrics.csv", "final_model.pt"]
    missing: list[str] = []
    for experiment_dir in expected_training_dirs():
        for filename in required_files:
            path = experiment_dir / filename
            if not path.exists():
                missing.append(str(path.relative_to(REPO_ROOT)))

    if not missing:
        return

    message = (
        "Missing expected Block A training artifacts:\n"
        + "\n".join(f"- {path}" for path in missing)
    )
    if allow_missing:
        print(message)
        return
    raise FileNotFoundError(message)


def run_fixed_attacker(
    args: argparse.Namespace,
    values: dict[str, Any],
) -> None:
    command = [
        sys.executable,
        str(FIXED_EVAL_SCRIPT),
        "--splits",
        *BLOCK_A_SPLITS,
        "--sigmas",
        *[str(sigma) for sigma in BLOCK_A_SIGMAS],
        "--client-ids",
        *[str(client_id) for client_id in values["client_ids"]],
        "--sample-indices",
        *[str(sample_index) for sample_index in values["sample_indices"]],
        "--attack-batch-size",
        str(values["attack_batch_size"]),
        "--attack-iters",
        str(values["attack_iters"]),
        "--num-trials",
        str(values["num_trials"]),
        "--attack-lr",
        str(values["attack_lr"]),
        "--distance",
        values["distance"],
        "--device",
        values["device"],
        "--jobs",
        str(values["jobs"]),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.rerun_attacks:
        command.append("--rerun")
    if args.dry_run:
        command.append("--dry-run")

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Fixed-attacker Block A evaluation failed with "
            f"return code {completed.returncode}."
        )


def load_model_table(output_dir: Path) -> pd.DataFrame:
    table_path = output_dir / "model_privacy_utility_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing model table: {table_path}")
    df = pd.read_csv(table_path)
    numeric_cols = [
        "alpha",
        "sigma",
        "test_accuracy",
        "test_macro_f1",
        "attack_success_rate",
        "median_mse",
        "median_leakage_score",
        "worst_mse",
        "worst_leakage_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def split_order(df: pd.DataFrame) -> list[str]:
    present = set(df["split_label"].dropna().unique())
    ordered = [split for split in SPLIT_ORDER if split in present]
    return ordered + sorted(present - set(ordered))


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_sigma(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in split_order(df):
        sub = df[df["split_label"] == split].sort_values("sigma")
        ax.plot(
            sub["sigma"],
            sub[y_col],
            marker="o",
            linewidth=1.8,
            color=SPLIT_COLORS.get(split, "black"),
            label=SPLIT_LABELS.get(split, split),
        )
    ax.set_xlabel("DP noise multiplier sigma")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(title="Split", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_frontier(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    output_base: Path,
    log_x: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in split_order(df):
        sub = df[df["split_label"] == split].sort_values("sigma")
        color = SPLIT_COLORS.get(split, "black")
        ax.plot(
            sub[x_col],
            sub["test_macro_f1"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=SPLIT_LABELS.get(split, split),
        )
        for _, row in sub.iterrows():
            if pd.notna(row[x_col]) and pd.notna(row["test_macro_f1"]):
                ax.annotate(
                    f"s={row['sigma']:g}",
                    xy=(row[x_col], row["test_macro_f1"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    alpha=0.85,
                )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Test macro-F1")
    ax.set_title(title)
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(title="Split", fontsize=9)
    if log_x:
        ax.set_xscale("log")
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_mse_frontier_with_attack_status(
    df: pd.DataFrame,
    title: str,
    output_base: Path,
) -> None:
    fig, (ax_mse, ax_status) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    for split in split_order(df):
        sub = df[df["split_label"] == split].sort_values("sigma")
        color = SPLIT_COLORS.get(split, "black")
        label = SPLIT_LABELS.get(split, split)

        valid = sub.dropna(subset=["median_mse", "test_macro_f1"])
        if not valid.empty:
            ax_mse.plot(
                valid["median_mse"],
                valid["test_macro_f1"],
                marker="o",
                linewidth=1.8,
                color=color,
                label=label,
            )
            for _, row in valid.iterrows():
                ax_mse.annotate(
                    f"s={row['sigma']:g}",
                    xy=(row["median_mse"], row["test_macro_f1"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    alpha=0.85,
                )

        ax_status.plot(
            sub["sigma"],
            sub["test_macro_f1"],
            linewidth=1.1,
            color=color,
            alpha=0.35,
        )
        has_mse = sub["median_mse"].notna() & sub["test_macro_f1"].notna()
        no_mse = sub["median_mse"].isna() & sub["test_macro_f1"].notna()
        ax_status.scatter(
            sub.loc[has_mse, "sigma"],
            sub.loc[has_mse, "test_macro_f1"],
            marker="o",
            s=34,
            color=color,
        )
        ax_status.scatter(
            sub.loc[no_mse, "sigma"],
            sub.loc[no_mse, "test_macro_f1"],
            marker="x",
            s=52,
            linewidths=1.8,
            color=color,
        )

    ax_mse.set_xlabel("Median reconstruction MSE, successful cells only")
    ax_mse.set_ylabel("Test macro-F1")
    ax_mse.set_title("Measured MSE frontier")
    ax_mse.set_xscale("log")
    ax_mse.grid(linestyle="--", alpha=0.35)
    ax_mse.legend(title="Split", fontsize=9)

    ax_status.set_xlabel("DP noise multiplier sigma")
    ax_status.set_ylabel("Test macro-F1")
    ax_status.set_title("All models and attack status")
    ax_status.grid(linestyle="--", alpha=0.35)
    status_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="None",
            label="Median MSE available",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            linestyle="None",
            markersize=7,
            label="No positive MSE",
        ),
    ]
    ax_status.legend(handles=status_handles, fontsize=9, loc="best")

    fig.suptitle(title)
    fig.text(
        0.5,
        0.01,
        "Left panel excludes no-MSE cells by definition; right panel shows every trained model.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_figure(fig, output_base)


def add_baseline_drops(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["macro_f1_drop_from_baseline"] = math.nan
    enriched["accuracy_drop_from_baseline"] = math.nan
    for split in split_order(enriched):
        mask = enriched["split_label"] == split
        baseline = enriched[mask & enriched["sigma"].eq(0.0)]
        if baseline.empty:
            continue
        base_f1 = baseline["test_macro_f1"].iloc[0]
        base_acc = baseline["test_accuracy"].iloc[0]
        enriched.loc[mask, "macro_f1_drop_from_baseline"] = (
            base_f1 - enriched.loc[mask, "test_macro_f1"]
        )
        enriched.loc[mask, "accuracy_drop_from_baseline"] = (
            base_acc - enriched.loc[mask, "test_accuracy"]
        )
    return enriched


def frontier_candidates(df: pd.DataFrame) -> pd.DataFrame:
    candidates = df.copy()
    candidates["within_5pp_macro_f1"] = candidates["macro_f1_drop_from_baseline"].le(0.05)
    candidates["within_10pp_macro_f1"] = candidates["macro_f1_drop_from_baseline"].le(0.10)
    candidates["within_15pp_macro_f1"] = candidates["macro_f1_drop_from_baseline"].le(0.15)
    return candidates


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def write_block_report(
    output_dir: Path,
    table: pd.DataFrame,
    protocol_path: Path,
    values: dict[str, Any],
) -> None:
    report_cols = [
        "experiment_name",
        "split_label",
        "sigma",
        "test_accuracy",
        "test_macro_f1",
        "macro_f1_drop_from_baseline",
        "attack_success_rate",
        "median_mse",
        "median_leakage_score",
        "worst_mse",
        "worst_leakage_score",
        "n_failed_or_no_mse",
        "within_5pp_macro_f1",
        "within_10pp_macro_f1",
        "within_15pp_macro_f1",
    ]
    available_cols = [col for col in report_cols if col in table.columns]
    total_cells = int(table["n_attack_cells"].sum()) if "n_attack_cells" in table else 0
    positive_cells = int(table["n_positive_mse"].sum()) if "n_positive_mse" in table else 0
    failed_cells = (
        int(table["n_failed_or_no_mse"].sum()) if "n_failed_or_no_mse" in table else 0
    )

    lines = [
        "# BloodMNIST Block A Frontier Report",
        "",
        f"- Generated at: {datetime.now().isoformat()}",
        f"- Protocol: `{protocol_path}`",
        f"- Splits: {', '.join(BLOCK_A_SPLITS)}",
        "- Sigmas: 0, " + ", ".join(str(sigma) for sigma in BLOCK_A_SIGMAS),
        (
            "- Fixed attacker: "
            f"clients={values['client_ids']}, samples={values['sample_indices']}, "
            f"batch={values['attack_batch_size']}, iters={values['attack_iters']}, "
            f"trials={values['num_trials']}, lr={values['attack_lr']}, "
            f"distance={values['distance']}"
        ),
        f"- Model rows: {len(table)}",
        f"- Attack cells retained: {total_cells}",
        f"- Positive-MSE cells: {positive_cells}",
        f"- Failed/no-MSE cells retained: {failed_cells}",
        "- Leakage direction: lower MSE means stronger leakage; higher leakage_score means stronger leakage.",
        "",
        "## Outputs",
        "",
        "- `model_privacy_utility_table.csv`",
        "- `block_a_model_privacy_utility_table.csv`",
        "- `block_a_report.md`",
        "- `figures/block_a_macro_f1_vs_median_leakage_score.png`",
        "- `figures/block_a_macro_f1_vs_median_mse.png` "
        "(two-panel plot: measured MSE frontier plus no-MSE attack status)",
        "- `figures/block_a_macro_f1_vs_sigma.png`",
        "- `figures/block_a_median_mse_vs_sigma.png`",
        "- `figures/block_a_attack_success_rate_vs_sigma.png`",
        "",
        "## Model Table",
        "",
        markdown_table(table[available_cols]),
        "",
    ]
    (output_dir / "block_a_report.md").write_text("\n".join(lines), encoding="utf-8")


def create_outputs(args: argparse.Namespace, values: dict[str, Any]) -> None:
    table = load_model_table(args.output_dir)
    table = add_baseline_drops(table)
    table = frontier_candidates(table)
    table = table.sort_values(["split_label", "sigma", "experiment_name"])
    table.to_csv(args.output_dir / "block_a_model_privacy_utility_table.csv", index=False)

    figures_dir = args.output_dir / "figures"
    plot_frontier(
        table.dropna(subset=["median_leakage_score", "test_macro_f1"]),
        x_col="median_leakage_score",
        x_label="Median leakage score, -log10(MSE). Higher means more leakage",
        title="BloodMNIST Block A: Macro-F1 vs Median Leakage",
        output_base=figures_dir / "block_a_macro_f1_vs_median_leakage_score",
    )
    plot_mse_frontier_with_attack_status(
        table,
        title="BloodMNIST Block A: Macro-F1 vs Median MSE",
        output_base=figures_dir / "block_a_macro_f1_vs_median_mse",
    )
    plot_metric_vs_sigma(
        table.dropna(subset=["test_macro_f1"]),
        y_col="test_macro_f1",
        y_label="Test macro-F1",
        title="BloodMNIST Block A: Utility vs Sigma",
        output_base=figures_dir / "block_a_macro_f1_vs_sigma",
    )
    plot_metric_vs_sigma(
        table.dropna(subset=["median_mse"]),
        y_col="median_mse",
        y_label="Median reconstruction MSE. Lower means more leakage",
        title="BloodMNIST Block A: Median MSE vs Sigma",
        output_base=figures_dir / "block_a_median_mse_vs_sigma",
    )
    plot_metric_vs_sigma(
        table.dropna(subset=["attack_success_rate"]),
        y_col="attack_success_rate",
        y_label="Attack success rate (positive-MSE cells / attack cells)",
        title="BloodMNIST Block A: Attack Success Rate vs Sigma",
        output_base=figures_dir / "block_a_attack_success_rate_vs_sigma",
    )
    write_block_report(args.output_dir, table, args.protocol, values)


def main() -> None:
    args = parse_args()
    args.protocol = args.protocol.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_yaml(args.protocol)
    values = protocol_values(protocol, args)
    validate_training_artifacts(args.allow_missing_models)

    if not args.skip_attacks:
        run_fixed_attacker(args, values)

    if args.dry_run:
        print("Dry run complete. Skipping plots/report because attacks were not executed.")
        return

    create_outputs(args, values)
    print(f"Saved Block A outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
