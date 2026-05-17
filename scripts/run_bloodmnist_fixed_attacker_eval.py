import argparse
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ATTACK_SCRIPT = SCRIPT_DIR / "gradient_inversion_bloodmnist_aijack.py"

DEFAULT_SPLITS = ["iid", "noniid_alpha_1", "noniid_alpha_05", "noniid_alpha_01"]
DEFAULT_SIGMAS = [0.25, 0.5, 0.75, 1.0, 2.0]
DEFAULT_CLIENT_IDS = [0, 1, 2]
DEFAULT_SAMPLE_INDICES = [0, 25, 50]

SPLIT_SPECS = {
    "iid": {
        "split_type": "iid",
        "alpha": 0.0,
        "baseline": Path("results/current/training/bloodmnist/baselines/iid_baseline"),
        "prefix": "iid",
    },
    "noniid_alpha_1": {
        "split_type": "dirichlet",
        "alpha": 1.0,
        "baseline": Path("results/current/training/bloodmnist/baselines/noniid_alpha_1"),
        "prefix": "noniid_alpha_1",
    },
    "noniid_alpha_05": {
        "split_type": "dirichlet",
        "alpha": 0.5,
        "baseline": Path("results/current/training/bloodmnist/baselines/noniid_alpha_05"),
        "prefix": "noniid_alpha_05",
    },
    "noniid_alpha_01": {
        "split_type": "dirichlet",
        "alpha": 0.1,
        "baseline": Path("results/current/training/bloodmnist/baselines/noniid_alpha_01"),
        "prefix": "noniid_alpha_01",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed BloodMNIST attacker across saved baseline and DP matrix models."
    )
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--sigmas", nargs="+", type=float, default=DEFAULT_SIGMAS)
    parser.add_argument("--client-ids", nargs="+", type=int, default=DEFAULT_CLIENT_IDS)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument("--attack-iters", type=int, default=300)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--attack-lr", type=float, default=0.05)
    parser.add_argument("--distance", choices=["l2", "cossim"], default="cossim")
    parser.add_argument("--dataset", default="bloodmnist")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="results/current/privacy_utility/bloodmnist_fixed_attacker_v1",
    )
    return parser.parse_args()


def sigma_tag(sigma: float) -> str:
    text = f"{sigma:g}"
    return re.sub(r"[^0-9A-Za-z]+", "", text)


def float_tag(value: float) -> str:
    text = f"{value:g}"
    return re.sub(r"[^0-9A-Za-z]+", "", text)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def load_first_csv_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def attack_run_name(args: argparse.Namespace, client_id: int, sample_index: int) -> str:
    return (
        f"fixed_attacker_batch{args.attack_batch_size}_{args.distance}_"
        f"{args.attack_iters}iters_{args.num_trials}trials_lr{float_tag(args.attack_lr)}_"
        f"client{client_id}_sample{sample_index}"
    )


def discover_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    invalid = [split for split in args.splits if split not in SPLIT_SPECS]
    if invalid:
        raise ValueError(f"Unsupported split labels: {invalid}")

    for split_label in args.splits:
        spec = SPLIT_SPECS[split_label]
        baseline = spec["baseline"]
        experiments.append(
            {
                "split_label": split_label,
                "split_type": spec["split_type"],
                "alpha": float(spec["alpha"]),
                "sigma": 0.0,
                "dp_enabled": False,
                "experiment_name": baseline.name,
                "experiment_dir": baseline,
            }
        )
        for sigma in args.sigmas:
            experiment_name = f"{spec['prefix']}_dp_sigma_{sigma_tag(float(sigma))}"
            experiments.append(
                {
                    "split_label": split_label,
                    "split_type": spec["split_type"],
                    "alpha": float(spec["alpha"]),
                    "sigma": float(sigma),
                    "dp_enabled": True,
                    "experiment_name": experiment_name,
                    "experiment_dir": Path("results/current/training/bloodmnist/dp_matrix")
                    / experiment_name,
                }
            )
    return experiments


def metadata_for_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    experiment_dir = Path(experiment["experiment_dir"])
    config = load_yaml(experiment_dir / "config.yaml")
    metrics = load_first_csv_row(experiment_dir / "test_metrics.csv")
    return {
        **experiment,
        "experiment_dir": str(experiment_dir.resolve()),
        "config_exists": (experiment_dir / "config.yaml").exists(),
        "model_exists": (experiment_dir / "final_model.pt").exists(),
        "test_accuracy": to_float(metrics.get("test_accuracy")),
        "test_macro_f1": to_float(metrics.get("test_macro_f1")),
        "test_loss": to_float(metrics.get("test_loss")),
        "clip_norm": to_float(config.get("clip_norm")),
        "noise_multiplier": to_float(config.get("noise_multiplier")),
        "epsilon": to_float(metrics.get("epsilon", config.get("epsilon"))),
        "delta": to_float(metrics.get("delta", config.get("delta"))),
    }


def build_cells(args: argparse.Namespace, experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for experiment in experiments:
        meta = metadata_for_experiment(experiment)
        experiment_dir = Path(meta["experiment_dir"])
        for client_id in args.client_ids:
            for sample_index in args.sample_indices:
                output_dir = experiment_dir / "attacks" / attack_run_name(args, client_id, sample_index)
                cells.append(
                    {
                        **meta,
                        "client_id": int(client_id),
                        "sample_index": int(sample_index),
                        "attack_batch_size": int(args.attack_batch_size),
                        "attack_iters": int(args.attack_iters),
                        "num_trials": int(args.num_trials),
                        "attack_lr": float(args.attack_lr),
                        "distance": args.distance,
                        "attack_output_dir": str(output_dir.resolve()),
                        "attack_metrics_path": str((output_dir / "attack_metrics.json").resolve()),
                        "attack_failed_path": str((output_dir / "attack_failed.json").resolve()),
                    }
                )
    return cells


def run_cell(args: argparse.Namespace, cell: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(cell["attack_output_dir"])
    metrics_path = Path(cell["attack_metrics_path"])
    failed_path = Path(cell["attack_failed_path"])
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"

    row = dict(cell)
    row.update(
        {
            "attack_status": "not_run",
            "attack_error": "",
            "reconstruction_mse": float("nan"),
            "leakage_score": float("nan"),
            "number_of_reconstructions": float("nan"),
            "command": "",
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
        }
    )

    if not row["model_exists"]:
        row["attack_status"] = "skipped_missing_model"
        return row

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ATTACK_SCRIPT),
        "--experiment-dir",
        str(Path(row["experiment_dir"])),
        "--client-id",
        str(row["client_id"]),
        "--sample-index",
        str(row["sample_index"]),
        "--attack-batch-size",
        str(row["attack_batch_size"]),
        "--attack-iters",
        str(row["attack_iters"]),
        "--num-trials",
        str(row["num_trials"]),
        "--attack-lr",
        str(row["attack_lr"]),
        "--distance",
        str(row["distance"]),
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
        "--dataset",
        args.dataset,
    ]
    row["command"] = " ".join(command)

    if metrics_path.exists() and not args.rerun:
        row["attack_status"] = "skipped_existing_metrics"
    elif args.dry_run:
        row["attack_status"] = "dry_run_planned"
    else:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            row["attack_status"] = "failed"
            row["attack_error"] = f"returncode={completed.returncode}"
            failed_payload = {
                **row,
                "timestamp": datetime.now().isoformat(),
                "command": command,
            }
            with failed_path.open("w", encoding="utf-8") as f:
                json.dump(failed_payload, f, indent=2)

    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            row["attack_status"] = metrics.get("attack_status") or row["attack_status"]
            row["attack_error"] = metrics.get("attack_error") or ""
            row["reconstruction_mse"] = to_float(metrics.get("reconstruction_mse"))
            row["number_of_reconstructions"] = to_float(metrics.get("number_of_reconstructions"))
        except Exception as error:
            row["attack_status"] = "metrics_parse_failed"
            row["attack_error"] = f"{type(error).__name__}: {error}"
    elif failed_path.exists() and row["attack_status"] not in {"dry_run_planned", "skipped_missing_model"}:
        row["attack_status"] = "failed"

    mse = to_float(row["reconstruction_mse"])
    if math.isfinite(mse) and mse > 0:
        row["leakage_score"] = -math.log10(mse)
    return row


def aggregate_model_table(cell_df: pd.DataFrame) -> pd.DataFrame:
    df = cell_df.copy()
    df["reconstruction_mse"] = pd.to_numeric(df["reconstruction_mse"], errors="coerce")
    df["leakage_score"] = pd.to_numeric(df["leakage_score"], errors="coerce")
    df["positive_mse"] = df["reconstruction_mse"].gt(0)

    group_cols = [
        "experiment_name",
        "split_label",
        "split_type",
        "alpha",
        "dp_enabled",
        "sigma",
        "test_accuracy",
        "test_macro_f1",
    ]
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_attack_cells=("attack_status", "size"),
            n_positive_mse=("positive_mse", "sum"),
            n_failed_or_no_mse=("positive_mse", lambda x: int((~x).sum())),
            median_mse=("reconstruction_mse", "median"),
            median_leakage_score=("leakage_score", "median"),
            worst_mse=("reconstruction_mse", "min"),
            worst_leakage_score=("leakage_score", "max"),
        )
        .reset_index()
    )
    grouped["attack_success_rate"] = grouped["n_positive_mse"] / grouped["n_attack_cells"]
    columns = [
        "experiment_name",
        "split_label",
        "split_type",
        "alpha",
        "dp_enabled",
        "sigma",
        "test_accuracy",
        "test_macro_f1",
        "attack_success_rate",
        "n_attack_cells",
        "n_positive_mse",
        "n_failed_or_no_mse",
        "median_mse",
        "median_leakage_score",
        "worst_mse",
        "worst_leakage_score",
    ]
    return grouped[columns].sort_values(["split_label", "dp_enabled", "sigma", "experiment_name"])


def write_report(output_dir: Path, cells: pd.DataFrame, table: pd.DataFrame) -> None:
    total = len(cells)
    positive = int(pd.to_numeric(cells["reconstruction_mse"], errors="coerce").gt(0).sum())
    failed_or_no_mse = total - positive
    lines = [
        "# BloodMNIST Fixed-Attacker Privacy-Utility Evaluation",
        "",
        f"- Generated at: {datetime.now().isoformat()}",
        f"- Total attack cells retained: {total}",
        f"- Positive-MSE attack cells: {positive}",
        f"- Failed/no-MSE attack cells retained: {failed_or_no_mse}",
        "- Leakage direction: lower MSE means stronger leakage; higher leakage_score means stronger leakage.",
        "",
        "## Model Table",
        "",
        markdown_table(table),
        "",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

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
        escaped = [str(row[col]).replace("|", "\\|") for col in view.columns]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if not ATTACK_SCRIPT.exists():
        raise FileNotFoundError(f"Missing attack script: {ATTACK_SCRIPT}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments(args)
    cells = build_cells(args, experiments)

    print(f"Prepared {len(cells)} attack cells across {len(experiments)} models.")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_cell = {executor.submit(run_cell, args, cell): cell for cell in cells}
        for index, future in enumerate(as_completed(future_to_cell), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(cells)}] {row['experiment_name']} "
                f"client{row['client_id']}_sample{row['sample_index']} "
                f"{row['attack_status']} mse={row['reconstruction_mse']}",
                flush=True,
            )

    cell_df = pd.DataFrame(rows).sort_values(["experiment_name", "client_id", "sample_index"])
    cell_path = output_dir / "attack_cell_summary.csv"
    cell_df.to_csv(cell_path, index=False)

    table = aggregate_model_table(cell_df)
    table_path = output_dir / "model_privacy_utility_table.csv"
    table.to_csv(table_path, index=False)

    write_report(output_dir, cell_df, table)

    print(f"Saved attack-cell summary: {cell_path.resolve()}")
    print(f"Saved model privacy-utility table: {table_path.resolve()}")
    print(f"Saved report: {(output_dir / 'analysis_report.md').resolve()}")


if __name__ == "__main__":
    main()
