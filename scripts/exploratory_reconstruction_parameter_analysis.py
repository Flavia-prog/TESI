import argparse
import itertools
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_MATRIX_CSV = (
    "results/current/privacy_utility/full_dp_privacy_utility_matrix/"
    "full_dp_privacy_utility_matrix_summary.csv"
)
DEFAULT_OUTPUT_DIR = "results/current/analysis/exploratory_reconstruction_parameter_analysis"
DEFAULT_AIJACK_SCRIPT = "scripts/gradient_inversion_medmnist_aijack.py"

LEAKAGE_METRIC_DESCRIPTION = (
    "leakage_score = -log10(reconstruction_mse); higher means stronger "
    "reconstruction/leakage. Raw MSE is retained separately, where lower means "
    "stronger reconstruction/leakage."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run laptop-scale AIJack gradient-inversion sweeps over attacker "
            "settings, then perform conservative exploratory regression of "
            "reconstruction quality."
        )
    )
    parser.add_argument(
        "--run-aijack-sweep",
        action="store_true",
        help=(
            "Execute AIJack gradient inversion attacks before analysis. "
            "Without this flag, the script only analyzes existing metrics."
        ),
    )
    parser.add_argument(
        "--sweep-design",
        choices=["full-factorial", "matched-ofat"],
        default="full-factorial",
        help=(
            "AIJack sweep design. full-factorial runs every combination. "
            "matched-ofat creates one-factor-at-a-time comparison blocks where "
            "only one attacker setting changes at a time. Default: full-factorial."
        ),
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Alias for the default behavior: do not execute new AIJack attacks.",
    )
    parser.add_argument(
        "--aijack-attack-script",
        default=DEFAULT_AIJACK_SCRIPT,
        help=f"AIJack attack script to execute. Default: {DEFAULT_AIJACK_SCRIPT}",
    )
    parser.add_argument(
        "--experiment-dir",
        action="append",
        default=[],
        help=(
            "Trained FedAvg experiment directory containing config.yaml and "
            "final_model.pt. Can be passed more than once. If omitted during "
            "a sweep, a small BloodMNIST baseline set under results/current is used."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Optional dataset override for the corresponding --experiment-dir. "
            "If one value is supplied it is reused for all experiment dirs."
        ),
    )
    parser.add_argument(
        "--clients",
        default="0",
        help="Comma-separated client ids for the AIJack sweep. Default: 0",
    )
    parser.add_argument(
        "--sample-indices",
        default="0,25",
        help="Comma-separated sample indices for the AIJack sweep. Default: 0,25",
    )
    parser.add_argument(
        "--attack-batch-sizes",
        default="1",
        help="Comma-separated attack batch sizes. Default: 1",
    )
    parser.add_argument(
        "--attack-iters-grid",
        default="300,1000",
        help="Comma-separated AIJack inversion iteration counts. Default: 300,1000",
    )
    parser.add_argument(
        "--num-trials-grid",
        default="3,5",
        help="Comma-separated AIJack trial counts. Default: 3,5",
    )
    parser.add_argument(
        "--attack-lrs",
        default="0.05,0.1",
        help="Comma-separated attack learning rates. Default: 0.05,0.1",
    )
    parser.add_argument(
        "--distances",
        default="l2,cossim",
        help="Comma-separated AIJack distance names. Default: l2,cossim",
    )
    parser.add_argument(
        "--ofat-anchor-clients",
        default=None,
        help=(
            "Comma-separated client ids used as repeated nuisance anchors for "
            "matched-ofat blocks. Default: first --clients value."
        ),
    )
    parser.add_argument(
        "--ofat-anchor-sample-indices",
        default=None,
        help=(
            "Comma-separated sample indices used as repeated nuisance anchors "
            "for matched-ofat blocks. Default: first --sample-indices value."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device forwarded to the AIJack attack script. Default: auto",
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Name for generated AIJack sweep outputs. Default: timestamped name.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on AIJack attack cells for quick screening.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the AIJack sweep manifest but do not execute attacks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run AIJack attack cells even if attack_metrics.json already exists.",
    )
    parser.add_argument(
        "--metrics-csv",
        action="append",
        default=[],
        help=(
            "CSV file containing attack metrics. Can be passed more than once. "
            f"Default: {DEFAULT_MATRIX_CSV} if it exists."
        ),
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help=(
            "Directory to recursively scan for attack_metrics.json and "
            "attack_failed.json. Can be passed more than once."
        ),
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help=(
            "Also scan results/_archive_low_value_20260516/individual_attack_outputs. "
            "Archived runs are preserved in the source column."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for tables, plots, and report. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--min-complete-rows",
        type=int,
        default=20,
        help="Minimum successful rows required for model-based screening.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=50,
        help="Permutation repeats for cross-validated feature screening.",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=2000,
        help="Bootstrap repeats for matched contrast confidence intervals.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def default_experiment_dirs() -> list[Path]:
    candidates = [
        Path("results/current/training/bloodmnist/baselines/iid_baseline"),
        Path("results/current/training/bloodmnist/baselines/noniid_alpha_01"),
        Path("results/current/training/bloodmnist/baselines/noniid_alpha_05"),
        Path("results/current/training/bloodmnist/baselines/noniid_alpha_1"),
    ]
    return [path for path in candidates if (path / "config.yaml").exists() and (path / "final_model.pt").exists()]


def safe_float_token(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def build_run_id(cell: dict[str, Any]) -> str:
    prefix = ""
    if cell.get("varied_parameter"):
        prefix = f"{cell['varied_parameter']}_{cell.get('comparison_level', 'level')}_"
    return prefix + (
        f"client{cell['client_id']}_sample{cell['sample_index']}_"
        f"bs{cell['attack_batch_size']}_{cell['distance']}_"
        f"iters{cell['attack_iters']}_trials{cell['num_trials']}_"
        f"lr{safe_float_token(float(cell['attack_lr']))}"
    )


def resolve_dataset_overrides(experiment_dirs: list[Path], datasets: list[str]) -> list[str | None]:
    if not datasets:
        return [None] * len(experiment_dirs)
    if len(datasets) == 1:
        return [datasets[0]] * len(experiment_dirs)
    if len(datasets) != len(experiment_dirs):
        raise ValueError(
            "Pass zero dataset overrides, one override reused for all experiment dirs, "
            "or one --dataset value per --experiment-dir."
        )
    return datasets


def build_aijack_sweep_cells(args: argparse.Namespace, sweep_root: Path) -> list[dict[str, Any]]:
    experiment_dirs = [Path(path) for path in args.experiment_dir] or default_experiment_dirs()
    if not experiment_dirs:
        raise SystemExit(
            "No trained experiment directories found. Pass --experiment-dir pointing "
            "to a directory with config.yaml and final_model.pt."
        )

    dataset_overrides = resolve_dataset_overrides(experiment_dirs, args.dataset)
    clients = parse_int_list(args.clients)
    sample_indices = parse_int_list(args.sample_indices)
    batch_sizes = parse_int_list(args.attack_batch_sizes)
    attack_iters = parse_int_list(args.attack_iters_grid)
    num_trials = parse_int_list(args.num_trials_grid)
    attack_lrs = parse_float_list(args.attack_lrs)
    distances = parse_str_list(args.distances)

    cells: list[dict[str, Any]] = []
    for experiment_dir, dataset in zip(experiment_dirs, dataset_overrides):
        experiment_name = experiment_dir.name
        for client_id, sample_index, batch_size, iters, trials, lr, distance in itertools.product(
            clients,
            sample_indices,
            batch_sizes,
            attack_iters,
            num_trials,
            attack_lrs,
            distances,
        ):
            cell = {
                "experiment_dir": str(experiment_dir),
                "experiment_name": experiment_name,
                "dataset_override": dataset,
                "client_id": client_id,
                "sample_index": sample_index,
                "attack_batch_size": batch_size,
                "attack_iters": iters,
                "num_trials": trials,
                "attack_lr": lr,
                "distance": distance,
            }
            run_id = build_run_id(cell)
            cell["run_id"] = run_id
            cell["output_dir"] = str(sweep_root / experiment_name / run_id)
            cells.append(cell)

    if args.max_runs is not None:
        cells = cells[: args.max_runs]
    return cells


def safe_level_token(value: Any) -> str:
    text = str(value).strip().replace("/", "_").replace("\\", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "missing"


def build_matched_ofat_sweep_cells(args: argparse.Namespace, sweep_root: Path) -> list[dict[str, Any]]:
    experiment_dirs = [Path(path) for path in args.experiment_dir] or default_experiment_dirs()
    if not experiment_dirs:
        raise SystemExit(
            "No trained experiment directories found. Pass --experiment-dir pointing "
            "to a directory with config.yaml and final_model.pt."
        )

    dataset_overrides = resolve_dataset_overrides(experiment_dirs, args.dataset)
    experiments = list(zip(experiment_dirs, dataset_overrides))

    clients = parse_int_list(args.clients)
    sample_indices = parse_int_list(args.sample_indices)
    batch_sizes = parse_int_list(args.attack_batch_sizes)
    attack_iters = parse_int_list(args.attack_iters_grid)
    num_trials = parse_int_list(args.num_trials_grid)
    attack_lrs = parse_float_list(args.attack_lrs)
    distances = parse_str_list(args.distances)
    anchor_clients = (
        parse_int_list(args.ofat_anchor_clients)
        if args.ofat_anchor_clients is not None
        else [clients[0]]
    )
    anchor_sample_indices = (
        parse_int_list(args.ofat_anchor_sample_indices)
        if args.ofat_anchor_sample_indices is not None
        else [sample_indices[0]]
    )

    if not all([clients, sample_indices, batch_sizes, attack_iters, num_trials, attack_lrs, distances]):
        raise SystemExit("All matched-ofat parameter grids must contain at least one value.")

    base_experiment_dir, base_dataset = experiments[0]
    base_cell = {
        "experiment_dir": str(base_experiment_dir),
        "experiment_name": base_experiment_dir.name,
        "dataset_override": base_dataset,
        "client_id": clients[0],
        "sample_index": sample_indices[0],
        "attack_batch_size": batch_sizes[0],
        "attack_iters": attack_iters[0],
        "num_trials": num_trials[0],
        "attack_lr": attack_lrs[0],
        "distance": distances[0],
    }

    dimensions: list[tuple[str, list[Any]]] = [
        ("experiment_dir", experiments),
        ("client_id", clients),
        ("sample_index", sample_indices),
        ("attack_batch_size", batch_sizes),
        ("attack_iters", attack_iters),
        ("num_trials", num_trials),
        ("attack_lr", attack_lrs),
        ("distance", distances),
    ]

    cells: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for varied_parameter, levels in dimensions:
        if len(levels) < 2:
            continue
        comparison_id = f"matched_ofat_{varied_parameter}"
        if varied_parameter == "client_id":
            anchor_pairs = [(None, sample_index) for sample_index in anchor_sample_indices]
        elif varied_parameter == "sample_index":
            anchor_pairs = [(client_id, None) for client_id in anchor_clients]
        else:
            anchor_pairs = list(itertools.product(anchor_clients, anchor_sample_indices))

        for anchor_client_id, anchor_sample_index in anchor_pairs:
            for level in levels:
                cell = dict(base_cell)
                if anchor_client_id is not None:
                    cell["client_id"] = anchor_client_id
                if anchor_sample_index is not None:
                    cell["sample_index"] = anchor_sample_index

                if varied_parameter == "experiment_dir":
                    experiment_dir, dataset = level
                    cell["experiment_dir"] = str(experiment_dir)
                    cell["experiment_name"] = experiment_dir.name
                    cell["dataset_override"] = dataset
                    comparison_level = experiment_dir.name
                else:
                    cell[varied_parameter] = level
                    comparison_level = level

                block_client = cell["client_id"] if varied_parameter != "client_id" else "varied"
                block_sample = cell["sample_index"] if varied_parameter != "sample_index" else "varied"
                comparison_block = f"client{block_client}_sample{block_sample}"

                cell["sweep_design"] = "matched-ofat"
                cell["comparison_id"] = comparison_id
                cell["comparison_block"] = comparison_block
                cell["varied_parameter"] = varied_parameter
                cell["comparison_level"] = safe_level_token(comparison_level)
                run_id = build_run_id(cell)
                cell["run_id"] = run_id
                cell["output_dir"] = str(
                    sweep_root
                    / comparison_id
                    / comparison_block
                    / cell["comparison_level"]
                    / run_id
                )

                dedupe_key = (comparison_id, comparison_block, str(comparison_level), run_id)
                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    cells.append(cell)

    if args.max_runs is not None:
        cells = cells[: args.max_runs]
    return cells


def run_aijack_sweep(args: argparse.Namespace, output_dir: Path) -> Path:
    sweep_name = args.sweep_name or datetime.now().strftime("aijack_sweep_%Y%m%d_%H%M%S")
    sweep_root = output_dir / "aijack_runs" / sweep_name
    sweep_root.mkdir(parents=True, exist_ok=True)

    if args.sweep_design == "matched-ofat":
        cells = build_matched_ofat_sweep_cells(args, sweep_root)
    else:
        cells = build_aijack_sweep_cells(args, sweep_root)
    manifest_rows: list[dict[str, Any]] = []
    attack_script = Path(args.aijack_attack_script)

    for index, cell in enumerate(cells, start=1):
        output_path = Path(cell["output_dir"])
        metrics_path = output_path / "attack_metrics.json"
        failed_path = output_path / "attack_failed.json"
        output_path.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(attack_script),
            "--experiment-dir",
            cell["experiment_dir"],
            "--client-id",
            str(cell["client_id"]),
            "--sample-index",
            str(cell["sample_index"]),
            "--attack-batch-size",
            str(cell["attack_batch_size"]),
            "--attack-iters",
            str(cell["attack_iters"]),
            "--num-trials",
            str(cell["num_trials"]),
            "--attack-lr",
            str(cell["attack_lr"]),
            "--distance",
            str(cell["distance"]),
            "--device",
            args.device,
            "--output-dir",
            str(output_path),
        ]
        if cell["dataset_override"]:
            command.extend(["--dataset", str(cell["dataset_override"])])

        row = dict(cell)
        row["command"] = " ".join(command)
        row["metrics_path"] = str(metrics_path)
        row["failed_path"] = str(failed_path)

        if metrics_path.exists() and not args.overwrite:
            row["run_status"] = "skipped_existing"
            manifest_rows.append(row)
            continue

        if args.dry_run:
            row["run_status"] = "dry_run"
            manifest_rows.append(row)
            continue

        print(f"[{index}/{len(cells)}] AIJack attack: {cell['experiment_name']} {cell['run_id']}", flush=True)
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        row["returncode"] = int(completed.returncode)
        row["stdout_path"] = str(output_path / "stdout.txt")
        row["stderr_path"] = str(output_path / "stderr.txt")
        (output_path / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (output_path / "stderr.txt").write_text(completed.stderr, encoding="utf-8")

        if completed.returncode == 0 and metrics_path.exists():
            row["run_status"] = "ok"
            try:
                metrics_payload = read_json(metrics_path)
                metrics_payload.update(
                    {
                        key: value
                        for key, value in cell.items()
                        if key
                        in {
                            "sweep_design",
                            "comparison_id",
                            "varied_parameter",
                            "comparison_block",
                            "comparison_level",
                            "run_id",
                        }
                    }
                )
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics_payload, f, indent=2)
            except Exception as error:
                row["metadata_merge_error"] = f"{type(error).__name__}: {error}"
        else:
            row["run_status"] = "failed"
            failure_payload = {
                **cell,
                "attack_status": "failed",
                "attack_error": f"returncode={completed.returncode}",
                "stdout_path": row["stdout_path"],
                "stderr_path": row["stderr_path"],
                "command": command,
            }
            with failed_path.open("w", encoding="utf-8") as f:
                json.dump(failure_payload, f, indent=2)

        manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(sweep_root / "aijack_sweep_manifest.csv", index=False)

    config = {
        "sweep_root": str(sweep_root),
        "sweep_design": args.sweep_design,
        "aijack_attack_script": str(attack_script),
        "device": args.device,
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "ofat_anchor_clients": args.ofat_anchor_clients,
        "ofat_anchor_sample_indices": args.ofat_anchor_sample_indices,
        "leakage_metric": LEAKAGE_METRIC_DESCRIPTION,
    }
    with (sweep_root / "sweep_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved AIJack sweep manifest: {sweep_root / 'aijack_sweep_manifest.csv'}")
    return sweep_root


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def infer_experiment_name(path: Path, payload: dict[str, Any]) -> str | None:
    experiment_dir = payload.get("experiment_dir")
    if experiment_dir:
        return Path(str(experiment_dir)).name

    parts = list(path.parts)
    if "individual_attack_outputs" in parts:
        idx = parts.index("individual_attack_outputs")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    if "training" in parts:
        idx = parts.index("training")
        if idx + 1 < len(parts):
            return parts[-3] if len(parts) >= 3 else None

    return path.parent.parent.name if path.parent.name == "attacks" else path.parent.name


def parse_decimal_token(token: str) -> float:
    if token.startswith("0") and len(token) > 1:
        return float(f"0.{token[1:]}")
    return float(token)


def parse_experiment_name(name: str | None) -> dict[str, Any]:
    if not name:
        return {}

    parsed: dict[str, Any] = {"experiment_name": name}
    lower = name.lower()

    if lower.startswith("iid"):
        parsed.setdefault("split_type", "iid")
    elif "noniid" in lower:
        parsed.setdefault("split_type", "dirichlet")

    alpha_match = re.search(r"alpha[_-]?(\d+)", lower)
    if alpha_match:
        parsed.setdefault("alpha", parse_decimal_token(alpha_match.group(1)))

    sigma_match = re.search(r"sigma[_-]?(\d+)", lower)
    if sigma_match:
        parsed.setdefault("sigma", parse_decimal_token(sigma_match.group(1)))
        parsed.setdefault("dp_enabled", True)
    elif "dp" in lower:
        parsed.setdefault("dp_enabled", True)
    else:
        parsed.setdefault("dp_enabled", False)
        parsed.setdefault("sigma", 0.0)

    for dataset in ("bloodmnist", "pathmnist", "dermamnist"):
        if dataset in lower:
            parsed.setdefault("dataset", dataset)

    return parsed


def flatten_attack_json(path: Path, status_override: str | None = None) -> dict[str, Any]:
    payload = read_json(path)
    row = dict(payload)
    row["source_file"] = str(path)
    row["source_kind"] = path.name
    row["experiment_name"] = infer_experiment_name(path, payload)

    inferred = parse_experiment_name(row.get("experiment_name"))
    for key, value in inferred.items():
        if row.get(key) in (None, ""):
            row[key] = value

    if status_override is not None:
        row["attack_status"] = status_override
    row.setdefault("attack_status", "failed" if path.name == "attack_failed.json" else None)
    return row


def load_json_rows(search_roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("attack_metrics.json")):
            rows.append(flatten_attack_json(path))
        for path in sorted(root.rglob("attack_failed.json")):
            rows.append(flatten_attack_json(path, status_override="failed"))
    return pd.DataFrame(rows)


def load_csv_rows(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frame["source_kind"] = "metrics_csv"
        if "experiment_name" not in frame.columns and "split_label" in frame.columns:
            frame["experiment_name"] = frame["split_label"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def coerce_bool(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return value


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "reconstruction_mse" not in df.columns:
        df["reconstruction_mse"] = np.nan

    for col in [
        "reconstruction_mse",
        "attack_batch_size",
        "attack_iters",
        "num_trials",
        "attack_lr",
        "alpha",
        "sigma",
        "epsilon",
        "client_id",
        "sample_index",
        "test_accuracy",
        "test_macro_f1",
        "number_of_reconstructions",
        "seed",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "dp_enabled" in df.columns:
        df["dp_enabled"] = df["dp_enabled"].map(coerce_bool)

    if "attack_status" not in df.columns:
        df["attack_status"] = np.where(df["reconstruction_mse"].notna(), "ok", "unknown")

    for col in ["dataset", "split_type", "distance", "experiment_name"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].astype("string")

    df.loc[df["dataset"].isna(), "dataset"] = "unknown"
    df.loc[df["distance"].isna(), "distance"] = "unknown"
    df.loc[df["split_type"].isna(), "split_type"] = "unknown"

    if "sigma" not in df.columns:
        df["sigma"] = np.nan
    if "dp_enabled" not in df.columns:
        df["dp_enabled"] = np.nan

    df.loc[df["sigma"].isna() & (df["dp_enabled"] == False), "sigma"] = 0.0
    df["successful_attack"] = (
        df["attack_status"].astype(str).str.lower().eq("ok")
        & df["reconstruction_mse"].notna()
        & (df["reconstruction_mse"] > 0)
    )
    df["log_mse"] = np.where(
        df["successful_attack"], np.log10(df["reconstruction_mse"]), np.nan
    )
    df["leakage_score"] = -df["log_mse"]
    return df


def robust_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for level, group in df.groupby(group_col, dropna=False):
        ok = group[group["successful_attack"]].copy()
        rows.append(
            {
                "parameter": group_col,
                "level": level,
                "n_total": int(len(group)),
                "n_success": int(len(ok)),
                "failure_rate": float(1.0 - len(ok) / len(group)) if len(group) else np.nan,
                "median_mse": float(ok["reconstruction_mse"].median()) if len(ok) else np.nan,
                "iqr_mse": (
                    float(ok["reconstruction_mse"].quantile(0.75) - ok["reconstruction_mse"].quantile(0.25))
                    if len(ok)
                    else np.nan
                ),
                "median_leakage_score": float(ok["leakage_score"].median()) if len(ok) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    categorical = [
        col
        for col in features
        if col in df.columns
        and (pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object or df[col].dtype == bool)
    ]
    numeric = [col for col in features if col not in categorical]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def choose_cv(df: pd.DataFrame, random_state: int):
    if "experiment_name" in df.columns and df["experiment_name"].nunique(dropna=True) >= 3:
        n_splits = min(5, int(df["experiment_name"].nunique(dropna=True)))
        return GroupKFold(n_splits=n_splits), df["experiment_name"].fillna("missing")

    n_splits = min(5, len(df))
    if n_splits < 2:
        return None, None
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), None


def cv_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    cv,
    groups: pd.Series | None,
    n_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, list[float]]:
    rng = np.random.default_rng(random_state)
    drops = {feature: [] for feature in features}
    baseline_scores: list[float] = []

    split_iter = cv.split(X, y, groups=groups) if groups is not None else cv.split(X, y)
    for train_idx, test_idx in split_iter:
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        baseline = r2_score(y_test, pipeline.predict(X_test))
        baseline_scores.append(float(baseline))

        for feature in features:
            repeated_drops = []
            for _ in range(n_repeats):
                X_perm = X_test.copy()
                X_perm[feature] = rng.permutation(X_perm[feature].to_numpy())
                permuted = r2_score(y_test, pipeline.predict(X_perm))
                repeated_drops.append(float(baseline - permuted))
            drops[feature].extend(repeated_drops)

    rows = []
    for feature, values in drops.items():
        arr = np.array(values, dtype=float)
        rows.append(
            {
                "parameter": feature,
                "heldout_importance_mean_r2_drop": float(np.mean(arr)),
                "heldout_importance_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            }
        )

    return (
        pd.DataFrame(rows).sort_values("heldout_importance_mean_r2_drop", ascending=False),
        baseline_scores,
    )


def model_screening(
    df: pd.DataFrame,
    features: list[str],
    min_complete_rows: int,
    n_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usable_cols = [col for col in features if col in df.columns]
    model_df = df[df["successful_attack"]].copy()
    model_df = model_df.dropna(subset=["leakage_score"])

    for col in usable_cols:
        if pd.api.types.is_numeric_dtype(model_df[col]):
            model_df[col] = model_df[col].fillna(model_df[col].median())
        else:
            model_df[col] = model_df[col].astype("string").fillna("missing")

    diagnostics = {
        "model_rows": int(len(model_df)),
        "features_requested": features,
        "features_used": usable_cols,
        "screening_model": "RidgeCV with grouped permutation importance",
        "note": (
            "Predictive screening only. Importance means the feature helps "
            "cross-validated prediction of leakage_score in this dataset; it "
            "is not a causal effect estimate."
        ),
    }

    if len(model_df) < min_complete_rows or len(usable_cols) == 0:
        diagnostics["skipped_reason"] = (
            f"Need at least {min_complete_rows} successful rows and at least one feature."
        )
        return pd.DataFrame(), diagnostics

    X = model_df[usable_cols]
    y = model_df["leakage_score"]
    cv, groups = choose_cv(model_df, random_state)
    if cv is None:
        diagnostics["skipped_reason"] = "Not enough rows for cross-validation."
        return pd.DataFrame(), diagnostics

    alphas = np.logspace(-3, 3, 13)
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(model_df, usable_cols)),
            ("model", RidgeCV(alphas=alphas)),
        ]
    )

    importance, scores = cv_permutation_importance(
        pipeline,
        X,
        y,
        usable_cols,
        cv,
        groups,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    diagnostics["cv_r2_mean"] = float(np.mean(scores))
    diagnostics["cv_r2_std"] = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    diagnostics["cv_r2_scores"] = [float(score) for score in scores]
    diagnostics["importance_method"] = "Permutation drops measured on held-out CV folds only."
    return importance, diagnostics


def bootstrap_ci(values: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    estimates = np.empty(repeats, dtype=float)
    for idx in range(repeats):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[idx] = np.median(sample)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def matched_contrasts(
    df: pd.DataFrame,
    candidate_params: list[str],
    bootstrap_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(random_state)
    base = df[df["successful_attack"]].copy()

    for param in candidate_params:
        if param not in base.columns or base[param].nunique(dropna=True) < 2:
            continue

        controls = [
            col
            for col in candidate_params
            if col != param and col in base.columns and base[col].nunique(dropna=True) > 1
        ]
        controls = [col for col in controls if not base[col].isna().all()]
        grouped = base.groupby(controls, dropna=False) if controls else [((), base)]

        contrasts = []
        for _, group in grouped:
            levels = sorted(group[param].dropna().unique(), key=lambda value: str(value))
            if len(levels) < 2:
                continue
            medians = group.groupby(param, dropna=True)["leakage_score"].median()
            for i, level_a in enumerate(levels):
                for level_b in levels[i + 1 :]:
                    if level_a in medians.index and level_b in medians.index:
                        contrasts.append(float(medians.loc[level_b] - medians.loc[level_a]))

        if not contrasts:
            continue

        values = np.array(contrasts, dtype=float)
        ci_low, ci_high = bootstrap_ci(values, bootstrap_repeats, rng)
        rows.append(
            {
                "parameter": param,
                "n_matched_contrasts": int(len(values)),
                "median_delta_leakage_score": float(np.median(values)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_abs_delta_leakage_score": float(np.mean(np.abs(values))),
                "interpretation": (
                    "Positive median means the later sorted level has higher leakage "
                    "within matched settings; inspect level ordering before making claims."
                ),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["n_matched_contrasts", "mean_abs_delta_leakage_score"],
        ascending=[False, False],
    )


def spearman_screening(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    base = df[df["successful_attack"]].copy()
    for col in features:
        if col not in base.columns or not pd.api.types.is_numeric_dtype(base[col]):
            continue
        sub = base[[col, "leakage_score"]].dropna()
        if len(sub) < 3 or sub[col].nunique() < 2:
            continue
        corr = sub[col].corr(sub["leakage_score"], method="spearman")
        rows.append({"parameter": col, "n": int(len(sub)), "spearman_with_leakage_score": float(corr)})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("spearman_with_leakage_score", key=lambda s: s.abs(), ascending=False)


def save_plots(output_dir: Path, importance: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    if not importance.empty:
        top = importance.sort_values("heldout_importance_mean_r2_drop", ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(top))))
        ax.barh(
            top["parameter"],
            top["heldout_importance_mean_r2_drop"],
            xerr=top["heldout_importance_std"],
        )
        ax.set_xlabel("Held-out permutation importance: R2 drop")
        ax.set_title("Predictive screening of reconstruction leakage")
        fig.tight_layout()
        fig.savefig(output_dir / "permutation_importance.png", dpi=200)
        plt.close(fig)

    if not contrasts.empty:
        top = contrasts.sort_values("mean_abs_delta_leakage_score", ascending=True).tail(12)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(top))))
        ax.barh(top["parameter"], top["mean_abs_delta_leakage_score"])
        ax.set_xlabel("Mean absolute matched delta in leakage score")
        ax.set_title("Matched within-setting contrasts")
        fig.tight_layout()
        fig.savefig(output_dir / "matched_contrasts.png", dpi=200)
        plt.close(fig)


def write_report(
    output_dir: Path,
    df: pd.DataFrame,
    importance: pd.DataFrame,
    contrasts: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> None:
    lines = [
        "# Exploratory Reconstruction Parameter Analysis",
        "",
        LEAKAGE_METRIC_DESCRIPTION,
        "",
        "## Data",
        "",
        f"- Total rows: {len(df)}",
        f"- Successful attacks with positive MSE: {int(df['successful_attack'].sum()) if not df.empty else 0}",
        f"- Failed/unknown/no-MSE rows retained in raw table: {int((~df['successful_attack']).sum()) if not df.empty else 0}",
        "",
        "## Model-Based Screening",
        "",
        (
            f"- Cross-validated R2: {diagnostics.get('cv_r2_mean', math.nan):.3f} "
            f"+/- {diagnostics.get('cv_r2_std', math.nan):.3f}"
            if "cv_r2_mean" in diagnostics
            else f"- Skipped: {diagnostics.get('skipped_reason', 'not available')}"
        ),
        "- Interpretation: use this only to prioritize deeper controlled experiments.",
        "",
    ]

    if not importance.empty:
        lines.extend(["Top permutation importances:", ""])
        for row in importance.head(10).itertuples(index=False):
            lines.append(
                f"- {row.parameter}: held-out mean R2 drop "
                f"{row.heldout_importance_mean_r2_drop:.4f} "
                f"(sd {row.heldout_importance_std:.4f})"
            )
        lines.append("")

    lines.extend(["## Matched Contrasts", ""])
    if contrasts.empty:
        lines.append("- No matched contrasts were available with the current data.")
    else:
        lines.append(
            "These compare one parameter while holding the other candidate "
            "parameters fixed where the data contain matched settings."
        )
        lines.append("")
        for row in contrasts.head(10).itertuples(index=False):
            lines.append(
                f"- {row.parameter}: n={row.n_matched_contrasts}, "
                f"median delta leakage={row.median_delta_leakage_score:.4f}, "
                f"95% bootstrap CI [{row.ci95_low:.4f}, {row.ci95_high:.4f}]"
            )
    lines.extend(
        [
            "",
            "## Cautions",
            "",
            "- This is observational exploratory analysis, not causal identification.",
            "- Repeated attacks on the same training run are statistically dependent.",
            "- Parameters with sparse or unbalanced coverage can look important for design reasons.",
            "- Treat stable findings across matched contrasts, datasets, and held-out runs as stronger evidence.",
            "",
        ]
    )

    with (output_dir / "analysis_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with (output_dir / "model_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_aijack_sweep and args.analysis_only:
        raise SystemExit("Use either --run-aijack-sweep or --analysis-only, not both.")

    generated_sweep_root = None
    if args.run_aijack_sweep:
        generated_sweep_root = run_aijack_sweep(args, output_dir)
        if args.dry_run and not args.metrics_csv and not args.search_root and not args.include_archive:
            print("Dry run complete; no analysis was run because no metrics were generated.")
            return

    csv_paths = [Path(path) for path in args.metrics_csv]
    if not args.run_aijack_sweep and not csv_paths and Path(DEFAULT_MATRIX_CSV).exists():
        csv_paths = [Path(DEFAULT_MATRIX_CSV)]

    search_roots = [Path(path) for path in args.search_root]
    if generated_sweep_root is not None:
        search_roots.append(generated_sweep_root)
    if args.include_archive:
        search_roots.append(Path("results/_archive_low_value_20260516/individual_attack_outputs"))

    csv_df = load_csv_rows(csv_paths)
    json_df = load_json_rows(search_roots)
    combined = pd.concat([csv_df, json_df], ignore_index=True, sort=False)
    combined = normalize_table(combined)

    if combined.empty:
        raise SystemExit(
            "No attack metrics found. Pass --metrics-csv and/or --search-root."
        )

    combined.to_csv(output_dir / "analysis_dataset.csv", index=False)

    candidate_params = [
        "dataset",
        "split_type",
        "alpha",
        "dp_enabled",
        "sigma",
        "epsilon",
        "test_accuracy",
        "test_macro_f1",
        "client_id",
        "sample_index",
        "attack_batch_size",
        "attack_iters",
        "num_trials",
        "attack_lr",
        "distance",
    ]
    candidate_params = [col for col in candidate_params if col in combined.columns]

    summaries = [
        robust_summary(combined, col)
        for col in candidate_params
        if col in combined.columns and combined[col].nunique(dropna=True) <= 30
    ]
    group_summary = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    group_summary.to_csv(output_dir / "group_level_summary.csv", index=False)

    spearman = spearman_screening(combined, candidate_params)
    spearman.to_csv(output_dir / "spearman_numeric_screening.csv", index=False)

    importance, diagnostics = model_screening(
        combined,
        candidate_params,
        min_complete_rows=args.min_complete_rows,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    importance.to_csv(output_dir / "permutation_importance.csv", index=False)

    contrast_params = [
        "dataset",
        "split_type",
        "alpha",
        "dp_enabled",
        "sigma",
        "client_id",
        "sample_index",
        "attack_batch_size",
        "attack_iters",
        "num_trials",
        "attack_lr",
        "distance",
    ]
    contrasts = matched_contrasts(
        combined,
        [col for col in contrast_params if col in combined.columns],
        bootstrap_repeats=args.bootstrap_repeats,
        random_state=args.random_state,
    )
    contrasts.to_csv(output_dir / "matched_contrasts.csv", index=False)

    save_plots(output_dir, importance, contrasts)
    write_report(output_dir, combined, importance, contrasts, diagnostics)

    print(f"Saved normalized dataset: {output_dir / 'analysis_dataset.csv'}")
    print(f"Saved group summaries: {output_dir / 'group_level_summary.csv'}")
    print(f"Saved model screening: {output_dir / 'permutation_importance.csv'}")
    print(f"Saved matched contrasts: {output_dir / 'matched_contrasts.csv'}")
    print(f"Saved report: {output_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
