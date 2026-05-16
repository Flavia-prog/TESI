from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import re
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplcache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torchmetrics.functional.image import structural_similarity_index_measure
from tqdm import tqdm

from fl_shared.models import available_model_arches
from fl_shared.runtime import collect_provenance
from gradient_inversion_bloodmnist_aijack import run_gradient_inversion_attack


DEFAULT_EXPERIMENT_DIRS = [
    "results/current/training/bloodmnist/baselines/iid_baseline",
    "results/current/training/bloodmnist/baselines/noniid_alpha_1",
    "results/current/training/bloodmnist/baselines/noniid_alpha_05",
    "results/current/training/bloodmnist/baselines/noniid_alpha_01",
]

DESIGN_FEATURES = [
    "experiment_dir",
    "distance",
    # attack_batch_size changes how many target samples/gradients are inverted jointly.
    # In gradient inversion this can materially change leakage quality and difficulty.
    "attack_batch_size",
    "attack_iters",
    "num_trials",
    "attack_lr",
    "client_id",
    "sample_index",
]

BALANCED_SCREENING_DEFAULT_MAX_RUNS = 128
BOOTSTRAP_REPEATS = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled sweep of AIJack gradient inversion attacks over trained "
            "FedAvg experiments, then estimate exploratory parameter impact."
        )
    )

    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        default=DEFAULT_EXPERIMENT_DIRS,
        help="Trained experiment directories, each containing config.yaml and final_model.pt.",
    )
    parser.add_argument("--client-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0, 25, 50])
    parser.add_argument("--attack-batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--attack-iters", nargs="+", type=int, default=[1000])
    parser.add_argument("--num-trials", nargs="+", type=int, default=[5])
    parser.add_argument("--attack-lrs", nargs="+", type=float, default=[0.1])
    parser.add_argument(
        "--distances",
        nargs="+",
        default=["cossim"],
        choices=["l2", "cossim"],
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset override for every run (e.g., bloodmnist, pathmnist).",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default=None,
        choices=available_model_arches(),
        help="Optional model architecture override for every run.",
    )
    parser.add_argument(
        "--design",
        choices=["full_factorial", "balanced_screening"],
        default="full_factorial",
        help=(
            "Design used to select attack runs. "
            "'full_factorial' keeps all combinations (optionally capped by --max-runs). "
            "'balanced_screening' builds a deterministic, capped subset for laptop-scale screening."
        ),
    )
    parser.add_argument(
        "--design-seed",
        type=int,
        default=42,
        help="Random seed used for deterministic balanced_screening selection.",
    )
    parser.add_argument(
        "--ensure-varies",
        nargs="+",
        default=[],
        choices=DESIGN_FEATURES,
        help=(
            "Parameters that should vary in the selected design when possible. "
            "Warnings are emitted if fewer than 2 unique values are selected."
        ),
    )
    parser.add_argument(
        "--dry-run-design",
        action="store_true",
        help="Build and report the design without running attacks.",
    )
    parser.add_argument(
        "--save-design-csv",
        action="store_true",
        help="Save selected design combinations as design_selected_runs.csv.",
    )
    parser.add_argument(
        "--target-metric",
        choices=["best_mse", "best_ssim"],
        default="best_ssim",
        help="Dependent variable used by the exploratory model.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Do not run attacks. Only aggregate/regress an existing sweep by --sweep-name.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rerun attacks even if attack_metrics.json already exists.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Optional cap on selected runs. In balanced_screening this cap is applied "
            "after coverage-aware subset selection."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of attacks to run in parallel. Use 1 for sequential execution.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/current/analysis/attack_parameter_impact/bloodmnist",
        help="Directory where analysis outputs are stored.",
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default=None,
        help=(
            "Human-readable sweep name. If omitted, a timestamped name is used. "
            "Only folders from this sweep are aggregated."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        choices=["inprocess", "subprocess"],
        default="inprocess",
        help="Use inprocess to avoid Python startup/reload overhead.",
    )
    parser.add_argument(
        "--attack-script",
        type=str,
        default="gradient_inversion_bloodmnist_aijack.py",
        help="Only used when --execution-mode subprocess.",
    )

    return parser.parse_args()


def sanitize_name(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def float_tag(value: float) -> str:
    text = f"{value:g}"
    return re.sub(r"[^0-9A-Za-z]+", "", text)


def build_attack_run_dir(
    experiment_dir: Path,
    sweep_name: str,
    client_id: int,
    sample_index: int,
    attack_batch_size: int,
    attack_iters: int,
    num_trials: int,
    attack_lr: float,
    distance: str,
) -> Path:
    run_name = (
        f"{sweep_name}_batch{attack_batch_size}_{distance}_{attack_iters}iters_"
        f"{num_trials}trials_lr{float_tag(attack_lr)}_"
        f"client{client_id}_sample{sample_index}"
    )
    return experiment_dir / "attacks" / run_name


def as_jsonable_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def unique_sorted(values: list[Any]) -> list[Any]:
    values_json = [as_jsonable_value(value) for value in values]
    seen = []
    seen_keys = set()
    for value in values_json:
        key = json.dumps(value, sort_keys=True)
        if key not in seen_keys:
            seen.append(value)
            seen_keys.add(key)
    return sorted(seen, key=lambda item: str(item))


def combo_to_row(combo: dict[str, Any]) -> dict[str, Any]:
    row = {}
    for key, value in combo.items():
        row[key] = str(value) if isinstance(value, Path) else value
    return row


def build_full_factorial_combos(
    args: argparse.Namespace,
    sweep_name: str,
) -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for (
        experiment,
        client_id,
        sample_index,
        attack_batch_size,
        attack_iters,
        num_trials,
        attack_lr,
        distance,
    ) in itertools.product(
        args.experiment_dirs,
        args.client_ids,
        args.sample_indices,
        args.attack_batch_sizes,
        args.attack_iters,
        args.num_trials,
        args.attack_lrs,
        args.distances,
    ):
        experiment_dir = Path(experiment)
        run_dir = build_attack_run_dir(
            experiment_dir=experiment_dir,
            sweep_name=sweep_name,
            client_id=client_id,
            sample_index=sample_index,
            attack_batch_size=attack_batch_size,
            attack_iters=attack_iters,
            num_trials=num_trials,
            attack_lr=attack_lr,
            distance=distance,
        )
        combos.append(
            {
                "experiment_dir": experiment_dir,
                "run_dir": run_dir,
                "client_id": client_id,
                "sample_index": sample_index,
                "attack_batch_size": attack_batch_size,
                "attack_iters": attack_iters,
                "num_trials": num_trials,
                "attack_lr": attack_lr,
                "distance": distance,
            }
        )
    return combos


def select_balanced_subset(
    combos: list[dict[str, Any]],
    max_runs: int,
    seed: int,
    ensure_varies: list[str],
) -> list[dict[str, Any]]:
    if max_runs >= len(combos):
        return list(combos)
    if max_runs < 1:
        return []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(combos))
    rng.shuffle(indices)
    randomized_combos = [combos[idx] for idx in indices]

    available_unique = {
        feature: len(unique_sorted([combo[feature] for combo in randomized_combos]))
        for feature in DESIGN_FEATURES
    }
    required_n_unique = {
        feature: min(2, available_unique[feature]) for feature in ensure_varies
    }

    selected: list[dict[str, Any]] = []
    remaining = list(range(len(randomized_combos)))
    value_counts = {
        feature: {}
        for feature in DESIGN_FEATURES
    }
    selected_unique = {feature: set() for feature in DESIGN_FEATURES}

    while remaining and len(selected) < max_runs:
        best_idx = None
        best_score = None

        for candidate_idx in remaining:
            candidate = randomized_combos[candidate_idx]
            score = 0.0

            for feature in DESIGN_FEATURES:
                value = as_jsonable_value(candidate[feature])
                count = value_counts[feature].get(value, 0)
                score += 1.0 / (1.0 + count)

                if feature in required_n_unique:
                    current_unique = len(selected_unique[feature])
                    if current_unique < required_n_unique[feature] and value not in selected_unique[feature]:
                        score += 5.0

            score += float(rng.uniform(0.0, 1e-6))

            if best_score is None or score > best_score:
                best_score = score
                best_idx = candidate_idx

        if best_idx is None:
            break

        chosen = randomized_combos[best_idx]
        selected.append(chosen)
        remaining.remove(best_idx)

        for feature in DESIGN_FEATURES:
            value = as_jsonable_value(chosen[feature])
            value_counts[feature][value] = value_counts[feature].get(value, 0) + 1
            selected_unique[feature].add(value)

    return selected


def build_design_report(
    full_combos: list[dict[str, Any]],
    selected_combos: list[dict[str, Any]],
    design: str,
    ensure_varies: list[str],
    design_seed: int,
    max_runs: int | None,
) -> dict[str, Any]:
    parameters = {}
    warnings = []
    assessable_parameters = []
    constant_parameters = []

    for feature in DESIGN_FEATURES:
        full_values = unique_sorted([combo[feature] for combo in full_combos])
        selected_values = unique_sorted([combo[feature] for combo in selected_combos])
        n_selected_unique = len(selected_values)
        assessable = n_selected_unique >= 2

        if assessable:
            assessable_parameters.append(feature)
        else:
            constant_parameters.append(feature)
            if feature == "experiment_dir":
                warnings.append(
                    "Parameter 'experiment_dir' is constant in selected design, so train split/alpha context may be underrepresented."
                )
            else:
                warnings.append(
                    f"Parameter '{feature}' is constant in selected design and will be dropped from feature-importance analysis."
                )

        if feature in ensure_varies and n_selected_unique < 2:
            warnings.append(
                f"Requested ensure-varies parameter '{feature}' has only {n_selected_unique} unique value(s) in the selected design."
            )

        parameters[feature] = {
            "n_unique_full_factorial": len(full_values),
            "values_full_factorial": full_values,
            "n_unique_selected": n_selected_unique,
            "values_selected": selected_values,
            "assessable_by_feature_importance": assessable,
        }

    report = {
        "design": design,
        "design_seed": design_seed,
        "max_runs_requested": max_runs,
        "n_total_full_factorial_combinations": len(full_combos),
        "n_selected_combinations": len(selected_combos),
        "ensure_varies": ensure_varies,
        "assessable_parameters": assessable_parameters,
        "constant_parameters": constant_parameters,
        "parameters": parameters,
        "warnings": warnings,
    }
    return report


def build_selected_combos(
    args: argparse.Namespace,
    sweep_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None]:
    full_combos = build_full_factorial_combos(args, sweep_name=sweep_name)
    requested_max_runs = args.max_runs

    if args.design == "full_factorial":
        selected_combos = full_combos
        if requested_max_runs is not None:
            selected_combos = selected_combos[:requested_max_runs]
        return full_combos, selected_combos, requested_max_runs

    design_cap = requested_max_runs
    if design_cap is None:
        design_cap = min(BALANCED_SCREENING_DEFAULT_MAX_RUNS, len(full_combos))

    selected_combos = select_balanced_subset(
        combos=full_combos,
        max_runs=design_cap,
        seed=args.design_seed,
        ensure_varies=args.ensure_varies,
    )
    return full_combos, selected_combos, design_cap


def select_device(requested_device: str) -> str:
    if requested_device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested --device mps, but MPS is not available in this PyTorch environment.")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --device cuda, but CUDA is not available in this environment.")

    return requested_device


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def compute_ssim_per_image(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    scores = []
    for image_index in range(original.size(0)):
        orig_img = original[image_index : image_index + 1]
        recon_img = reconstructed[image_index : image_index + 1]
        score = structural_similarity_index_measure(orig_img, recon_img, data_range=1.0)
        scores.append(score)
    return torch.stack(scores)


def compute_best_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> tuple[float | None, float | None]:
    if original.ndim != 4 or reconstructed.ndim != 4:
        return None, None

    orig = denormalize(original.float())
    recon = denormalize(reconstructed.float())

    n_orig = orig.size(0)
    n_recon = recon.size(0)

    if n_orig <= 0 or n_recon <= 0:
        return None, None

    if n_recon == n_orig:
        candidates = recon.unsqueeze(0)
    elif n_recon % n_orig == 0:
        candidates = recon.view(n_recon // n_orig, n_orig, *orig.shape[1:])
    else:
        return None, None

    mses = []
    ssims = []

    for candidate in candidates:
        mse = torch.mean((candidate - orig) ** 2).item()
        ssim = compute_ssim_per_image(orig, candidate).mean().item()
        mses.append(mse)
        ssims.append(ssim)

    return float(min(mses)), float(max(ssims))


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_attack(
    execution_mode: str,
    attack_script_path: Path,
    experiment_dir: Path,
    run_dir: Path,
    client_id: int,
    sample_index: int,
    attack_batch_size: int,
    attack_iters: int,
    num_trials: int,
    attack_lr: float,
    distance: str,
    device: str,
    dataset_override: str | None,
    model_arch_override: str | None,
) -> tuple[bool, str, str, int]:
    run_dir.mkdir(parents=True, exist_ok=True)

    if execution_mode == "subprocess":
        import sys

        cmd = [
            sys.executable,
            str(attack_script_path),
            "--experiment-dir",
            str(experiment_dir),
            "--client-id",
            str(client_id),
            "--sample-index",
            str(sample_index),
            "--attack-batch-size",
            str(attack_batch_size),
            "--attack-iters",
            str(attack_iters),
            "--num-trials",
            str(num_trials),
            "--attack-lr",
            str(attack_lr),
            "--distance",
            distance,
            "--device",
            device,
            "--output-dir",
            str(run_dir),
        ]

        if dataset_override:
            cmd.extend(["--dataset", dataset_override])
        if model_arch_override:
            cmd.extend(["--model-arch", model_arch_override])

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        stdout_text = result.stdout or ""
        stderr_text = result.stderr or ""
        returncode = result.returncode
    else:
        cmd = [
            "python",
            str(attack_script_path),
            "--experiment-dir",
            str(experiment_dir),
            "--client-id",
            str(client_id),
            "--sample-index",
            str(sample_index),
            "--attack-batch-size",
            str(attack_batch_size),
            "--attack-iters",
            str(attack_iters),
            "--num-trials",
            str(num_trials),
            "--attack-lr",
            str(attack_lr),
            "--distance",
            distance,
            "--device",
            device,
            "--output-dir",
            str(run_dir),
        ]
        if dataset_override:
            cmd.extend(["--dataset", dataset_override])
        if model_arch_override:
            cmd.extend(["--model-arch", model_arch_override])

        try:
            metrics = run_gradient_inversion_attack(
                experiment_dir=experiment_dir,
                client_id=client_id,
                sample_index=sample_index,
                attack_batch_size=attack_batch_size,
                attack_iters=attack_iters,
                num_trials=num_trials,
                attack_lr=attack_lr,
                distance=distance,
                device_arg=device,
                output_dir=run_dir,
                dataset_override=dataset_override,
                model_arch_override=model_arch_override,
            )
            stdout_text = json.dumps(
                {
                    "status": metrics.get("attack_status", "ok"),
                    "output_dir": metrics.get("output_dir"),
                }
            )
            stderr_text = ""
            returncode = 0
        except Exception as error:
            stdout_text = ""
            stderr_text = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
            returncode = 1

    with (run_dir / "attack_stdout.txt").open("w", encoding="utf-8") as f:
        f.write(stdout_text)

    with (run_dir / "attack_stderr.txt").open("w", encoding="utf-8") as f:
        f.write(stderr_text)

    if returncode != 0:
        failure_info = {
            "timestamp": datetime.now().isoformat(),
            "command": cmd,
            "execution_mode": execution_mode,
            "returncode": returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "experiment_dir": str(experiment_dir),
            "client_id": client_id,
            "sample_index": sample_index,
            "attack_batch_size": attack_batch_size,
            "attack_iters": attack_iters,
            "num_trials": num_trials,
            "attack_lr": attack_lr,
            "distance": distance,
            "dataset_override": dataset_override,
            "model_arch_override": model_arch_override,
        }
        with (run_dir / "attack_failed.json").open("w", encoding="utf-8") as f:
            json.dump(failure_info, f, indent=2)

    return returncode == 0, stdout_text.strip(), stderr_text.strip(), returncode


def collect_rows_from_run_dirs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []

    for run_dir in sorted(set(run_dirs)):
        metrics_path = run_dir / "attack_metrics.json"
        if not metrics_path.exists():
            failed_path = run_dir / "attack_failed.json"
            row = {
                "attack_run_dir": str(run_dir.resolve()),
                "run_status": "failed" if failed_path.exists() else "missing_metrics",
            }
            if failed_path.exists():
                try:
                    failed = json.loads(failed_path.read_text())
                    row.update(
                        {
                            "experiment_dir": failed.get("experiment_dir"),
                            "client_id": failed.get("client_id"),
                            "sample_index": failed.get("sample_index"),
                            "attack_batch_size": failed.get("attack_batch_size"),
                            "attack_iters": failed.get("attack_iters"),
                            "num_trials": failed.get("num_trials"),
                            "attack_lr": failed.get("attack_lr"),
                            "distance": failed.get("distance"),
                            "returncode": failed.get("returncode"),
                        }
                    )
                except Exception:
                    pass
            rows.append(row)
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        row = dict(metrics)
        row["attack_run_dir"] = str(run_dir.resolve())
        row["run_status"] = "ok"

        experiment_dir = Path(row.get("experiment_dir", "")).resolve()
        config = load_yaml(experiment_dir / "config.yaml")

        for key in [
            "dataset",
            "model_arch",
            "num_clients",
            "num_rounds",
            "local_epochs",
            "batch_size",
            "lr",
            "split_type",
            "alpha",
            "optimizer",
            "seed",
        ]:
            row[f"train_{key}"] = config.get(key)

        original_pt = run_dir / "original_images.pt"
        reconstructed_pt = run_dir / "reconstructed_images.pt"

        best_mse = None
        best_ssim = None

        if original_pt.exists() and reconstructed_pt.exists():
            try:
                original = torch.load(original_pt, map_location="cpu")
                reconstructed = torch.load(reconstructed_pt, map_location="cpu")
                row["original_tensor_shape"] = list(original.shape)
                row["reconstructed_tensor_shape"] = list(reconstructed.shape)
                best_mse, best_ssim = compute_best_metrics(original, reconstructed)
            except Exception as error:
                row["metric_compute_error"] = f"{type(error).__name__}: {error}"

        if best_mse is None:
            best_mse = row.get("reconstruction_mse")

        row["best_mse"] = best_mse
        row["best_ssim"] = best_ssim

        rows.append(row)

    return rows


def discover_run_dirs_by_sweep_name(
    experiment_dirs: list[str],
    sweep_name: str,
) -> list[Path]:
    run_dirs = []
    for experiment in experiment_dirs:
        attacks_dir = Path(experiment) / "attacks"
        if not attacks_dir.exists():
            continue
        run_dirs.extend(sorted(attacks_dir.glob(f"{sweep_name}_*")))
    return run_dirs


def prepare_regression_data(
    df: pd.DataFrame,
    target_metric: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    # Feature-importance from regression can only evaluate parameters that vary in the
    # selected sweep. A constant column provides no split/permutation signal and must be dropped.
    feature_columns = [
        "attack_batch_size",
        "attack_iters",
        "num_trials",
        "attack_lr",
        "distance",
        "client_id",
        "sample_index",
        "train_dataset",
        "train_model_arch",
        "train_split_type",
        "train_alpha",
        "train_num_clients",
        "train_num_rounds",
        "train_local_epochs",
        "train_batch_size",
        "train_lr",
        "train_seed",
    ]

    required_columns = feature_columns + [target_metric, "attack_status"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in aggregated data: {missing}")

    filtered = df[df["attack_status"] == "ok"].copy()
    filtered = filtered.dropna(subset=[target_metric]).copy()

    if filtered.empty:
        raise ValueError(
            "No usable rows after filtering by attack_status == 'ok' and target metric availability."
        )

    X = filtered[feature_columns].copy()
    y = filtered[target_metric].astype(float)
    groups = build_target_groups(filtered)

    for categorical_identifier in ["client_id", "sample_index", "train_seed"]:
        if categorical_identifier in X.columns:
            X[categorical_identifier] = X[categorical_identifier].astype(str)

    varying_columns = []
    dropped_constant_columns = []

    for col in feature_columns:
        if X[col].nunique(dropna=False) > 1:
            varying_columns.append(col)
        else:
            dropped_constant_columns.append(col)

    if not varying_columns:
        raise ValueError("All candidate features are constant. Add more diverse experiments/settings.")

    X = X[varying_columns]

    return X, y, groups, varying_columns, dropped_constant_columns


def build_target_groups(df: pd.DataFrame) -> pd.Series:
    group_cols = ["experiment_dir", "client_id", "sample_index", "attack_batch_size"]
    available_cols = [col for col in group_cols if col in df.columns]
    if not available_cols:
        return pd.Series(np.arange(len(df)), index=df.index, dtype=str)

    return df[available_cols].astype(str).agg("|".join, axis=1)


def bootstrap_ci(values: list[float], seed: int = 42) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [
        float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(BOOTSTRAP_REPEATS)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def metric_leakage_direction(target_metric: str) -> int:
    if target_metric == "best_mse":
        return -1
    return 1


def save_controlled_pairwise_effects(
    aggregated: pd.DataFrame,
    analysis_dir: Path,
    target_metric: str,
) -> pd.DataFrame:
    """
    Estimate parameter effects using matched contrasts.

    For each parameter, rows are paired only when all other design parameters are
    identical. This is much more defensible than interpreting a single model
    importance number as a causal effect, and it costs no extra attack runs.
    """
    if "attack_status" not in aggregated.columns or target_metric not in aggregated.columns:
        return pd.DataFrame()

    available_design_cols = [col for col in DESIGN_FEATURES if col in aggregated.columns]
    df = aggregated[aggregated["attack_status"] == "ok"].copy()
    df = df.dropna(subset=[target_metric])

    rows = []
    leakage_direction = metric_leakage_direction(target_metric)

    for parameter in available_design_cols:
        if df[parameter].nunique(dropna=False) < 2:
            continue

        nuisance_cols = [col for col in available_design_cols if col != parameter]
        if not nuisance_cols:
            continue

        deltas_by_pair: dict[tuple[str, str], list[float]] = {}

        for _, group in df.groupby(nuisance_cols, dropna=False):
            level_values = (
                group.groupby(parameter, dropna=False)[target_metric]
                .mean()
                .sort_index(key=lambda idx: idx.astype(str))
            )

            if len(level_values) < 2:
                continue

            levels = list(level_values.index)
            for lower_idx, higher_idx in itertools.combinations(range(len(levels)), 2):
                baseline_level = levels[lower_idx]
                comparison_level = levels[higher_idx]
                pair_key = (str(baseline_level), str(comparison_level))
                delta = float(level_values.iloc[higher_idx] - level_values.iloc[lower_idx])
                deltas_by_pair.setdefault(pair_key, []).append(delta)

        for (baseline_level, comparison_level), deltas in deltas_by_pair.items():
            mean_delta = float(np.mean(deltas))
            ci_low, ci_high = bootstrap_ci(deltas)
            leakage_ci_values = [
                value * leakage_direction if value is not None else None
                for value in [ci_low, ci_high]
            ]
            leakage_ci_values = [
                value for value in leakage_ci_values if value is not None
            ]
            ci_low_leakage = min(leakage_ci_values) if leakage_ci_values else None
            ci_high_leakage = max(leakage_ci_values) if leakage_ci_values else None
            rows.append(
                {
                    "parameter": parameter,
                    "baseline_level": baseline_level,
                    "comparison_level": comparison_level,
                    "target_metric": target_metric,
                    "n_matched_contrasts": len(deltas),
                    "mean_metric_delta": mean_delta,
                    "ci95_low_metric_delta": ci_low,
                    "ci95_high_metric_delta": ci_high,
                    "mean_leakage_delta": leakage_direction * mean_delta,
                    "ci95_low_leakage_delta": ci_low_leakage,
                    "ci95_high_leakage_delta": ci_high_leakage,
                    "leakage_direction_note": (
                        "Positive mean_leakage_delta means stronger reconstruction. "
                        "For best_mse this is -metric_delta; for best_ssim this is metric_delta."
                    ),
                }
            )

    effects = pd.DataFrame(rows)
    if not effects.empty:
        effects = effects.sort_values(
            ["parameter", "baseline_level", "comparison_level"],
            kind="stable",
        )
    effects.to_csv(analysis_dir / "controlled_pairwise_effects.csv", index=False)
    return effects


def compact_level_label(value: Any, max_chars: int = 36) -> str:
    text = str(value)
    if "/" in text:
        text = Path(text).name
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def readable_contrast_label(row: pd.Series) -> str:
    baseline = compact_level_label(row["baseline_level"])
    comparison = compact_level_label(row["comparison_level"])
    return f"{row['parameter']}: {baseline} -> {comparison}"


def save_parameter_importance_plot(
    importance_df: pd.DataFrame,
    analysis_dir: Path,
    top_n: int = 12,
) -> Path | None:
    if importance_df.empty:
        return None

    plot_df = importance_df.head(top_n).iloc[::-1].copy()
    fig_height = max(4.0, 0.45 * len(plot_df) + 1.5)

    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(
        plot_df["parameter"],
        plot_df["importance_mean_mae_increase"],
        xerr=plot_df["importance_std_mae_increase"],
        color="#4C78A8",
        ecolor="#2F3A45",
        capsize=3,
    )
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Predictive Parameter Importance")
    ax.set_xlabel("Permutation importance: MAE increase")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.88, bottom=0.14)

    path = analysis_dir / "parameter_importance.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_controlled_effects_plot(
    effects_df: pd.DataFrame,
    analysis_dir: Path,
    top_n: int = 16,
) -> Path | None:
    required_cols = {
        "mean_leakage_delta",
        "ci95_low_leakage_delta",
        "ci95_high_leakage_delta",
        "n_matched_contrasts",
    }
    if effects_df.empty or not required_cols.issubset(effects_df.columns):
        return None

    plot_df = effects_df.copy()
    plot_df = plot_df.dropna(
        subset=["mean_leakage_delta", "ci95_low_leakage_delta", "ci95_high_leakage_delta"]
    )
    if plot_df.empty:
        return None

    plot_df["abs_effect"] = plot_df["mean_leakage_delta"].abs()
    plot_df = plot_df.sort_values("abs_effect", ascending=False).head(top_n)
    plot_df = plot_df.iloc[::-1].copy()
    plot_df["label"] = plot_df.apply(readable_contrast_label, axis=1)

    y_positions = np.arange(len(plot_df))
    x_values = plot_df["mean_leakage_delta"].to_numpy(dtype=float)
    ci_low_leakage = plot_df["ci95_low_leakage_delta"].to_numpy(dtype=float)
    ci_high_leakage = plot_df["ci95_high_leakage_delta"].to_numpy(dtype=float)
    left_errors = x_values - ci_low_leakage
    right_errors = ci_high_leakage - x_values

    fig_height = max(5.0, 0.5 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = np.where(x_values >= 0, "#2F855A", "#C05621")
    ax.barh(y_positions, x_values, color=colors, alpha=0.85)
    ax.errorbar(
        x_values,
        y_positions,
        xerr=np.vstack([left_errors, right_errors]),
        fmt="none",
        ecolor="#333333",
        capsize=3,
        linewidth=1,
    )
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["label"])
    ax.set_title("Controlled Matched Contrasts")
    ax.set_xlabel("Mean leakage delta; positive means stronger reconstruction")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)

    for y_pos, (_, row) in zip(y_positions, plot_df.iterrows(), strict=False):
        ax.text(
            0.01,
            y_pos,
            f"n={int(row['n_matched_contrasts'])}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8,
            color="#222222",
        )

    fig.subplots_adjust(left=0.36, right=0.97, top=0.9, bottom=0.12)

    path = analysis_dir / "controlled_pairwise_effects.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def split_train_test_with_groups(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    if len(X) < 2:
        raise ValueError("At least 2 regression rows are required.")

    n_unique_groups = groups.nunique(dropna=False)
    if n_unique_groups >= 2 and len(X) >= 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        return (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
            "group_shuffle_by_experiment_client_sample_batch",
        )

    test_size_count = max(1, int(round(0.2 * len(X))))
    if test_size_count >= len(X):
        test_size_count = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size_count,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test, "random_row_split_fallback"


def grouped_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    baseline = mean_absolute_error(y_test, pipeline.predict(X_test))

    rows = []

    for column in X_test.columns:
        deltas = []

        for _ in range(repeats):
            permuted = X_test.copy()
            permuted[column] = rng.permutation(permuted[column].values)
            score = mean_absolute_error(y_test, pipeline.predict(permuted))
            deltas.append(score - baseline)

        rows.append(
            {
                "parameter": column,
                "importance_mean_mae_increase": float(np.mean(deltas)),
                "importance_std_mae_increase": float(np.std(deltas)),
            }
        )

    return pd.DataFrame(rows).sort_values("importance_mean_mae_increase", ascending=False)


def save_group_summaries(aggregated: pd.DataFrame, analysis_dir: Path) -> None:
    metric_cols = [col for col in ["best_mse", "best_ssim", "reconstruction_mse"] if col in aggregated.columns]

    group_specs = {
        "group_summary_by_dataset_model.csv": ["train_dataset", "train_model_arch"],
        "group_summary_by_split.csv": ["train_split_type", "train_alpha"],
        "group_summary_by_batch_size.csv": ["attack_batch_size"],
        "group_summary_by_distance.csv": ["distance"],
        "group_summary_by_attack_iters.csv": ["attack_iters"],
        "group_summary_by_client.csv": ["client_id"],
    }

    for filename, group_cols in group_specs.items():
        existing_group_cols = [col for col in group_cols if col in aggregated.columns]
        if not existing_group_cols or not metric_cols:
            continue

        summary = (
            aggregated.groupby(existing_group_cols, dropna=False)[metric_cols]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
        )

        summary.columns = [
            "_".join(str(part) for part in col if part != "") if isinstance(col, tuple) else str(col)
            for col in summary.columns
        ]

        summary.to_csv(analysis_dir / filename, index=False)


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1.")
    if args.max_runs is not None and args.max_runs < 1:
        raise ValueError("--max-runs must be >= 1 when provided.")
    if args.analysis_only and args.dry_run_design:
        raise ValueError("--dry-run-design cannot be combined with --analysis-only.")
    if args.jobs > 2:
        print(
            "WARNING: --jobs > 2 may be memory-heavy on a laptop. "
            "For MacBook Air class hardware, --jobs 1 or 2 is usually safer."
        )

    resolved_device = select_device(args.device)
    if resolved_device == "mps":
        print(
            "WARNING: Using MPS. Depending on AIJack/PyTorch operators, MPS may or may not "
            "improve runtime versus CPU."
        )

    attack_script_path = Path(args.attack_script)
    if not attack_script_path.is_absolute():
        attack_script_path = Path(__file__).resolve().parent / attack_script_path
    attack_script_path = attack_script_path.resolve()
    if args.execution_mode == "subprocess" and not attack_script_path.exists():
        raise FileNotFoundError(f"Attack script not found: {attack_script_path}")

    output_root = Path(args.output_root)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.sweep_name:
        sweep_name = sanitize_name(args.sweep_name)
    else:
        sweep_name = f"sweep_{run_stamp}"

    analysis_dir = output_root / sweep_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    design_report_path = analysis_dir / "design_report.json"
    design_csv_path = analysis_dir / "design_selected_runs.csv"

    full_combos: list[dict[str, Any]] = []
    selected_combos: list[dict[str, Any]] = []
    design_cap_used: int | None = None
    design_report: dict[str, Any] | None = None
    run_records: list[dict[str, Any]] = []
    current_run_dirs: list[Path] = []

    if not args.analysis_only:
        full_combos, selected_combos, design_cap_used = build_selected_combos(
            args,
            sweep_name=sweep_name,
        )
        if not selected_combos:
            raise ValueError("No design combinations selected. Check --max-runs and design settings.")

        design_report = build_design_report(
            full_combos=full_combos,
            selected_combos=selected_combos,
            design=args.design,
            ensure_varies=args.ensure_varies,
            design_seed=args.design_seed,
            max_runs=design_cap_used,
        )
        with design_report_path.open("w", encoding="utf-8") as f:
            json.dump(design_report, f, indent=2)

        if args.save_design_csv:
            pd.DataFrame([combo_to_row(combo) for combo in selected_combos]).to_csv(
                design_csv_path,
                index=False,
            )

        print("\nDesign preflight report:")
        print(
            f"- full-factorial combinations: {design_report['n_total_full_factorial_combinations']}"
        )
        print(f"- selected combinations: {design_report['n_selected_combinations']}")
        print(f"- report path: {design_report_path.resolve()}")
        for feature in DESIGN_FEATURES:
            feature_info = design_report["parameters"][feature]
            print(
                f"  - {feature}: n_unique={feature_info['n_unique_selected']}, "
                f"values={feature_info['values_selected']}, "
                f"assessable={feature_info['assessable_by_feature_importance']}"
            )

        if args.save_design_csv:
            print(f"- design csv: {design_csv_path.resolve()}")

        for warning in design_report["warnings"]:
            print(f"WARNING: {warning}")

        if args.dry_run_design:
            print("\nDry-run complete. No attacks were executed.")
            return

        current_run_dirs = [combo["run_dir"] for combo in selected_combos]

    if not args.analysis_only:
        combos_to_run = []
        for combo in selected_combos:
            run_dir = combo["run_dir"]
            metrics_path = run_dir / "attack_metrics.json"
            if metrics_path.exists() and not args.rerun_existing:
                run_records.append(
                    {
                        **{k: v for k, v in combo.items() if k != "run_dir"},
                        "run_dir": str(run_dir),
                        "status": "skipped_existing",
                        "returncode": 0,
                    }
                )
                continue
            combos_to_run.append(combo)

        def execute_combo(combo: dict[str, Any]) -> dict[str, Any]:
            ok, stdout, stderr, returncode = run_attack(
                execution_mode=args.execution_mode,
                attack_script_path=attack_script_path,
                experiment_dir=combo["experiment_dir"],
                run_dir=combo["run_dir"],
                client_id=combo["client_id"],
                sample_index=combo["sample_index"],
                attack_batch_size=combo["attack_batch_size"],
                attack_iters=combo["attack_iters"],
                num_trials=combo["num_trials"],
                attack_lr=combo["attack_lr"],
                distance=combo["distance"],
                device=resolved_device,
                dataset_override=args.dataset,
                model_arch_override=args.model_arch,
            )
            return {
                **{k: v for k, v in combo.items() if k != "run_dir"},
                "run_dir": str(combo["run_dir"]),
                "status": "ok" if ok else "failed",
                "returncode": returncode,
                "stdout_tail": stdout[-500:],
                "stderr_tail": stderr[-500:],
            }

        if args.jobs == 1:
            iterator = tqdm(combos_to_run, desc=f"Running attacks [{sweep_name}]")
            for combo in iterator:
                run_records.append(execute_combo(combo))
        else:
            progress = tqdm(total=len(combos_to_run), desc=f"Running attacks [{sweep_name}]")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
                future_to_combo = {executor.submit(execute_combo, combo): combo for combo in combos_to_run}
                for future in concurrent.futures.as_completed(future_to_combo):
                    combo = future_to_combo[future]
                    try:
                        run_records.append(future.result())
                    except Exception as error:
                        run_records.append(
                            {
                                **{k: v for k, v in combo.items() if k != "run_dir"},
                                "run_dir": str(combo["run_dir"]),
                                "status": "failed",
                                "returncode": -1,
                                "stdout_tail": "",
                                "stderr_tail": f"{type(error).__name__}: {error}",
                            }
                        )
                    progress.update(1)
            progress.close()

    if args.analysis_only:
        current_run_dirs = discover_run_dirs_by_sweep_name(args.experiment_dirs, sweep_name)

    if not current_run_dirs:
        raise ValueError(f"No run directories found for sweep_name={sweep_name}")

    all_rows = collect_rows_from_run_dirs(current_run_dirs)

    if not all_rows:
        raise ValueError(
            "No attack metrics were found for the current sweep. "
            "Run without --analysis-only first, or verify --sweep-name."
        )

    aggregated = pd.DataFrame(all_rows)
    aggregated_path = analysis_dir / "aggregated_attack_results.csv"
    aggregated.to_csv(aggregated_path, index=False)

    if run_records:
        run_df = pd.DataFrame(run_records)
        run_manifest_path = analysis_dir / "run_manifest.csv"
        run_df.to_csv(run_manifest_path, index=False)

    save_group_summaries(aggregated, analysis_dir)
    controlled_effects = save_controlled_pairwise_effects(
        aggregated=aggregated,
        analysis_dir=analysis_dir,
        target_metric=args.target_metric,
    )

    regression_warning = None

    X, y, groups, varying_columns, dropped_constant_columns = prepare_regression_data(
        aggregated,
        target_metric=args.target_metric,
    )

    if len(X) < 50:
        regression_warning = (
            f"Only {len(X)} usable regression rows. Parameter importance is unstable "
            "and should be used only for debugging/exploratory screening."
        )
        print(f"WARNING: {regression_warning}")

    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test, validation_split = split_train_test_with_groups(
        X,
        y,
        groups,
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, preds))
    test_r2 = float(r2_score(y_test, preds)) if len(y_test) >= 2 else None

    importance_df = grouped_permutation_importance(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        repeats=20,
        random_state=42,
    )

    importance_path = analysis_dir / "parameter_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    importance_plot_path = save_parameter_importance_plot(
        importance_df=importance_df,
        analysis_dir=analysis_dir,
    )
    controlled_effects_plot_path = save_controlled_effects_plot(
        effects_df=controlled_effects,
        analysis_dir=analysis_dir,
    )

    top_rows = importance_df.head(10)

    summary = {
        "timestamp": run_stamp,
        "sweep_name": sweep_name,
        "analysis_dir": str(analysis_dir.resolve()),
        "design": args.design,
        "design_report_path": str(design_report_path.resolve()) if design_report_path.exists() else None,
        "target_metric": args.target_metric,
        "execution_mode": args.execution_mode,
        "dataset_override": args.dataset,
        "model_arch_override": args.model_arch,
        "n_total_rows": int(len(aggregated)),
        "n_regression_rows": int(len(X)),
        "validation_split": validation_split,
        "n_features_used": int(len(varying_columns)),
        "features_used": varying_columns,
        "dropped_constant_features": dropped_constant_columns,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "regression_warning": regression_warning,
        "controlled_pairwise_effects_path": str((analysis_dir / "controlled_pairwise_effects.csv").resolve()),
        "controlled_pairwise_effects_plot_path": (
            str(controlled_effects_plot_path.resolve())
            if controlled_effects_plot_path is not None
            else None
        ),
        "n_controlled_pairwise_effect_rows": int(len(controlled_effects)),
        "parameter_importance_plot_path": (
            str(importance_plot_path.resolve())
            if importance_plot_path is not None
            else None
        ),
        "top_parameters": top_rows.to_dict(orient="records"),
        "permutation_importance_scope_note": (
            "Permutation importance is only defined for parameters that varied in the selected sweep. "
            "It is a predictive screening statistic, not a causal estimate."
        ),
        "controlled_effects_scope_note": (
            "controlled_pairwise_effects.csv reports matched contrasts where all other design "
            "parameters are held fixed. Prefer these effect sizes for thesis claims when enough "
            "matched contrasts are available."
        ),
        "important_note": (
            # This remains screening-oriented: observational sweeps can rank sensitivity,
            # but they do not establish causal effects without controlled causal design.
            "This analysis is exploratory. Treat random-forest importance as screening; "
            "use controlled matched contrasts and confirmation runs for claims."
        ),
    }

    summary_path = analysis_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    provenance = collect_provenance(
        extra={
            "script": "attack_parameter_impact_bloodmnist.py",
            "sweep_name": sweep_name,
            "execution_mode": args.execution_mode,
        }
    )
    with (analysis_dir / "provenance.json").open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    print("Exploratory attack-impact analysis completed.")
    print(f"Sweep name: {sweep_name}")
    print(f"Analysis directory: {analysis_dir.resolve()}")
    print(f"Aggregated results: {aggregated_path.resolve()}")

    if run_records:
        print(f"Run manifest: {(analysis_dir / 'run_manifest.csv').resolve()}")

    if design_report_path.exists():
        print(f"Design report: {design_report_path.resolve()}")

    print(f"Controlled pairwise effects: {(analysis_dir / 'controlled_pairwise_effects.csv').resolve()}")
    print(f"Parameter importance: {importance_path.resolve()}")
    if controlled_effects_plot_path is not None:
        print(f"Controlled effects plot: {controlled_effects_plot_path.resolve()}")
    if importance_plot_path is not None:
        print(f"Parameter importance plot: {importance_plot_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Validation split: {validation_split}")

    print("\nTop parameters by grouped permutation importance:")
    for _, row in top_rows.iterrows():
        print(
            f"- {row['parameter']}: mean_mae_increase="
            f"{row['importance_mean_mae_increase']:.6f}, "
            f"std_mae_increase={row['importance_std_mae_increase']:.6f}"
        )


if __name__ == "__main__":
    main()
