import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


DEFAULT_EXPERIMENT_DIRS = [
    "results/iid_baseline",
    "results/noniid_alpha_1",
    "results/noniid_alpha_05",
    "results/noniid_alpha_01",
]


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
        help="Optional cap for debugging/pilot runs.",
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
        default="results/attack_parameter_impact",
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
        "--attack-script",
        type=str,
        default="gradient_inversion_bloodmnist_aijack.py",
        help=(
            "Attack script to execute for each run. Relative paths are resolved "
            "from the scripts directory."
        ),
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


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def compute_ssim_per_image(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Global SSIM-like score per image/channel, then averaged across channels.

    This is NOT windowed SSIM. It is acceptable for exploratory screening,
    but should be described as a global SSIM-like similarity proxy.
    """
    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = original.mean(dim=(-1, -2), keepdim=True)
    mu_y = reconstructed.mean(dim=(-1, -2), keepdim=True)

    sigma_x = ((original - mu_x) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_y = ((reconstructed - mu_y) ** 2).mean(dim=(-1, -2), keepdim=True)
    sigma_xy = ((original - mu_x) * (reconstructed - mu_y)).mean(
        dim=(-1, -2), keepdim=True
    )

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator

    return ssim_map.squeeze(-1).squeeze(-1).mean(dim=1)


def compute_best_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> tuple[float | None, float | None]:
    """
    Computes best MSE and best SSIM-like score.

    If reconstructed contains multiple candidates/trials stacked as:
    [num_candidates * batch_size, C, H, W],
    this function compares each candidate batch to the original batch and keeps:
    - minimum MSE
    - maximum SSIM-like score
    """
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
    python_exe: str,
    script_path: Path,
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
) -> tuple[bool, str, str, int]:
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        str(script_path),
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

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    with (run_dir / "attack_stdout.txt").open("w", encoding="utf-8") as f:
        f.write(result.stdout or "")

    with (run_dir / "attack_stderr.txt").open("w", encoding="utf-8") as f:
        f.write(result.stderr or "")

    if result.returncode != 0:
        failure_info = {
            "timestamp": datetime.now().isoformat(),
            "command": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "experiment_dir": str(experiment_dir),
            "client_id": client_id,
            "sample_index": sample_index,
            "attack_batch_size": attack_batch_size,
            "attack_iters": attack_iters,
            "num_trials": num_trials,
            "attack_lr": attack_lr,
            "distance": distance,
        }
        with (run_dir / "attack_failed.json").open("w", encoding="utf-8") as f:
            json.dump(failure_info, f, indent=2)

    return result.returncode == 0, result.stdout.strip(), result.stderr.strip(), result.returncode


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
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    feature_columns = [
        "attack_batch_size",
        "attack_iters",
        "num_trials",
        "attack_lr",
        "distance",
        "client_id",
        "sample_index",
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

    # Treat identifiers as categorical, not numeric quantities.
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

    return X, y, varying_columns, dropped_constant_columns


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

    return pd.DataFrame(rows).sort_values(
        "importance_mean_mae_increase",
        ascending=False,
    )


def save_group_summaries(aggregated: pd.DataFrame, analysis_dir: Path) -> None:
    metric_cols = [col for col in ["best_mse", "best_ssim", "reconstruction_mse"] if col in aggregated.columns]

    group_specs = {
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

        # Flatten multi-index columns.
        summary.columns = [
            "_".join(str(part) for part in col if part != "")
            if isinstance(col, tuple)
            else str(col)
            for col in summary.columns
        ]

        summary.to_csv(analysis_dir / filename, index=False)


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1.")

    script_path = Path(args.attack_script)
    if not script_path.is_absolute():
        script_path = Path(__file__).resolve().parent / script_path
    script_path = script_path.resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Attack script not found: {script_path}")

    output_root = Path(args.output_root)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.sweep_name:
        sweep_name = sanitize_name(args.sweep_name)
    else:
        sweep_name = f"sweep_{run_stamp}"

    analysis_dir = output_root / sweep_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    combos = []

    for experiment in args.experiment_dirs:
        experiment_dir = Path(experiment)
        for client_id in args.client_ids:
            for sample_index in args.sample_indices:
                for attack_batch_size in args.attack_batch_sizes:
                    for attack_iters in args.attack_iters:
                        for num_trials in args.num_trials:
                            for attack_lr in args.attack_lrs:
                                for distance in args.distances:
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

    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    run_records = []
    current_run_dirs = [combo["run_dir"] for combo in combos]

    if not args.analysis_only:
        combos_to_run = []
        for combo in combos:
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
                python_exe=sys.executable,
                script_path=script_path,
                experiment_dir=combo["experiment_dir"],
                run_dir=combo["run_dir"],
                client_id=combo["client_id"],
                sample_index=combo["sample_index"],
                attack_batch_size=combo["attack_batch_size"],
                attack_iters=combo["attack_iters"],
                num_trials=combo["num_trials"],
                attack_lr=combo["attack_lr"],
                distance=combo["distance"],
                device=args.device,
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
                future_to_combo = {
                    executor.submit(execute_combo, combo): combo for combo in combos_to_run
                }
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

    regression_warning = None

    X, y, varying_columns, dropped_constant_columns = prepare_regression_data(
        aggregated,
        target_metric=args.target_metric,
    )

    if len(X) < 50:
        regression_warning = (
            f"Only {len(X)} usable regression rows. Parameter importance is unstable "
            "and should be used only for debugging/exploratory screening."
        )
        print(f"WARNING: {regression_warning}")

    numeric_cols = [
        col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
    ]
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

    n_rows = len(X)
    if n_rows < 2:
        raise ValueError("At least 2 regression rows are required.")

    test_size_count = max(1, int(round(0.2 * n_rows)))
    if test_size_count >= n_rows:
        test_size_count = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size_count,
        random_state=42,
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

    top_rows = importance_df.head(10)

    summary = {
        "timestamp": run_stamp,
        "sweep_name": sweep_name,
        "analysis_dir": str(analysis_dir.resolve()),
        "target_metric": args.target_metric,
        "n_total_rows": int(len(aggregated)),
        "n_regression_rows": int(len(X)),
        "n_features_used": int(len(varying_columns)),
        "features_used": varying_columns,
        "dropped_constant_features": dropped_constant_columns,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "regression_warning": regression_warning,
        "top_parameters": top_rows.to_dict(orient="records"),
        "important_note": (
            "This analysis is exploratory and should be used for parameter screening, "
            "not causal proof."
        ),
    }

    summary_path = analysis_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Exploratory attack-impact analysis completed.")
    print(f"Sweep name: {sweep_name}")
    print(f"Analysis directory: {analysis_dir.resolve()}")
    print(f"Aggregated results: {aggregated_path.resolve()}")

    if run_records:
        print(f"Run manifest: {(analysis_dir / 'run_manifest.csv').resolve()}")

    print(f"Parameter importance: {importance_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")

    print("\nTop parameters by grouped permutation importance:")
    for _, row in top_rows.iterrows():
        print(
            f"- {row['parameter']}: mean_mae_increase="
            f"{row['importance_mean_mae_increase']:.6f}, "
            f"std_mae_increase={row['importance_std_mae_increase']:.6f}"
        )


if __name__ == "__main__":
    main()
