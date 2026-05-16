from __future__ import annotations

import concurrent.futures
import json
import re
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from thesis.attacks import run_attack
from thesis.experiments.config import AttackConfig, SweepConfig
from thesis.metrics import compute_best_reconstruction_metrics
from thesis.utils import collect_provenance, ensure_dir, load_yaml, save_json, timestamp


def _sanitize_name(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _float_tag(value: float) -> str:
    text = f"{value:g}"
    return re.sub(r"[^0-9A-Za-z]+", "", text)


def _build_attack_run_dir(
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
        f"{num_trials}trials_lr{_float_tag(attack_lr)}_"
        f"client{client_id}_sample{sample_index}"
    )
    return experiment_dir / "attacks" / run_name


def _run_attack_cell(experiment_dir: Path, run_dir: Path, attack_cfg: AttackConfig) -> tuple[bool, str, str, int]:
    ensure_dir(run_dir)

    try:
        metrics = run_attack(experiment_dir=experiment_dir, cfg=attack_cfg, output_dir=run_dir)
        stdout_text = json.dumps({"status": metrics.get("attack_status", "ok")})
        stderr_text = ""
        returncode = 0
    except Exception as error:
        stdout_text = ""
        stderr_text = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        returncode = 1

    (run_dir / "attack_stdout.txt").write_text(stdout_text, encoding="utf-8")
    (run_dir / "attack_stderr.txt").write_text(stderr_text, encoding="utf-8")

    if returncode != 0:
        failure = {
            "timestamp": datetime.now().isoformat(),
            "returncode": returncode,
            "stderr": stderr_text,
            "attack_config": asdict(attack_cfg),
            "experiment_dir": str(experiment_dir),
        }
        save_json(run_dir / "attack_failed.json", failure)

    return returncode == 0, stdout_text, stderr_text, returncode


def _collect_rows_from_run_dirs(run_dirs: list[Path]) -> list[dict[str, Any]]:
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
                    failed = json.loads(failed_path.read_text(encoding="utf-8"))
                    row.update({"returncode": failed.get("returncode")})
                    attack_cfg = failed.get("attack_config") or {}
                    row.update(attack_cfg)
                except Exception:
                    pass
            rows.append(row)
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        row = dict(metrics)
        row["attack_run_dir"] = str(run_dir.resolve())
        row["run_status"] = "ok"

        experiment_dir = Path(row.get("experiment_dir", "")).resolve()
        train_cfg = load_yaml(experiment_dir / "config.yaml")
        for key in [
            "dataset",
            "model_name",
            "num_clients",
            "num_rounds",
            "local_epochs",
            "batch_size",
            "lr",
            "split_type",
            "alpha",
            "seed",
        ]:
            row[f"train_{key}"] = train_cfg.get(key)

        original_pt = run_dir / "original_images.pt"
        reconstructed_pt = run_dir / "reconstructed_images.pt"
        best_mse = None
        best_ssim = None

        if original_pt.exists() and reconstructed_pt.exists():
            try:
                original = torch.load(original_pt, map_location="cpu")
                reconstructed = torch.load(reconstructed_pt, map_location="cpu")
                best_mse, best_ssim = compute_best_reconstruction_metrics(original, reconstructed)
            except Exception as error:
                row["metric_compute_error"] = f"{type(error).__name__}: {error}"

        if best_mse is None:
            best_mse = row.get("reconstruction_mse")

        row["best_mse"] = best_mse
        row["best_ssim"] = best_ssim
        rows.append(row)

    return rows


def _prepare_regression_data(df: pd.DataFrame, target_metric: str):
    feature_columns = [
        "attack_batch_size",
        "attack_iters",
        "num_trials",
        "attack_lr",
        "distance",
        "client_id",
        "sample_index",
        "train_dataset",
        "train_model_name",
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
        raise ValueError("No usable rows after filtering by attack_status == 'ok' and target availability.")

    X = filtered[feature_columns].copy()
    y = filtered[target_metric].astype(float)

    for col in ["client_id", "sample_index", "train_seed"]:
        if col in X.columns:
            X[col] = X[col].astype(str)

    varying_columns = []
    dropped_columns = []

    for col in feature_columns:
        if X[col].nunique(dropna=False) > 1:
            varying_columns.append(col)
        else:
            dropped_columns.append(col)

    if not varying_columns:
        raise ValueError("All candidate features are constant.")

    X = X[varying_columns]
    return X, y, varying_columns, dropped_columns


def run_sweep(cfg: SweepConfig, target_metric: str = "best_ssim") -> dict[str, Any]:
    if cfg.jobs < 1:
        raise ValueError("jobs must be >= 1")

    run_stamp = timestamp()
    sweep_name = _sanitize_name(cfg.sweep_name) if cfg.sweep_name else f"sweep_{run_stamp}"

    output_root = Path(cfg.output_root)
    analysis_dir = ensure_dir(output_root / sweep_name)

    combos = []
    for experiment in cfg.experiment_dirs or []:
        experiment_dir = Path(experiment)
        for client_id in cfg.client_ids or []:
            for sample_index in cfg.sample_indices or []:
                for attack_batch_size in cfg.attack_batch_sizes or []:
                    for attack_iters in cfg.attack_iters or []:
                        for num_trials in cfg.num_trials or []:
                            for attack_lr in cfg.attack_lrs or []:
                                for distance in cfg.distances or []:
                                    run_dir = _build_attack_run_dir(
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
                                            "attack_cfg": AttackConfig(
                                                client_id=client_id,
                                                sample_index=sample_index,
                                                attack_batch_size=attack_batch_size,
                                                attack_iters=attack_iters,
                                                num_trials=num_trials,
                                                attack_lr=attack_lr,
                                                distance=distance,
                                                device=str(cfg.device),
                                                dataset=cfg.dataset,
                                                model_name=cfg.model_name,
                                            ),
                                        }
                                    )

    if cfg.max_runs is not None:
        combos = combos[: cfg.max_runs]

    run_records = []
    run_dirs = [combo["run_dir"] for combo in combos]

    combos_to_run = []
    for combo in combos:
        metrics_path = combo["run_dir"] / "attack_metrics.json"
        if metrics_path.exists() and not cfg.rerun_existing:
            run_records.append({
                "run_dir": str(combo["run_dir"]),
                "status": "skipped_existing",
                "returncode": 0,
            })
            continue
        combos_to_run.append(combo)

    def execute(combo: dict[str, Any]) -> dict[str, Any]:
        ok, stdout, stderr, returncode = _run_attack_cell(
            experiment_dir=combo["experiment_dir"],
            run_dir=combo["run_dir"],
            attack_cfg=combo["attack_cfg"],
        )
        return {
            "run_dir": str(combo["run_dir"]),
            "status": "ok" if ok else "failed",
            "returncode": returncode,
            "stdout_tail": stdout[-500:],
            "stderr_tail": stderr[-500:],
        }

    if cfg.jobs == 1:
        for combo in combos_to_run:
            run_records.append(execute(combo))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.jobs) as executor:
            futures = {executor.submit(execute, combo): combo for combo in combos_to_run}
            for future in concurrent.futures.as_completed(futures):
                combo = futures[future]
                try:
                    run_records.append(future.result())
                except Exception as error:
                    run_records.append(
                        {
                            "run_dir": str(combo["run_dir"]),
                            "status": "failed",
                            "returncode": -1,
                            "stdout_tail": "",
                            "stderr_tail": f"{type(error).__name__}: {error}",
                        }
                    )

    all_rows = _collect_rows_from_run_dirs(run_dirs)
    aggregated = pd.DataFrame(all_rows)
    aggregated_path = analysis_dir / "aggregated_attack_results.csv"
    aggregated.to_csv(aggregated_path, index=False)

    run_manifest_path = analysis_dir / "run_manifest.csv"
    pd.DataFrame(run_records).to_csv(run_manifest_path, index=False)

    X, y, features_used, dropped_features = _prepare_regression_data(aggregated, target_metric)

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

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    n_rows = len(X)
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

    summary = {
        "timestamp": run_stamp,
        "sweep_name": sweep_name,
        "analysis_dir": str(analysis_dir.resolve()),
        "target_metric": target_metric,
        "n_total_rows": int(len(aggregated)),
        "n_regression_rows": int(len(X)),
        "n_features_used": int(len(features_used)),
        "features_used": features_used,
        "dropped_constant_features": dropped_features,
        "test_mae": float(mean_absolute_error(y_test, preds)),
        "test_r2": float(r2_score(y_test, preds)) if len(y_test) >= 2 else None,
    }

    summary_path = analysis_dir / "summary.json"
    save_json(summary_path, summary)

    save_json(
        analysis_dir / "provenance.json",
        collect_provenance(
            extra={
                "module": "thesis.experiments.runner",
                "sweep_name": sweep_name,
            }
        ),
    )

    return {
        "sweep_name": sweep_name,
        "analysis_dir": str(analysis_dir.resolve()),
        "aggregated_path": str(aggregated_path.resolve()),
        "run_manifest_path": str(run_manifest_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }
