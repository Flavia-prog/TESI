import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DEFAULT_SIGMAS = [0.25, 0.5, 0.75, 1.0, 2.0]
DEFAULT_SPLITS = ["iid", "noniid_alpha_1", "noniid_alpha_05", "noniid_alpha_01"]
DEFAULT_CLIENT_IDS = [0]
DEFAULT_SAMPLE_INDICES = [0, 25, 50, 75, 100]

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "fedavg_bloodmnist_aijack_dp.py"
ATTACK_SCRIPT = SCRIPT_DIR / "gradient_inversion_bloodmnist_aijack.py"

SPLIT_SPECS: dict[str, dict[str, Any]] = {
    "iid": {
        "split_type": "iid",
        "alpha": 0.5,
        "base_no_dp_dir": Path("results/iid_baseline"),
        "experiment_prefix": "iid",
    },
    "noniid_alpha_1": {
        "split_type": "dirichlet",
        "alpha": 1.0,
        "base_no_dp_dir": Path("results/noniid_alpha_1"),
        "experiment_prefix": "noniid_alpha_1",
    },
    "noniid_alpha_05": {
        "split_type": "dirichlet",
        "alpha": 0.5,
        "base_no_dp_dir": Path("results/noniid_alpha_05"),
        "experiment_prefix": "noniid_alpha_05",
    },
    "noniid_alpha_01": {
        "split_type": "dirichlet",
        "alpha": 0.1,
        "base_no_dp_dir": Path("results/noniid_alpha_01"),
        "experiment_prefix": "noniid_alpha_01",
    },
}

SUMMARY_COLUMNS = [
    "split_label",
    "split_type",
    "alpha",
    "sigma",
    "dp_enabled",
    "experiment_name",
    "experiment_dir",
    "epsilon",
    "delta",
    "sampling_rate",
    "clients_per_round",
    "clip_norm",
    "noise_multiplier",
    "effective_noise_multiplier",
    "actual_noise_std",
    "test_accuracy",
    "test_macro_f1",
    "test_loss",
    "best_round",
    "best_val_loss",
    "client_id",
    "sample_index",
    "attack_batch_size",
    "attack_iters",
    "num_trials",
    "attack_lr",
    "distance",
    "attack_status",
    "attack_error",
    "reconstruction_mse",
    "number_of_reconstructions",
    "original_shape",
    "reconstructed_shape",
    "attack_output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full privacy-utility matrix for BloodMNIST FedAvg-DP "
            "with optional training, attacks, and aggregated reporting."
        )
    )
    parser.add_argument("--sigmas", nargs="+", type=float, default=DEFAULT_SIGMAS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--client-ids", nargs="+", type=int, default=DEFAULT_CLIENT_IDS)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument("--attack-iters", type=int, default=1000)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--attack-lr", type=float, default=0.1)
    parser.add_argument("--distance", choices=["l2", "cossim"], default="cossim")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-attacks", action="store_true")
    parser.add_argument("--rerun-training", action="store_true")
    parser.add_argument("--rerun-attacks", action="store_true")

    parser.add_argument(
        "--include-baseline-attacks",
        action="store_true",
        help="Accepted for compatibility; baseline attacks are included by default.",
    )
    parser.add_argument("--no-baseline-attacks", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="results/full_dp_privacy_utility_matrix",
        help="Directory for final summary/report artifacts.",
    )
    return parser.parse_args()


def sigma_tag(sigma: float) -> str:
    text = f"{sigma:g}"
    return re.sub(r"[^0-9A-Za-z]+", "", text)


def float_tag(value: float) -> str:
    text = f"{value:g}"
    return re.sub(r"[^0-9A-Za-z]+", "", text)


def now_iso() -> str:
    return datetime.now().isoformat()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                return loaded
    except Exception:
        return {}
    return {}


def read_first_csv_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        if isinstance(value, str) and value.strip() == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def to_int(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return int(value)
    except Exception:
        return float("nan")


def build_dp_config(experiment_name: str, split_type: str, alpha: float, sigma: float) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "dataset": "bloodmnist",
        "framework": "aijack",
        "algorithm": "fedavg_dp",
        "num_clients": 5,
        "num_rounds": 20,
        "local_epochs": 1,
        "batch_size": 64,
        "lr": 0.01,
        "optimizer": "SGD",
        "seed": 42,
        "split_type": split_type,
        "alpha": float(alpha),
        "device": "cpu",
        "data_dir": "./data",
        "dp_enabled": True,
        "clip_norm": 1.0,
        "noise_multiplier": float(sigma),
        "dp_noise_std": None,
        "delta": 1.0e-5,
    }


def ensure_yaml_config(config_path: Path, config_data: dict[str, Any], dry_run: bool) -> str:
    if config_path.exists():
        return "existing"
    if dry_run:
        return "planned_create"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, sort_keys=False)
    return "created"


def run_subprocess(
    cmd: list[str],
    stdout_path: Path,
    stderr_path: Path,
    dry_run: bool,
) -> tuple[bool, int, str]:
    if dry_run:
        return True, 0, "dry_run"

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    return result.returncode == 0, result.returncode, "ran"


def write_json(path: Path, payload: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def collect_experiment_metadata(
    split_label: str,
    split_type: str,
    alpha: float,
    sigma: float,
    dp_enabled: bool,
    experiment_name: str,
    experiment_dir: Path,
) -> dict[str, Any]:
    config = read_yaml(experiment_dir / "config.yaml")
    test_row = read_first_csv_row(experiment_dir / "test_metrics.csv")

    if not dp_enabled:
        epsilon = float("nan")
        delta = float("nan")
        sampling_rate = float("nan")
        clients_per_round = float("nan")
        clip_norm = float("nan")
        noise_multiplier = float("nan")
        effective_noise_multiplier = float("nan")
        actual_noise_std = float("nan")
    else:
        epsilon = to_float(test_row.get("epsilon", config.get("epsilon")))
        delta = to_float(test_row.get("delta", config.get("delta")))
        sampling_rate = to_float(test_row.get("sampling_rate", config.get("sampling_rate")))
        clients_per_round = to_float(
            test_row.get("clients_per_round", config.get("clients_per_round"))
        )
        clip_norm = to_float(test_row.get("clip_norm", config.get("clip_norm")))
        noise_multiplier = to_float(
            test_row.get("noise_multiplier", config.get("noise_multiplier"))
        )
        effective_noise_multiplier = to_float(
            test_row.get(
                "effective_noise_multiplier",
                config.get("effective_noise_multiplier"),
            )
        )
        actual_noise_std = to_float(
            test_row.get("actual_noise_std", config.get("actual_noise_std"))
        )

    return {
        "split_label": split_label,
        "split_type": split_type,
        "alpha": float(alpha),
        "sigma": float(sigma),
        "dp_enabled": bool(dp_enabled),
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir.resolve()),
        "epsilon": epsilon,
        "delta": delta,
        "sampling_rate": sampling_rate,
        "clients_per_round": clients_per_round,
        "clip_norm": clip_norm,
        "noise_multiplier": noise_multiplier,
        "effective_noise_multiplier": effective_noise_multiplier,
        "actual_noise_std": actual_noise_std,
        "test_accuracy": to_float(test_row.get("test_accuracy")),
        "test_macro_f1": to_float(test_row.get("test_macro_f1")),
        "test_loss": to_float(test_row.get("test_loss")),
        "best_round": to_int(test_row.get("best_round")),
        "best_val_loss": to_float(test_row.get("best_val_loss")),
    }


def build_attack_output_dir(
    experiment_dir: Path,
    attack_batch_size: int,
    distance: str,
    attack_iters: int,
    num_trials: int,
    attack_lr: float,
    client_id: int,
    sample_index: int,
) -> Path:
    run_name = (
        f"full_matrix_batch{attack_batch_size}_{distance}_{attack_iters}iters_"
        f"{num_trials}trials_lr{float_tag(attack_lr)}_client{client_id}_sample{sample_index}"
    )
    return experiment_dir / "attacks" / run_name


def aggregate_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["reconstruction_mse_num"] = pd.to_numeric(out["reconstruction_mse"], errors="coerce")
    out["test_accuracy_num"] = pd.to_numeric(out["test_accuracy"], errors="coerce")
    out["test_macro_f1_num"] = pd.to_numeric(out["test_macro_f1"], errors="coerce")
    out["epsilon_num"] = pd.to_numeric(out["epsilon"], errors="coerce")

    grouped = (
        out.groupby(group_cols, dropna=False)
        .agg(
            reconstruction_mse_mean=("reconstruction_mse_num", "mean"),
            reconstruction_mse_median=("reconstruction_mse_num", "median"),
            reconstruction_mse_std=("reconstruction_mse_num", "std"),
            test_accuracy_mean=("test_accuracy_num", "mean"),
            test_macro_f1_mean=("test_macro_f1_num", "mean"),
            epsilon_mean=("epsilon_num", "mean"),
            number_of_attacks=("attack_status", "size"),
        )
        .reset_index()
    )
    return grouped


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    headers = [
        "split_label",
        "sigma",
        "epsilon_mean",
        "test_accuracy_mean",
        "test_macro_f1_mean",
        "reconstruction_mse_mean",
        "reconstruction_mse_median",
        "number_of_attacks",
    ]
    existing = [h for h in headers if h in df.columns]
    view = df[existing].copy()
    for col in view.columns:
        if col in {"split_label", "number_of_attacks"}:
            continue
        if col == "sigma":
            view[col] = view[col].map(lambda x: f"{x:g}" if pd.notna(x) else "NaN")
        else:
            view[col] = view[col].map(
                lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float, np.floating)) else "NaN"
            )

    lines = []
    lines.append("| " + " | ".join(view.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(view.columns)) + " |")
    for _, row in view.iterrows():
        values = [str(row[c]) for c in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")
    if not ATTACK_SCRIPT.exists():
        raise FileNotFoundError(f"Attack script not found: {ATTACK_SCRIPT}")

    invalid_splits = [split for split in args.splits if split not in SPLIT_SPECS]
    if invalid_splits:
        raise ValueError(f"Unsupported split labels: {invalid_splits}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_baselines = not args.no_baseline_attacks

    experiments: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    training_failures = 0

    for split_label in args.splits:
        spec = SPLIT_SPECS[split_label]
        split_type = spec["split_type"]
        alpha = float(spec["alpha"])
        baseline_dir = spec["base_no_dp_dir"]
        prefix = spec["experiment_prefix"]

        experiments.append(
            {
                "split_label": split_label,
                "split_type": split_type,
                "alpha": alpha,
                "sigma": 0.0,
                "dp_enabled": False,
                "experiment_name": baseline_dir.name,
                "experiment_dir": baseline_dir,
                "is_baseline": True,
                "config_path": None,
                "config_status": "baseline_existing",
                "train_status": "baseline_not_trained",
            }
        )

        for sigma in args.sigmas:
            sigma = float(sigma)
            exp_name = f"{prefix}_dp_sigma_{sigma_tag(sigma)}"
            config_path = Path("configs") / f"{exp_name}.yaml"
            config_data = build_dp_config(
                experiment_name=exp_name,
                split_type=split_type,
                alpha=alpha,
                sigma=sigma,
            )

            config_status = ensure_yaml_config(config_path, config_data, args.dry_run)
            manifest_rows.append(
                {
                    "timestamp": now_iso(),
                    "stage": "config",
                    "split_label": split_label,
                    "sigma": sigma,
                    "experiment_name": exp_name,
                    "experiment_dir": str((Path("results") / exp_name).resolve()),
                    "status": config_status,
                    "returncode": 0,
                    "command": "",
                }
            )

            experiment = {
                "split_label": split_label,
                "split_type": split_type,
                "alpha": alpha,
                "sigma": sigma,
                "dp_enabled": True,
                "experiment_name": exp_name,
                "experiment_dir": Path("results") / exp_name,
                "is_baseline": False,
                "config_path": config_path,
                "config_status": config_status,
                "train_status": "not_run",
            }

            metrics_path = experiment["experiment_dir"] / "test_metrics.csv"
            train_stdout = experiment["experiment_dir"] / "train_stdout.txt"
            train_stderr = experiment["experiment_dir"] / "train_stderr.txt"
            train_failed = experiment["experiment_dir"] / "train_failed.json"

            if args.skip_training:
                experiment["train_status"] = "skipped_by_flag"
                manifest_rows.append(
                    {
                        "timestamp": now_iso(),
                        "stage": "training",
                        "split_label": split_label,
                        "sigma": sigma,
                        "experiment_name": exp_name,
                        "experiment_dir": str(experiment["experiment_dir"].resolve()),
                        "status": "skipped_by_flag",
                        "returncode": 0,
                        "command": "",
                    }
                )
            elif metrics_path.exists() and not args.rerun_training:
                experiment["train_status"] = "skipped_existing_metrics"
                manifest_rows.append(
                    {
                        "timestamp": now_iso(),
                        "stage": "training",
                        "split_label": split_label,
                        "sigma": sigma,
                        "experiment_name": exp_name,
                        "experiment_dir": str(experiment["experiment_dir"].resolve()),
                        "status": "skipped_existing_metrics",
                        "returncode": 0,
                        "command": "",
                    }
                )
            else:
                cmd = [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--config",
                    str(config_path),
                ]
                ok, returncode, run_mode = run_subprocess(
                    cmd=cmd,
                    stdout_path=train_stdout,
                    stderr_path=train_stderr,
                    dry_run=args.dry_run,
                )
                if ok:
                    experiment["train_status"] = "ok" if run_mode == "ran" else "dry_run_planned"
                else:
                    experiment["train_status"] = "failed"
                    training_failures += 1
                    write_json(
                        train_failed,
                        {
                            "timestamp": now_iso(),
                            "command": cmd,
                            "returncode": returncode,
                            "experiment_name": exp_name,
                            "config_path": str(config_path.resolve()),
                        },
                        dry_run=args.dry_run,
                    )

                manifest_rows.append(
                    {
                        "timestamp": now_iso(),
                        "stage": "training",
                        "split_label": split_label,
                        "sigma": sigma,
                        "experiment_name": exp_name,
                        "experiment_dir": str(experiment["experiment_dir"].resolve()),
                        "status": experiment["train_status"],
                        "returncode": returncode,
                        "command": " ".join(cmd),
                        "stdout_path": str(train_stdout.resolve()),
                        "stderr_path": str(train_stderr.resolve()),
                    }
                )

            experiments.append(experiment)

    summary_rows: list[dict[str, Any]] = []
    failed_attacks = 0

    for experiment in experiments:
        split_label = experiment["split_label"]
        split_type = experiment["split_type"]
        alpha = float(experiment["alpha"])
        sigma = float(experiment["sigma"])
        dp_enabled = bool(experiment["dp_enabled"])
        experiment_name = experiment["experiment_name"]
        experiment_dir = Path(experiment["experiment_dir"])
        is_baseline = bool(experiment["is_baseline"])

        if is_baseline and not include_baselines:
            continue

        if not experiment_dir.exists():
            manifest_rows.append(
                {
                    "timestamp": now_iso(),
                    "stage": "attack",
                    "split_label": split_label,
                    "sigma": sigma,
                    "experiment_name": experiment_name,
                    "experiment_dir": str(experiment_dir.resolve()),
                    "status": "skipped_missing_experiment_dir",
                    "returncode": float("nan"),
                    "command": "",
                }
            )
            continue

        meta = collect_experiment_metadata(
            split_label=split_label,
            split_type=split_type,
            alpha=alpha,
            sigma=sigma,
            dp_enabled=dp_enabled,
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
        )

        for client_id in args.client_ids:
            for sample_index in args.sample_indices:
                attack_out_dir = build_attack_output_dir(
                    experiment_dir=experiment_dir,
                    attack_batch_size=args.attack_batch_size,
                    distance=args.distance,
                    attack_iters=args.attack_iters,
                    num_trials=args.num_trials,
                    attack_lr=args.attack_lr,
                    client_id=client_id,
                    sample_index=sample_index,
                )
                attack_metrics_path = attack_out_dir / "attack_metrics.json"
                attack_failed_path = attack_out_dir / "attack_failed.json"
                stdout_path = attack_out_dir / "attack_stdout.txt"
                stderr_path = attack_out_dir / "attack_stderr.txt"

                row = {
                    **meta,
                    "client_id": int(client_id),
                    "sample_index": int(sample_index),
                    "attack_batch_size": int(args.attack_batch_size),
                    "attack_iters": int(args.attack_iters),
                    "num_trials": int(args.num_trials),
                    "attack_lr": float(args.attack_lr),
                    "distance": args.distance,
                    "attack_status": "not_run",
                    "attack_error": "",
                    "reconstruction_mse": float("nan"),
                    "number_of_reconstructions": float("nan"),
                    "original_shape": np.nan,
                    "reconstructed_shape": np.nan,
                    "attack_output_dir": str(attack_out_dir.resolve()),
                }

                should_run_attack = True
                if args.skip_attacks:
                    should_run_attack = False
                if attack_metrics_path.exists() and not args.rerun_attacks:
                    should_run_attack = False

                command_str = ""
                returncode: float | int = 0

                if should_run_attack:
                    cmd = [
                        sys.executable,
                        str(ATTACK_SCRIPT),
                        "--experiment-dir",
                        str(experiment_dir),
                        "--client-id",
                        str(client_id),
                        "--sample-index",
                        str(sample_index),
                        "--attack-batch-size",
                        str(args.attack_batch_size),
                        "--attack-iters",
                        str(args.attack_iters),
                        "--num-trials",
                        str(args.num_trials),
                        "--attack-lr",
                        str(args.attack_lr),
                        "--distance",
                        args.distance,
                        "--device",
                        args.device,
                        "--output-dir",
                        str(attack_out_dir),
                    ]
                    command_str = " ".join(cmd)
                    ok, returncode, run_mode = run_subprocess(
                        cmd=cmd,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        dry_run=args.dry_run,
                    )
                    if not ok:
                        failed_attacks += 1
                        write_json(
                            attack_failed_path,
                            {
                                "timestamp": now_iso(),
                                "command": cmd,
                                "returncode": returncode,
                                "experiment_dir": str(experiment_dir.resolve()),
                                "client_id": int(client_id),
                                "sample_index": int(sample_index),
                                "attack_batch_size": int(args.attack_batch_size),
                                "attack_iters": int(args.attack_iters),
                                "num_trials": int(args.num_trials),
                                "attack_lr": float(args.attack_lr),
                                "distance": args.distance,
                            },
                            dry_run=args.dry_run,
                        )
                        row["attack_status"] = "failed"
                        row["attack_error"] = f"Attack command failed with returncode={returncode}"
                    elif run_mode == "dry_run":
                        row["attack_status"] = "dry_run_planned"
                    else:
                        row["attack_status"] = "ok"
                else:
                    if args.skip_attacks:
                        row["attack_status"] = "skipped_by_flag"
                    elif attack_metrics_path.exists():
                        row["attack_status"] = "skipped_existing_metrics"
                    else:
                        row["attack_status"] = "missing_metrics"

                if attack_metrics_path.exists():
                    try:
                        with attack_metrics_path.open("r", encoding="utf-8") as f:
                            attack_metrics = json.load(f)
                        row["attack_status"] = attack_metrics.get("attack_status", row["attack_status"])
                        row["attack_error"] = attack_metrics.get("attack_error") or ""
                        row["reconstruction_mse"] = to_float(attack_metrics.get("reconstruction_mse"))
                        row["number_of_reconstructions"] = to_int(
                            attack_metrics.get("number_of_reconstructions")
                        )
                        row["original_shape"] = json.dumps(attack_metrics.get("original_shape"))
                        row["reconstructed_shape"] = json.dumps(
                            attack_metrics.get("reconstructed_shape")
                        )
                        row["attack_output_dir"] = attack_metrics.get(
                            "output_dir", row["attack_output_dir"]
                        )
                    except Exception as error:
                        row["attack_status"] = "metrics_parse_failed"
                        row["attack_error"] = f"{type(error).__name__}: {error}"
                        failed_attacks += 1
                elif attack_failed_path.exists():
                    try:
                        with attack_failed_path.open("r", encoding="utf-8") as f:
                            failed_info = json.load(f)
                        row["attack_status"] = "failed"
                        row["attack_error"] = (
                            failed_info.get("attack_error")
                            or failed_info.get("stderr")
                            or f"Attack failed with returncode={failed_info.get('returncode')}"
                        )
                    except Exception:
                        row["attack_status"] = "failed"
                        row["attack_error"] = "attack_failed.json exists but could not be parsed"

                summary_rows.append(row)
                manifest_rows.append(
                    {
                        "timestamp": now_iso(),
                        "stage": "attack",
                        "split_label": split_label,
                        "sigma": sigma,
                        "experiment_name": experiment_name,
                        "experiment_dir": str(experiment_dir.resolve()),
                        "status": row["attack_status"],
                        "returncode": returncode,
                        "command": command_str,
                        "client_id": int(client_id),
                        "sample_index": int(sample_index),
                        "attack_output_dir": row["attack_output_dir"],
                        "stdout_path": str(stdout_path.resolve()),
                        "stderr_path": str(stderr_path.resolve()),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
    else:
        for col in SUMMARY_COLUMNS:
            if col not in summary_df.columns:
                summary_df[col] = np.nan
        summary_df = summary_df[SUMMARY_COLUMNS]

    summary_path = output_dir / "full_dp_privacy_utility_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    grouped_split_sigma = aggregate_group(summary_df, ["split_label", "sigma"])
    grouped_sigma = aggregate_group(summary_df, ["sigma"])
    grouped_split = aggregate_group(summary_df, ["split_label"])

    group_split_sigma_path = output_dir / "group_by_split_sigma.csv"
    group_sigma_path = output_dir / "group_by_sigma.csv"
    group_split_path = output_dir / "group_by_split.csv"

    grouped_split_sigma.to_csv(group_split_sigma_path, index=False)
    grouped_sigma.to_csv(group_sigma_path, index=False)
    grouped_split.to_csv(group_split_path, index=False)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "run_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    failed_attacks_in_summary = int((summary_df["attack_status"] == "failed").sum()) if not summary_df.empty else 0
    failed_attacks_total = max(failed_attacks, failed_attacks_in_summary)

    total_models = len(experiments)
    total_attacks = len(summary_df)

    report_lines = [
        "# Full DP Privacy-Utility Report",
        "",
        f"- Generated at: {datetime.now().isoformat()}",
        f"- Total models: {total_models}",
        f"- Total attacks attempted: {total_attacks}",
        f"- Number of failed trainings: {training_failures}",
        f"- Number of failed attacks: {failed_attacks_total}",
        "",
        "## Split/Sigma Summary",
        "",
        markdown_table(grouped_split_sigma),
        "",
        "This is an empirical privacy-utility evaluation. Reconstruction metrics are stochastic and should be interpreted across multiple samples and clients, not from a single attack.",
        "",
    ]
    report_path = output_dir / "full_dp_privacy_utility_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Full DP privacy-utility matrix run completed.")
    print(f"Summary CSV: {summary_path.resolve()}")
    print(f"Grouped by split/sigma: {group_split_sigma_path.resolve()}")
    print(f"Grouped by sigma: {group_sigma_path.resolve()}")
    print(f"Grouped by split: {group_split_path.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Markdown report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
