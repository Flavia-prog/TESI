from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis.experiments.config import sweep_config_from_yaml
from thesis.experiments.runner import run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attack sweep in-process")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-dirs", nargs="+", default=None)
    parser.add_argument("--client-ids", nargs="+", type=int, default=None)
    parser.add_argument("--sample-indices", nargs="+", type=int, default=None)
    parser.add_argument("--attack-batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--attack-iters", nargs="+", type=int, default=None)
    parser.add_argument("--num-trials", nargs="+", type=int, default=None)
    parser.add_argument("--attack-lrs", nargs="+", type=float, default=None)
    parser.add_argument("--distances", nargs="+", default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--sweep-name", type=str, default=None)
    parser.add_argument("--target-metric", choices=["best_mse", "best_ssim"], default="best_ssim")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "experiment_dirs": args.experiment_dirs,
        "client_ids": args.client_ids,
        "sample_indices": args.sample_indices,
        "attack_batch_sizes": args.attack_batch_sizes,
        "attack_iters": args.attack_iters,
        "num_trials": args.num_trials,
        "attack_lrs": args.attack_lrs,
        "distances": args.distances,
        "dataset": args.dataset,
        "model_name": args.model_name,
        "device": args.device,
        "jobs": args.jobs,
        "rerun_existing": args.rerun_existing,
        "max_runs": args.max_runs,
        "output_root": args.output_root,
        "sweep_name": args.sweep_name,
    }
    cfg = sweep_config_from_yaml(args.config, overrides=overrides)
    result = run_sweep(cfg, target_metric=args.target_metric)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
