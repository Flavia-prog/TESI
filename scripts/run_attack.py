from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis.attacks import run_attack
from thesis.experiments.config import attack_config_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one gradient inversion attack in-process")
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--client-id", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--attack-batch-size", type=int, default=None)
    parser.add_argument("--attack-iters", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--attack-lr", type=float, default=None)
    parser.add_argument("--distance", choices=["l2", "cossim"], default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "client_id": args.client_id,
        "sample_index": args.sample_index,
        "attack_batch_size": args.attack_batch_size,
        "attack_iters": args.attack_iters,
        "num_trials": args.num_trials,
        "attack_lr": args.attack_lr,
        "distance": args.distance,
        "device": args.device,
        "dataset": args.dataset,
        "model_name": args.model_name,
    }
    cfg = attack_config_from_yaml(args.config, overrides=overrides)
    result = run_attack(args.experiment_dir, cfg, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
