from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis.experiments.config import train_config_from_yaml
from thesis.federated import train_fedavg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FedAvg via thesis package")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--split-type", choices=["iid", "dirichlet"], default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-root", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "experiment_name": args.experiment_name,
        "dataset": args.dataset,
        "model_name": args.model_name,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "seed": args.seed,
        "device": args.device,
        "split_type": args.split_type,
        "data_dir": args.data_dir,
        "results_root": args.results_root,
    }
    cfg = train_config_from_yaml(args.config, overrides=overrides)
    result = train_fedavg(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
