"""
aggregate_dp_matrix.py
======================

Walk the results/ tree and build the aggregated summary CSV expected by
plot_frontier.py. This rebuilds what run_full_dp_privacy_utility_matrix.py
should have written at the end of its run.

For every experiment directory matching iid* or noniid_alpha_* :
    - parse split_label, alpha, dp_enabled, sigma from the directory name
    - read test_metrics.csv for utility + epsilon
    - walk attacks/full_matrix_*/attack_metrics.json for leakage
    - emit one row per (split, sigma, client_id, sample_index)

Output: results/full_dp_privacy_utility_matrix/full_dp_privacy_utility_matrix_summary.csv

Run
---
    python aggregate_dp_matrix.py
    # or with a different attack-run prefix:
    python aggregate_dp_matrix.py --attack-prefix full_matrix_
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# Directory-name parsers ------------------------------------------------------

# Matches: iid_baseline  iid  iid_dp_sigma_05  iid_dp_sigma_1  iid_dp_sigma_025  iid_dp_sigma_2
IID_RE = re.compile(r"^iid(?:_baseline)?(?:_dp_sigma_(?P<sigma>\d+))?$")

# Matches: noniid_alpha_01  noniid_alpha_05  noniid_alpha_1
#          noniid_alpha_01_dp_sigma_025  noniid_alpha_05_dp_sigma_1  etc.
NONIID_RE = re.compile(
    r"^noniid_alpha_(?P<alpha>\d+)(?:_dp_sigma_(?P<sigma>\d+))?$"
)

# Maps the directory-suffix encoding to the real numerical value.
# Your directories use:  alpha_01 -> 0.1,  alpha_05 -> 0.5,  alpha_1 -> 1.0
# and:  sigma_025 -> 0.25,  sigma_05 -> 0.5,  sigma_075 -> 0.75,  sigma_1 -> 1.0,  sigma_2 -> 2.0
ALPHA_MAP = {"01": 0.1, "05": 0.5, "1": 1.0}
SIGMA_MAP = {"025": 0.25, "05": 0.5, "075": 0.75, "1": 1.0, "2": 2.0}


def parse_experiment_dir(name: str) -> dict | None:
    """Return split metadata for a directory name, or None if it doesn't match."""
    m = IID_RE.match(name)
    if m:
        sigma_key = m.group("sigma")
        return {
            "split_label": "iid_baseline" if sigma_key is None else f"iid_dp_sigma_{sigma_key}",
            "split_type": "iid",
            "alpha": None,
            "dp_enabled": sigma_key is not None,
            "sigma": SIGMA_MAP.get(sigma_key, 0.0) if sigma_key else 0.0,
        }

    m = NONIID_RE.match(name)
    if m:
        alpha_key = m.group("alpha")
        sigma_key = m.group("sigma")
        alpha_val = ALPHA_MAP.get(alpha_key)
        if alpha_val is None:
            return None
        if sigma_key is None:
            split_label = f"noniid_alpha_{alpha_key}"
        else:
            split_label = f"noniid_alpha_{alpha_key}_dp_sigma_{sigma_key}"
        # The "headline split" — the label used to color the plots — is the
        # heterogeneity, not the heterogeneity+sigma combination.
        # We expose both: plot_label (for grouping in the plot) and split_label
        # (the raw directory). plot_frontier.py uses split_label by default,
        # so we collapse the DP cells onto their non-DP parent label.
        plot_label = f"noniid_alpha_{alpha_key}"
        return {
            "split_label": plot_label,
            "split_type": "noniid",
            "alpha": alpha_val,
            "dp_enabled": sigma_key is not None,
            "sigma": SIGMA_MAP.get(sigma_key, 0.0) if sigma_key else 0.0,
            "_raw_dirname": split_label,
        }

    return None


def parse_attack_run_name(name: str, attack_prefix: str) -> dict | None:
    """
    Parse e.g. 'full_matrix_batch1_cossim_1000iters_5trials_lr01_client0_sample25'
    and return {'client_id': 0, 'sample_index': 25}.
    """
    if not name.startswith(attack_prefix):
        return None
    m_client = re.search(r"client(\d+)", name)
    m_sample = re.search(r"sample(\d+)", name)
    if not (m_client and m_sample):
        return None
    return {
        "client_id": int(m_client.group(1)),
        "sample_index": int(m_sample.group(1)),
    }


# Readers --------------------------------------------------------------------

def read_utility(test_metrics_csv: Path) -> dict:
    """Read the single-row test_metrics.csv produced by the training script."""
    df = pd.read_csv(test_metrics_csv)
    if len(df) == 0:
        return {}
    # Most recent / only row.
    row = df.iloc[-1].to_dict()
    out = {
        "test_accuracy": row.get("test_accuracy"),
        "test_macro_f1": row.get("test_macro_f1"),
        "test_loss": row.get("test_loss"),
        "epsilon": row.get("epsilon"),
        "noise_multiplier": row.get("noise_multiplier"),
        "clip_norm": row.get("clip_norm"),
        "dp_enabled_from_csv": row.get("dp_enabled"),
        "seed": row.get("seed"),
    }
    return out


def read_attack(metrics_json: Path) -> dict:
    """Read a single attack_metrics.json file."""
    with open(metrics_json) as f:
        data = json.load(f)
    return {
        "attack_status": data.get("attack_status"),
        "reconstruction_mse": data.get("reconstruction_mse"),
        "attack_iters": data.get("attack_iters"),
        "num_trials": data.get("num_trials"),
        "attack_lr": data.get("attack_lr"),
        "distance": data.get("distance"),
        "attack_batch_size": data.get("attack_batch_size"),
    }


# Main pipeline ---------------------------------------------------------------

def collect_rows(
    results_root: Path,
    attack_prefix: str,
) -> tuple[list[dict], list[dict]]:
    """
    Walk results/ and return (rows, skipped) where rows are dicts ready for
    a DataFrame and skipped is a list of directories we did not understand.
    """
    rows: list[dict] = []
    skipped: list[dict] = []

    for exp_dir in sorted(results_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        meta = parse_experiment_dir(exp_dir.name)
        if meta is None:
            # Not an experiment directory we recognize. That's fine - skip silently
            # for things like attack_parameter_impact/, old_pilot_attack_runs/, etc.
            continue

        test_csv = exp_dir / "test_metrics.csv"
        if not test_csv.exists():
            skipped.append({"dir": exp_dir.name, "reason": "no test_metrics.csv"})
            continue

        utility = read_utility(test_csv)

        attack_root = exp_dir / "attacks"
        if not attack_root.exists():
            # Training cell with no attacks; we still emit one row with NaN leakage.
            rows.append({
                **meta,
                **utility,
                "client_id": None,
                "sample_index": None,
                "attack_status": "no_attack",
                "reconstruction_mse": None,
            })
            continue

        # Find attacks matching the canonical prefix.
        found_any = False
        for attack_dir in sorted(attack_root.iterdir()):
            if not attack_dir.is_dir():
                continue
            parsed = parse_attack_run_name(attack_dir.name, attack_prefix)
            if parsed is None:
                continue
            metrics_path = attack_dir / "attack_metrics.json"
            if not metrics_path.exists():
                skipped.append({
                    "dir": str(attack_dir.relative_to(results_root)),
                    "reason": "no attack_metrics.json",
                })
                continue

            attack = read_attack(metrics_path)
            rows.append({
                **meta,
                **utility,
                **parsed,
                **attack,
            })
            found_any = True

        if not found_any:
            # Experiment exists with attacks/ but none matched the prefix.
            rows.append({
                **meta,
                **utility,
                "client_id": None,
                "sample_index": None,
                "attack_status": "no_matching_attack",
                "reconstruction_mse": None,
            })

    return rows, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-experiment utility + per-attack leakage into a single "
            "summary CSV for plot_frontier.py."
        )
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory holding the experiment subdirectories.",
    )
    parser.add_argument(
        "--attack-prefix",
        type=str,
        default="full_matrix_",
        help=(
            "Only attack-run subdirectories starting with this prefix are "
            "included. The orchestrator used 'full_matrix_' for the DP matrix."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/full_dp_privacy_utility_matrix/full_dp_privacy_utility_matrix_summary.csv",
        help="Where to write the aggregated CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    out_path = Path(args.output_csv)

    if not results_root.exists():
        raise FileNotFoundError(f"results-root not found: {results_root}")

    rows, skipped = collect_rows(results_root, args.attack_prefix)

    if not rows:
        print("No rows collected. Check --results-root and --attack-prefix.")
        return

    df = pd.DataFrame(rows)

    # Order columns sensibly.
    preferred_order = [
        "split_label", "split_type", "alpha", "sigma", "dp_enabled",
        "client_id", "sample_index",
        "test_accuracy", "test_macro_f1", "test_loss",
        "reconstruction_mse",
        "epsilon", "noise_multiplier", "clip_norm",
        "attack_status", "attack_iters", "num_trials", "attack_lr",
        "distance", "attack_batch_size",
        "seed",
    ]
    columns = [c for c in preferred_order if c in df.columns]
    columns += [c for c in df.columns if c not in columns]
    df = df[columns].sort_values(
        ["split_type", "alpha", "sigma", "client_id", "sample_index"],
        na_position="last",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
    print()
    print("Cell counts (split_label x sigma x attack_status):")
    counts = (
        df.groupby(["split_label", "sigma", "attack_status"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    print(counts.to_string())
    print()
    print("Utility + leakage summary per (split, sigma):")
    cells = (
        df[df["attack_status"] == "ok"]
        .groupby(["split_label", "sigma"], dropna=False)
        .agg(
            n_attacks=("reconstruction_mse", "size"),
            test_accuracy=("test_accuracy", "mean"),
            epsilon=("epsilon", "mean"),
            mse_mean=("reconstruction_mse", "mean"),
            mse_std=("reconstruction_mse", "std"),
        )
    )
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_rows", None,
                           "display.width", 140):
        print(cells.to_string())

    if skipped:
        print()
        print(f"Skipped {len(skipped)} item(s):")
        for s in skipped[:20]:
            print(f"  - {s}")


if __name__ == "__main__":
    main()