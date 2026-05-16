"""
plot_frontier.py
================

Produces the headline privacy-utility frontier plots for the thesis
(Section 4.4 in the experiment manifest).

Inputs
------
Reads the aggregated summary CSV written by run_full_dp_privacy_utility_matrix.py:
    results/current/privacy_utility/full_dp_privacy_utility_matrix/full_dp_privacy_utility_matrix_summary.csv

Expected columns (subset; all are produced by the orchestrator):
    split_label, split_type, alpha, sigma, dp_enabled, epsilon,
    test_accuracy, test_macro_f1, reconstruction_mse,
    attack_status, client_id, sample_index

Outputs
-------
Written to <output-dir>:
    frontier_accuracy_vs_leakage.png / .pdf
    frontier_accuracy_vs_epsilon.png  / .pdf
    accuracy_vs_sigma.png             / .pdf
    leakage_vs_sigma.png              / .pdf
    pareto_operating_points.csv

Each plot uses one color per heterogeneity level (split_label / alpha)
so the reader can see how the frontier shifts with non-IID severity.

Run
---
    python plot_frontier.py \
        --summary-csv results/current/privacy_utility/full_dp_privacy_utility_matrix/full_dp_privacy_utility_matrix_summary.csv \
        --output-dir  results/current/privacy_utility/full_dp_privacy_utility_matrix/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless safe; works inside VS Code too
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Stable color per split_label so the four plots are visually consistent.
SPLIT_COLORS = {
    "iid": "#1f77b4",              # blue
    "noniid_alpha_1": "#2ca02c",   # green
    "noniid_alpha_05": "#ff7f0e",  # orange
    "noniid_alpha_01": "#d62728",  # red
}

# Display order: from most homogeneous to most heterogeneous.
SPLIT_ORDER = ["iid", "noniid_alpha_1", "noniid_alpha_05", "noniid_alpha_01"]

SPLIT_PRETTY = {
    "iid": "IID",
    "noniid_alpha_1": r"Dirichlet $\alpha=1.0$",
    "noniid_alpha_05": r"Dirichlet $\alpha=0.5$",
    "noniid_alpha_01": r"Dirichlet $\alpha=0.1$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot privacy-utility frontier from the DP matrix summary CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/current/privacy_utility/full_dp_privacy_utility_matrix/full_dp_privacy_utility_matrix_summary.csv",
        help="Path to the aggregated summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/current/privacy_utility/full_dp_privacy_utility_matrix/figures",
        help="Directory to write figures and Pareto table.",
    )
    parser.add_argument(
        "--leakage-metric",
        type=str,
        default="reconstruction_mse",
        help=(
            "Column name to use as the leakage axis. "
            "Defaults to reconstruction_mse. "
            "When LPIPS / windowed SSIM are added, pass e.g. --leakage-metric best_lpips."
        ),
    )
    parser.add_argument(
        "--leakage-direction",
        choices=["lower_is_more_leaky", "higher_is_more_leaky"],
        default="lower_is_more_leaky",
        help=(
            "Semantics of the leakage metric. "
            "MSE is lower-is-more-leaky (close to 0 means perfect reconstruction). "
            "SSIM and LPIPS are higher-is-more-leaky."
        ),
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_summary(path: Path, leakage_metric: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Summary CSV not found at {path}. "
            "Run run_full_dp_privacy_utility_matrix.py first, "
            "or pass --summary-csv to point at the correct file."
        )

    df = pd.read_csv(path)

    required = {
        "split_label", "alpha", "sigma", "dp_enabled",
        "epsilon", "test_accuracy", "attack_status",
        leakage_metric,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in summary CSV: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )

    # Use only attacks that completed successfully; everything else is noise.
    df = df[df["attack_status"] == "ok"].copy()

    for col in ("alpha", "sigma", "epsilon", "test_accuracy", leakage_metric):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["test_accuracy", leakage_metric]).copy()

    if df.empty:
        raise ValueError(
            "After filtering by attack_status=='ok' and dropping NaNs in "
            "test_accuracy and the leakage metric, no rows remain. "
            "Check the summary CSV contents."
        )
    return df


def aggregate_per_cell(df: pd.DataFrame, leakage_metric: str) -> pd.DataFrame:
    """Average over (client_id, sample_index) within each (split_label, sigma)."""
    grouped = (
        df.groupby(["split_label", "alpha", "sigma", "dp_enabled"], dropna=False)
        .agg(
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            leakage_mean=(leakage_metric, "mean"),
            leakage_std=(leakage_metric, "std"),
            epsilon_mean=("epsilon", "mean"),
            n_attacks=("test_accuracy", "size"),
        )
        .reset_index()
    )
    return grouped


def order_splits(df: pd.DataFrame) -> list[str]:
    """Return present split labels in the canonical display order."""
    present = set(df["split_label"].unique())
    ordered = [s for s in SPLIT_ORDER if s in present]
    leftover = sorted(present - set(ordered))
    return ordered + leftover


def plot_axis_vs_sigma(
    agg: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for split in order_splits(agg):
        sub = agg[agg["split_label"] == split].sort_values("sigma")
        color = SPLIT_COLORS.get(split, "black")
        label = SPLIT_PRETTY.get(split, split)
        ax.plot(
            sub["sigma"], sub[y_col],
            marker="o", color=color, label=label, linewidth=1.8,
        )

    ax.set_xlabel(r"DP noise multiplier $\sigma$")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(title="Heterogeneity", fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_base)


def plot_frontier(
    agg: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    out_base: Path,
    invert_x: bool = False,
    annotate_sigma: bool = True,
) -> None:
    """
    x_col is the leakage axis (reconstruction_mse, or LPIPS later, or epsilon).
    invert_x = True flips the axis so that "more leaky / less private" is on the right.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for split in order_splits(agg):
        sub = agg[agg["split_label"] == split].sort_values(x_col)
        color = SPLIT_COLORS.get(split, "black")
        label = SPLIT_PRETTY.get(split, split)
        ax.plot(
            sub[x_col], sub["test_accuracy_mean"],
            marker="o", color=color, label=label, linewidth=1.8,
        )
        if annotate_sigma:
            for _, row in sub.iterrows():
                ax.annotate(
                    f"σ={row['sigma']:g}",
                    xy=(row[x_col], row["test_accuracy_mean"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=color, alpha=0.85,
                )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Test accuracy")
    ax.set_title(title)
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(title="Heterogeneity", fontsize=9)
    if invert_x:
        ax.invert_xaxis()
    fig.tight_layout()
    save_figure(fig, out_base)


def compute_pareto(
    agg: pd.DataFrame,
    leakage_direction: str,
) -> pd.DataFrame:
    """
    Per split, return the Pareto-efficient set:
    points where no other point has both higher accuracy *and* better privacy.

    Better privacy means:
        - smaller leakage_mean if direction == "higher_is_more_leaky" (e.g. SSIM/LPIPS)
        - larger leakage_mean if direction == "lower_is_more_leaky"  (e.g. MSE)
    """
    pareto_rows = []
    for split in order_splits(agg):
        sub = agg[agg["split_label"] == split].copy()
        if sub.empty:
            continue

        # Normalize so that "more private" = larger leakage_score.
        if leakage_direction == "higher_is_more_leaky":
            sub["leakage_score"] = -sub["leakage_mean"]
        else:
            sub["leakage_score"] = sub["leakage_mean"]

        points = sub[["test_accuracy_mean", "leakage_score"]].to_numpy()

        keep = np.ones(len(points), dtype=bool)
        for i, p in enumerate(points):
            for j, q in enumerate(points):
                if i == j:
                    continue
                # q dominates p if q is >= on both axes and strictly > on one.
                if (q[0] >= p[0] and q[1] >= p[1]) and (q[0] > p[0] or q[1] > p[1]):
                    keep[i] = False
                    break

        sub_pareto = sub.loc[keep].drop(columns=["leakage_score"]).copy()
        sub_pareto["is_pareto"] = True
        pareto_rows.append(sub_pareto)

    if not pareto_rows:
        return pd.DataFrame()

    out = pd.concat(pareto_rows, ignore_index=True).sort_values(
        ["split_label", "sigma"]
    )
    return out


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    output_dir = Path(args.output_dir)

    df = load_summary(summary_path, args.leakage_metric)
    print(f"Loaded {len(df)} attack rows from {summary_path}")

    # Standardize leakage_metric to a shared column name for plotting.
    df["leakage"] = df[args.leakage_metric]
    agg = aggregate_per_cell(df, "leakage")
    print(f"Aggregated to {len(agg)} (split, sigma) cells")

    # Plot 1: utility vs sigma
    plot_axis_vs_sigma(
        agg,
        y_col="test_accuracy_mean",
        y_label="Test accuracy",
        title="Utility vs DP noise multiplier",
        out_base=output_dir / "accuracy_vs_sigma",
    )

    # Plot 2: leakage vs sigma
    plot_axis_vs_sigma(
        agg,
        y_col="leakage_mean",
        y_label=f"Leakage ({args.leakage_metric})",
        title="Leakage vs DP noise multiplier",
        out_base=output_dir / "leakage_vs_sigma",
    )

    # Plot 3: HEADLINE frontier — accuracy vs leakage
    # For MSE: higher x = less leaky = more private (we don't invert).
    # For SSIM/LPIPS later: higher x = more leaky = less private (we will invert).
    invert_x = args.leakage_direction == "higher_is_more_leaky"
    plot_frontier(
        agg,
        x_col="leakage_mean",
        x_label=f"Leakage ({args.leakage_metric})  →  less leaky to the {'left' if invert_x else 'right'}",
        title="Privacy-utility frontier (per heterogeneity level)",
        out_base=output_dir / "frontier_accuracy_vs_leakage",
        invert_x=invert_x,
    )

    # Plot 4: frontier with epsilon on x-axis (DP-budget view).
    # Higher epsilon = less private, so we put epsilon on x with no inversion.
    # Skip baseline (no DP) rows which have NaN epsilon.
    agg_dp = agg.dropna(subset=["epsilon_mean"]).copy()
    if not agg_dp.empty:
        plot_frontier(
            agg_dp,
            x_col="epsilon_mean",
            x_label=r"DP budget $\varepsilon$ (lower is more private)",
            title=r"Privacy-utility frontier in $(\varepsilon, \mathrm{accuracy})$ space",
            out_base=output_dir / "frontier_accuracy_vs_epsilon",
            invert_x=True,  # show "more private" on the right, matching plot 3 convention
        )
    else:
        print("No DP rows with finite epsilon — skipping epsilon-axis frontier plot.")

    # Pareto table
    pareto = compute_pareto(agg, leakage_direction=args.leakage_direction)
    pareto_path = output_dir / "pareto_operating_points.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pareto.to_csv(pareto_path, index=False)
    print(f"Wrote Pareto-efficient operating points to {pareto_path}")

    # Brief stdout summary so you can sanity-check without opening the PNGs.
    print("\nPer-split summary (mean across attacks per (split, sigma)):")
    pretty = agg.sort_values(["split_label", "sigma"])[
        ["split_label", "sigma", "epsilon_mean",
         "test_accuracy_mean", "leakage_mean", "n_attacks"]
    ]
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_rows", None):
        print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
