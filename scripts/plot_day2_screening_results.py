import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Day 2 thesis plots from attack parameter impact outputs "
            "(feature importance, attack quality factors, heterogeneity effect)."
        )
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default="results/current/analysis/attack_parameter_impact/bloodmnist/screening_blood_v1",
        help="Directory containing parameter_importance.csv and aggregated_attack_results.csv",
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_plot_1_importance(df: pd.DataFrame, out_dir: Path) -> None:
    top10 = df.head(10).copy()
    top10 = top10.iloc[::-1]  # Highest at top in horizontal bar chart

    vals = top10["importance_mean_mae_increase"]
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top10["parameter"], vals, color=colors, alpha=0.9)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Mean MAE Increase After Permutation")
    ax.set_ylabel("Parameter")
    ax.set_title("Top-10 Parameter Importance (target = best_ssim)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_dir / "plot1_top10_feature_importance")


def build_plot_2_quality_by_distance_and_iters(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.copy()
    plot_df = plot_df[plot_df["attack_status"] == "ok"]
    plot_df = plot_df.dropna(subset=["best_ssim"])
    plot_df["distance_iters"] = (
        plot_df["distance"].astype(str) + "_iters" + plot_df["attack_iters"].astype(int).astype(str)
    )

    categories = []
    for dist in ["cossim", "l2"]:
        for it in sorted(plot_df["attack_iters"].dropna().unique().tolist()):
            categories.append(f"{dist}_iters{int(it)}")
    categories = [c for c in categories if c in set(plot_df["distance_iters"])]

    grouped = [plot_df.loc[plot_df["distance_iters"] == c, "best_ssim"].values for c in categories]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(grouped, tick_labels=categories, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#7aa6c2")
        patch.set_alpha(0.8)
    ax.set_xlabel("Distance and Attack Iterations")
    ax.set_ylabel("best_ssim")
    ax.set_title("Attack Quality by Distance and Attack Iterations")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_dir / "plot2_best_ssim_by_distance_and_attack_iters")


def build_plot_3_heterogeneity_effect(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.copy()
    plot_df = plot_df[plot_df["attack_status"] == "ok"]
    plot_df = plot_df.dropna(subset=["best_ssim"])

    def split_alpha_label(row) -> str:
        split_type = str(row["train_split_type"])
        alpha = row["train_alpha"]
        if split_type == "iid":
            return "iid"
        return f"dirichlet_alpha_{alpha:g}"

    plot_df["split_alpha"] = plot_df.apply(split_alpha_label, axis=1)

    order = ["iid", "dirichlet_alpha_1", "dirichlet_alpha_0.5", "dirichlet_alpha_0.1"]
    categories = [c for c in order if c in set(plot_df["split_alpha"])]
    grouped = [plot_df.loc[plot_df["split_alpha"] == c, "best_ssim"].values for c in categories]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(grouped, tick_labels=categories, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ec27d")
        patch.set_alpha(0.85)
    ax.set_xlabel("Training Data Distribution")
    ax.set_ylabel("best_ssim")
    ax.set_title("Data Heterogeneity Effect on Attack Quality")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, out_dir / "plot3_best_ssim_by_split_type_alpha")


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    importance_path = sweep_dir / "parameter_importance.csv"
    aggregated_path = sweep_dir / "aggregated_attack_results.csv"
    figures_dir = sweep_dir / "figures"

    if not importance_path.exists():
        raise FileNotFoundError(f"Missing file: {importance_path}")
    if not aggregated_path.exists():
        raise FileNotFoundError(f"Missing file: {aggregated_path}")

    importance_df = pd.read_csv(importance_path).sort_values(
        "importance_mean_mae_increase", ascending=False
    )
    aggregated_df = pd.read_csv(aggregated_path)

    build_plot_1_importance(importance_df, figures_dir)
    build_plot_2_quality_by_distance_and_iters(aggregated_df, figures_dir)
    build_plot_3_heterogeneity_effect(aggregated_df, figures_dir)

    print("Saved figures to:", figures_dir.resolve())
    for stem in [
        "plot1_top10_feature_importance",
        "plot2_best_ssim_by_distance_and_attack_iters",
        "plot3_best_ssim_by_split_type_alpha",
    ]:
        print("-", (figures_dir / f"{stem}.png").resolve())
        print("-", (figures_dir / f"{stem}.pdf").resolve())


if __name__ == "__main__":
    main()
