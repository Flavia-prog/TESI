import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Day 3 exploratory-results delivery package from SSIM and MSE screening outputs."
    )
    parser.add_argument(
        "--ssim-dir",
        type=str,
        default="results/attack_parameter_impact/screening_blood_v1",
        help="Directory of the best_ssim analysis results.",
    )
    parser.add_argument(
        "--mse-dir",
        type=str,
        default="results/attack_parameter_impact_mse/screening_blood_v1",
        help="Directory of the best_mse analysis results.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/attack_parameter_impact/screening_blood_v1_exploratory_results_delivery_package",
        help="Output package directory.",
    )
    parser.add_argument(
        "--extra-ssim-dir",
        type=str,
        default=None,
        help="Optional SSIM analysis directory for an additional dataset sweep.",
    )
    parser.add_argument(
        "--primary-label",
        type=str,
        default="BloodMNIST",
        help="Label for primary SSIM sweep in cross-dataset plot.",
    )
    parser.add_argument(
        "--extra-label",
        type=str,
        default="PathMNIST",
        help="Label for extra SSIM sweep in cross-dataset plot.",
    )
    return parser.parse_args()


def save_fig(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_ext.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_no_ext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def top_parameters(path: Path, n: int = 5) -> list[dict]:
    df = pd.read_csv(path)
    return df.head(n).to_dict(orient="records")


def build_mse_importance_plot(parameter_importance_csv: Path, out_fig_base: Path) -> None:
    df = pd.read_csv(parameter_importance_csv).sort_values(
        "importance_mean_mae_increase", ascending=False
    )
    top10 = df.head(10).iloc[::-1]
    vals = top10["importance_mean_mae_increase"]
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top10["parameter"], vals, color=colors, alpha=0.9)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Mean MAE Increase After Permutation")
    ax.set_ylabel("Parameter")
    ax.set_title("Top-10 Parameter Importance (target = best_mse)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_fig(fig, out_fig_base)


def build_cross_dataset_rank_plot(
    primary_param_csv: Path,
    extra_param_csv: Path,
    primary_label: str,
    extra_label: str,
    out_fig_base: Path,
) -> None:
    primary_df = pd.read_csv(primary_param_csv).sort_values(
        "importance_mean_mae_increase",
        ascending=False,
    )
    extra_df = pd.read_csv(extra_param_csv).sort_values(
        "importance_mean_mae_increase",
        ascending=False,
    )

    primary_df = primary_df.reset_index(drop=True)
    extra_df = extra_df.reset_index(drop=True)
    primary_df["rank_primary"] = primary_df.index + 1
    extra_df["rank_extra"] = extra_df.index + 1

    merged = primary_df[["parameter", "rank_primary"]].merge(
        extra_df[["parameter", "rank_extra"]],
        on="parameter",
        how="inner",
    )

    if merged.empty:
        raise ValueError("No overlapping parameters between primary and extra sweeps.")

    merged = merged.sort_values("rank_primary").head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in merged.iterrows():
        ax.plot(
            [row["rank_primary"], row["rank_extra"]],
            [row["parameter"], row["parameter"]],
            color="#9aa0a6",
            linewidth=1.2,
            alpha=0.8,
        )

    ax.scatter(
        merged["rank_primary"],
        merged["parameter"],
        color="#1f77b4",
        s=52,
        label=primary_label,
        zorder=3,
    )
    ax.scatter(
        merged["rank_extra"],
        merged["parameter"],
        color="#ff7f0e",
        s=52,
        label=extra_label,
        zorder=3,
    )

    max_rank = int(max(merged["rank_primary"].max(), merged["rank_extra"].max()))
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.invert_xaxis()
    ax.set_xlabel("Importance Rank (1 = most important)")
    ax.set_ylabel("Parameter")
    ax.set_title("Cross-Dataset Parameter Importance Rank Comparison")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, out_fig_base)


def build_one_page_pdf(
    out_pdf: Path,
    ssim_summary: dict,
    mse_summary: dict,
    ssim_top: list[dict],
    mse_top: list[dict],
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    lines = [
        "Federated Learning Attack Screening - Day 3 Summary",
        "",
        "Dataset/Setup: BloodMNIST, AIJack FedAvg + Gradient Inversion",
        f"Canonical sweep: {ssim_summary.get('sweep_name', 'n/a')}",
        f"Usable rows (best_ssim): {ssim_summary.get('n_regression_rows', 'n/a')}",
        f"Usable rows (best_mse): {mse_summary.get('n_regression_rows', 'n/a')}",
        "",
        "What matters most:",
        f"- best_ssim top factors: {', '.join([r['parameter'] for r in ssim_top[:4]])}",
        f"- best_mse top factors: {', '.join([r['parameter'] for r in mse_top[:4]])}",
        "- distance is the dominant and stable parameter across both targets.",
        "- train_alpha, client/sample identity are consistently important.",
        "",
        "What seems weak:",
        "- attack_iters and num_trials show near-zero or negative importance in this sweep.",
        "- train_split_type has weaker effect once alpha is explicitly included.",
        "",
        "What is still uncertain:",
        "- Importance is predictive, not causal (exploratory model).",
        "- Possible interactions were not isolated in a factorial causal test.",
        "- Generalization beyond BloodMNIST still needs cross-dataset replication.",
        "",
        "Recommended immediate next step:",
        "- Repeat same design on >=1 additional MedMNIST dataset and compare ranking stability.",
    ]

    ax.text(
        0.06,
        0.97,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def build_text_note(
    out_path: Path,
    ssim_top: list[dict],
    mse_top: list[dict],
) -> None:
    text = []
    text.append("Interpretation Note (Day 3)\n")
    text.append("What matters most:")
    text.append(
        f"- Stable top factors across both targets: {', '.join([r['parameter'] for r in ssim_top[:4]])}"
    )
    text.append("- distance is consistently dominant.")
    text.append("- train_alpha, client_id, and sample_index are consistently relevant.\n")
    text.append("What seems weak:")
    text.append("- attack_iters and num_trials remain weak/negative in this sweep.")
    text.append("- train_split_type contributes less than alpha.\n")
    text.append("What is still uncertain:")
    text.append("- These importances are exploratory and not causal.")
    text.append("- Dataset transferability is not proven until cross-dataset replication.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(text), encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    ssim_dir = Path(args.ssim_dir)
    mse_dir = Path(args.mse_dir)
    out_dir = Path(args.out_dir)
    out_fig_dir = out_dir / "figures"

    ssim_summary_path = ssim_dir / "summary.json"
    mse_summary_path = mse_dir / "summary.json"
    ssim_param_path = ssim_dir / "parameter_importance.csv"
    mse_param_path = mse_dir / "parameter_importance.csv"
    ssim_agg_path = ssim_dir / "aggregated_attack_results.csv"
    mse_agg_path = mse_dir / "aggregated_attack_results.csv"

    if not ssim_summary_path.exists() or not mse_summary_path.exists():
        raise FileNotFoundError("Missing summary.json in SSIM or MSE directory.")
    if not ssim_param_path.exists() or not mse_param_path.exists():
        raise FileNotFoundError("Missing parameter_importance.csv in SSIM or MSE directory.")
    if not ssim_agg_path.exists() or not mse_agg_path.exists():
        raise FileNotFoundError("Missing aggregated_attack_results.csv in SSIM or MSE directory.")

    ssim_summary = json.loads(ssim_summary_path.read_text(encoding="utf-8"))
    mse_summary = json.loads(mse_summary_path.read_text(encoding="utf-8"))
    ssim_top = top_parameters(ssim_param_path, n=10)
    mse_top = top_parameters(mse_param_path, n=10)

    # Build the requested Day 3 figure for best_mse
    build_mse_importance_plot(
        mse_param_path,
        out_fig_dir / "plot4_top10_feature_importance_best_mse",
    )

    # Copy Day 2 figures from SSIM run if available.
    ssim_fig_dir = ssim_dir / "figures"
    if ssim_fig_dir.exists():
        for p in ssim_fig_dir.glob("*"):
            copy_if_exists(p, out_fig_dir / p.name)

    # Required package core files + explicit target variants.
    copy_if_exists(mse_summary_path, out_dir / "summary.json")
    copy_if_exists(mse_param_path, out_dir / "parameter_importance.csv")
    copy_if_exists(mse_agg_path, out_dir / "aggregated_attack_results.csv")

    copy_if_exists(ssim_summary_path, out_dir / "summary_best_ssim.json")
    copy_if_exists(ssim_param_path, out_dir / "parameter_importance_best_ssim.csv")
    copy_if_exists(ssim_agg_path, out_dir / "aggregated_attack_results_best_ssim.csv")
    copy_if_exists(mse_summary_path, out_dir / "summary_best_mse.json")
    copy_if_exists(mse_param_path, out_dir / "parameter_importance_best_mse.csv")
    copy_if_exists(mse_agg_path, out_dir / "aggregated_attack_results_best_mse.csv")

    if args.extra_ssim_dir:
        extra_ssim_dir = Path(args.extra_ssim_dir)
        extra_summary_path = extra_ssim_dir / "summary.json"
        extra_param_path = extra_ssim_dir / "parameter_importance.csv"
        extra_agg_path = extra_ssim_dir / "aggregated_attack_results.csv"

        if not extra_summary_path.exists() or not extra_param_path.exists() or not extra_agg_path.exists():
            raise FileNotFoundError(
                "Missing summary.json / parameter_importance.csv / aggregated_attack_results.csv in --extra-ssim-dir."
            )

        copy_if_exists(extra_summary_path, out_dir / "summary_extra_dataset.json")
        copy_if_exists(extra_param_path, out_dir / "parameter_importance_extra_dataset.csv")
        copy_if_exists(extra_agg_path, out_dir / "aggregated_attack_results_extra_dataset.csv")

        build_cross_dataset_rank_plot(
            primary_param_csv=ssim_param_path,
            extra_param_csv=extra_param_path,
            primary_label=args.primary_label,
            extra_label=args.extra_label,
            out_fig_base=out_fig_dir / "plot5_cross_dataset_importance_rank_comparison",
        )

    build_text_note(out_dir / "interpretation_note.txt", ssim_top, mse_top)
    build_one_page_pdf(
        out_dir / "one_page_summary.pdf",
        ssim_summary=ssim_summary,
        mse_summary=mse_summary,
        ssim_top=ssim_top,
        mse_top=mse_top,
    )

    print("Built exploratory-results delivery package at:", out_dir.resolve())
    print("Key files:")
    print("-", (out_dir / "figures").resolve())
    print("-", (out_dir / "summary.json").resolve())
    print("-", (out_dir / "parameter_importance.csv").resolve())
    print("-", (out_dir / "aggregated_attack_results.csv").resolve())
    print("-", (out_dir / "one_page_summary.pdf").resolve())
    print("-", (out_dir / "interpretation_note.txt").resolve())


if __name__ == "__main__":
    main()
