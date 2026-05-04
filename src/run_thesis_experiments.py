from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline_a_bloodmnist import run_image_plan
from src.pipeline_b_text import run_text_plan


def _add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sigma_x_batch"] = out["Noise_Sigma"] * out["Batch_Size"]
    out["sigma_x_clip"] = out["Noise_Sigma"] * out["Clip_C"]
    out["clip_x_batch"] = out["Clip_C"] * out["Batch_Size"]
    out["is_text"] = (out["Modality"] == "text").astype(int)
    out["is_text_x_sigma"] = out["is_text"] * out["Noise_Sigma"]
    out["is_text_x_clip"] = out["is_text"] * out["Clip_C"]
    out["is_text_x_batch"] = out["is_text"] * out["Batch_Size"]
    return out


def _ols_coefficients(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    x = df[x_cols].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    colnames = ["intercept"] + x_cols
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return pd.DataFrame({"term": colnames, "coefficient": beta})


def run_all(plan_csv: str = "experiment_plan.csv", out_dir: str = "results/tradeoff"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    plan_df = pd.read_csv(plan_csv)
    image_df = run_image_plan(plan_df, output_csv=str(out_path / "image_results.csv"))
    text_df = run_text_plan(plan_df, output_csv=str(out_path / "text_results.csv"))

    master = pd.concat([image_df, text_df], ignore_index=True)
    master = _add_interaction_terms(master)
    master.to_csv(out_path / "tradeoff_master.csv", index=False)

    feature_cols = [
        "Batch_Size",
        "Noise_Sigma",
        "Clip_C",
        "Learning_Rate",
        "Epsilon",
        "is_text",
        "sigma_x_batch",
        "sigma_x_clip",
        "clip_x_batch",
        "is_text_x_sigma",
        "is_text_x_clip",
        "is_text_x_batch",
    ]
    acc_coef = _ols_coefficients(master, "Test_Accuracy", feature_cols)
    rec_coef = _ols_coefficients(master, "Reconstruction_Score", feature_cols)
    acc_coef.to_csv(out_path / "mlr_coefficients_accuracy.csv", index=False)
    rec_coef.to_csv(out_path / "mlr_coefficients_reconstruction.csv", index=False)

    return master


if __name__ == "__main__":
    run_all()
