# Experiments Run So Far

Generated: 2026-05-16 20:53:08

## 1) Training Runs (one row per `results/<experiment>` with `test_metrics.csv`)

| Experiment | Dataset | Split | DP | Train status | Test acc | Test macro-F1 | Attack runs ok | Attack runs failed |
|---|---|---|---|---|---:|---:|---:|---:|
| dermamnist_iid_baseline | dermamnist | iid | no | ok | 0.6688 | 0.1145 | 32 | 0 |
| dermamnist_noniid_alpha_01 | dermamnist | dirichlet(alpha=0.1) | no | ok | 0.6688 | 0.1145 | 32 | 0 |
| iid_baseline | bloodmnist | iid | no | ok | 0.6659 | 0.5787 | 157 | 0 |
| iid_dp_sigma_025 | bloodmnist | iid | yes(sigma=0.25, eps=330.13) | ok | 0.5533 | 0.3881 | 6 | 0 |
| iid_dp_sigma_05 | bloodmnist | iid | yes(sigma=0.5, eps=90.13) | ok | 0.5004 | 0.3909 | 6 | 0 |
| iid_dp_sigma_075 | bloodmnist | iid | yes(sigma=0.75, eps=45.68) | ok | 0.4367 | 0.3437 | 6 | 0 |
| iid_dp_sigma_1 | bloodmnist | iid | yes(sigma=1.0, eps=30.13) | ok | 0.2982 | 0.2193 | 6 | 0 |
| iid_dp_sigma_2 | bloodmnist | iid | yes(sigma=2.0, eps=12.30) | ok | 0.1941 | 0.0413 | 6 | 0 |
| noniid_alpha_01 | bloodmnist | dirichlet(alpha=0.1) | no | ok | 0.4826 | 0.3736 | 145 | 4 |
| noniid_alpha_01_dp_sigma_025 | bloodmnist | dirichlet(alpha=0.1) | yes(sigma=0.25, eps=330.13) | failed_latest | 0.3935 | 0.2634 | 0 | 5 |
| noniid_alpha_01_dp_sigma_05 | bloodmnist | dirichlet(alpha=0.1) | yes(sigma=0.5, eps=90.13) | failed_latest | 0.3403 | 0.2466 | 0 | 5 |
| noniid_alpha_01_dp_sigma_075 | bloodmnist | dirichlet(alpha=0.1) | yes(sigma=0.75, eps=45.68) | failed_latest | 0.2736 | 0.2017 | 0 | 5 |
| noniid_alpha_01_dp_sigma_1 | bloodmnist | dirichlet(alpha=0.1) | yes(sigma=1.0, eps=30.13) | failed_latest | 0.1508 | 0.0393 | 0 | 5 |
| noniid_alpha_01_dp_sigma_2 | bloodmnist | dirichlet(alpha=0.1) | yes(sigma=2.0, eps=12.30) | failed_latest | 0.2172 | 0.0846 | 0 | 5 |
| noniid_alpha_05 | bloodmnist | dirichlet(alpha=0.5) | no | ok | 0.6352 | 0.5509 | 149 | 0 |
| noniid_alpha_05_dp_sigma_025 | bloodmnist | dirichlet(alpha=0.5) | yes(sigma=0.25, eps=330.13) | failed_latest | 0.4414 | 0.3221 | 5 | 5 |
| noniid_alpha_05_dp_sigma_05 | bloodmnist | dirichlet(alpha=0.5) | yes(sigma=0.5, eps=90.13) | failed_latest | 0.3970 | 0.2779 | 5 | 5 |
| noniid_alpha_05_dp_sigma_075 | bloodmnist | dirichlet(alpha=0.5) | yes(sigma=0.75, eps=45.68) | failed_latest | 0.3853 | 0.2701 | 5 | 5 |
| noniid_alpha_05_dp_sigma_1 | bloodmnist | dirichlet(alpha=0.5) | yes(sigma=1.0, eps=30.13) | failed_latest | 0.2973 | 0.1962 | 5 | 5 |
| noniid_alpha_05_dp_sigma_2 | bloodmnist | dirichlet(alpha=0.5) | yes(sigma=2.0, eps=12.30) | failed_latest | 0.1649 | 0.0480 | 4 | 5 |
| noniid_alpha_1 | bloodmnist | dirichlet(alpha=1.0) | no | ok | 0.6343 | 0.5260 | 150 | 0 |
| noniid_alpha_1_dp_sigma_025 | bloodmnist | dirichlet(alpha=1.0) | yes(sigma=0.25, eps=330.13) | failed_latest | 0.5282 | 0.3745 | 5 | 5 |
| noniid_alpha_1_dp_sigma_05 | bloodmnist | dirichlet(alpha=1.0) | yes(sigma=0.5, eps=90.13) | failed_latest | 0.4294 | 0.3022 | 5 | 5 |
| noniid_alpha_1_dp_sigma_075 | bloodmnist | dirichlet(alpha=1.0) | yes(sigma=0.75, eps=45.68) | failed_latest | 0.3362 | 0.2016 | 5 | 5 |
| noniid_alpha_1_dp_sigma_1 | bloodmnist | dirichlet(alpha=1.0) | yes(sigma=1.0, eps=30.13) | failed_latest | 0.1947 | 0.0407 | 5 | 5 |
| noniid_alpha_1_dp_sigma_2 | bloodmnist | dirichlet(alpha=1.0) | yes(sigma=2.0, eps=12.30) | failed_latest | 0.1950 | 0.0422 | 5 | 5 |
| pathmnist_iid_baseline | pathmnist | iid | no | ok | 0.7138 | 0.6591 | 32 | 0 |
| pathmnist_noniid_alpha_01 | pathmnist | dirichlet(alpha=0.1) | no | ok | 0.5531 | 0.3999 | 32 | 0 |

## 2) Attack-Parameter Sweep Analyses (`summary.json`)

| Group folder | Sweep | Rows | Target metric | Test R2 | Test MAE | Timestamp | Output path |
|---|---|---:|---|---:|---:|---|---|
| attack_parameter_impact | 20260510_124701 | 2 | best_ssim | NA | 0.7606 | 2026-05-10 12:47:01 | `results/attack_parameter_impact/20260510_124701` |
| attack_parameter_impact | 20260510_124755 | 2 | best_ssim | NA | 0.7606 | 2026-05-10 12:47:55 | `results/attack_parameter_impact/20260510_124755` |
| attack_parameter_impact | 20260510_141508 | 38 | best_ssim | -0.2280 | 0.2791 | 2026-05-10 14:15:08 | `results/attack_parameter_impact/20260510_141508` |
| attack_parameter_impact | bs1_allclients_samples0_25_50_75_100 | 100 | best_ssim | -0.1663 | 0.2138 | 2026-05-10 14:51:40 | `results/attack_parameter_impact/bs1_allclients_samples0_25_50_75_100` |
| attack_parameter_impact | medium_bs1_clients012_samples02550 | 36 | best_ssim | -0.3794 | 0.3198 | 2026-05-10 14:35:09 | `results/attack_parameter_impact/medium_bs1_clients012_samples02550` |
| attack_parameter_impact | pilot_v1 | 16 | best_ssim | -3766.4414 | 0.4729 | 2026-05-15 15:42:06 | `results/attack_parameter_impact/pilot_v1` |
| attack_parameter_impact | screening_blood_v1 | 384 | best_ssim | 0.8333 | 0.1060 | 2026-05-15 15:57:07 | `results/attack_parameter_impact/screening_blood_v1` |
| attack_parameter_impact | screening_blood_v1 | 384 | best_mse | 0.7357 | 0.0495 | 2026-05-16 12:40:32 | `results/attack_parameter_impact/screening_blood_v1_exploratory_results_delivery_package` |
| attack_parameter_impact_dermamnist | screening_derma_v1_reduced | 64 | best_ssim | 0.9565 | 0.0635 | 2026-05-16 14:26:47 | `results/attack_parameter_impact_dermamnist/screening_derma_v1_reduced` |
| attack_parameter_impact_mse | screening_blood_v1 | 384 | best_mse | 0.7357 | 0.0495 | 2026-05-16 12:40:32 | `results/attack_parameter_impact_mse/screening_blood_v1` |
| attack_parameter_impact_pathmnist | screening_path_v1_reduced | 64 | best_ssim | 0.9532 | 0.0538 | 2026-05-16 14:15:59 | `results/attack_parameter_impact_pathmnist/screening_path_v1_reduced` |
| attack_parameter_impact_pathmnist_partial | screening_path_v1_reduced | 21 | best_ssim | 0.9715 | 0.0346 | 2026-05-16 13:47:21 | `results/attack_parameter_impact_pathmnist_partial/screening_path_v1_reduced` |

## 3) Full DP Privacy-Utility Matrix Orchestration

| Models in matrix | Training ok/existing | Training failed | Attacks attempted | Attacks ok/existing | Attacks failed | Manifest |
|---:|---:|---:|---:|---:|---:|---|
| 20 | 5 | 15 | 120 | 45 | 75 | `results/full_dp_privacy_utility_matrix/run_manifest.csv` |
