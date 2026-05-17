# BloodMNIST Privacy-Utility Checklist

Use this as the working checklist after the BloodMNIST attack-calibration run.
Mark each item as checked only after the corresponding artifact exists and has
been inspected.

## Fixed Attacker

- [x] Confirm fixed attacker settings from calibration:
  `attack_batch_size=1`, `distance=cossim`, `attack_lr=0.05`,
  `attack_iters=300`, `num_trials=3`, clients `0,1,2`, samples `0,25,50`.
  Fixed protocol saved at
  `configs/current/attack_protocols/bloodmnist_fixed_attacker_v1.yaml`.
  Evidence inspected:
  `results/current/analysis/attack_calibration_bloodmnist_v3_cpu/analysis_report.md`,
  `group_level_summary.csv`, and `matched_contrasts.csv`. Batch size 1 is the
  only tested batch-size level with positive-MSE successes; `cossim` gives the
  strongest median leakage among tested distances; `attack_lr=0.05` is stronger
  than higher tested learning rates; and increasing iterations/trials did not
  improve matched median leakage.

## Smoke Run

- [x] Run the fixed-attacker smoke evaluation on:
  `iid_baseline`, `iid_dp_sigma_05`, and `noniid_alpha_05`.
  Smoke artifacts saved under
  `results/current/analysis/bloodmnist_fixed_attacker_smoke_v2/`.
- [x] Inspect the smoke `analysis_report.md` and `group_level_summary.csv`.
- [x] Confirm attacks mostly produce valid positive-MSE metrics.
  Fresh run produced 27 intended attack cells, 23 positive-MSE metrics, and 4
  retained no-MSE cells.

## Full BloodMNIST Evaluation

- [x] Run the fixed attacker across selected BloodMNIST baseline and DP-matrix
  models.
  Selected matrix: all four BloodMNIST splits (`iid`, `noniid_alpha_1`,
  `noniid_alpha_05`, `noniid_alpha_01`) with no-DP baselines and sigma=0.5 DP
  counterparts. Artifacts saved under
  `results/current/privacy_utility/bloodmnist_fixed_attacker_sigma05_v1/`.
- [x] Confirm every failed/no-MSE attack cell is retained and reported.
  The evaluation retained 72 attack cells, including 67 positive-MSE cells and
  5 no-MSE cells.
- [x] Aggregate results into a table with:
  experiment name, split, alpha, DP status, sigma, accuracy, macro-F1,
  attack success rate, median MSE, and median leakage score.
  Table saved as
  `results/current/privacy_utility/bloodmnist_fixed_attacker_sigma05_v1/model_privacy_utility_table.csv`.

## Figures

- [ ] Plot macro-F1 versus median leakage score.
- [ ] Plot macro-F1 versus median reconstruction MSE.
- [ ] Use color for `sigma` and shape/facet for split or `alpha`.

## Write-Up

- [ ] Write the BloodMNIST result paragraph explaining that attack calibration
  is separate from privacy-utility evaluation.
- [ ] State the leakage direction clearly:
  lower MSE means stronger leakage; higher leakage score means stronger leakage.

## Validation Dataset

- [ ] Repeat the fixed-attacker protocol on PathMNIST or DermaMNIST after the
  BloodMNIST table and plots are complete.
