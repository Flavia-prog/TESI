# BloodMNIST Privacy-Utility Checklist

Use this as the working checklist after the BloodMNIST attack-calibration run.
Mark each item as checked only after the corresponding artifact exists and has
been inspected.

## Fixed Attacker

- [ ] Confirm fixed attacker settings from calibration:
  `attack_batch_size=1`, `distance=cossim`, `attack_lr=0.05`,
  `attack_iters=300`, `num_trials=3`, clients `0,1,2`, samples `0,25,50`.

## Smoke Run

- [ ] Run the fixed-attacker smoke evaluation on:
  `iid_baseline`, `iid_dp_sigma_05`, and `noniid_alpha_05`.
- [ ] Inspect the smoke `analysis_report.md` and `group_level_summary.csv`.
- [ ] Confirm attacks mostly produce valid positive-MSE metrics.

## Full BloodMNIST Evaluation

- [ ] Run the fixed attacker across selected BloodMNIST baseline and DP-matrix
  models.
- [ ] Confirm every failed/no-MSE attack cell is retained and reported.
- [ ] Aggregate results into a table with:
  experiment name, split, alpha, DP status, sigma, accuracy, macro-F1,
  attack success rate, median MSE, and median leakage score.

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
