# AGENTS.md

This file is the source of truth for future coding agents working on this
repository. Read it before making changes. Treat `README.md`, experiment logs,
and status files as historical/background context when they disagree with this
file.

After any major change to the research direction, experiment design, scripts,
configs, datasets, metrics, result layout, or current findings, update this file
in the same work session.

## Project Goal

This is a master thesis project studying gradient inversion attacks in
federated learning. The core setting is an honest-but-curious server that tries
to reconstruct client training data from shared gradients.

The required framework is AIJack. Prefer AIJack-native implementations for
federated learning, attacks, and defenses whenever feasible. Relevant AIJack
capabilities include FedAvg, DLG/iDLG-style gradient inversion, GradInversion,
DP-SGD, gradient compression, and Soteria-style defenses.

The current thesis framing is privacy-utility focused:

> Which federated learning design and defense choices reduce reconstruction
> leakage under strong gradient inversion attacks while preserving model utility?

The exploratory attack-condition work is now treated as attack calibration. Use
it to define a strong or worst-case adversary, then evaluate FL configurations
under that adversary. The authoritative current experiment plan is
`PRIVACY_UTILITY_EXPERIMENT_PLAN.md`.

The broader thesis framing remains the privacy-utility trade-off in federated
learning: FL design and defense mechanisms can reduce reconstruction quality,
which is good for privacy, but they can also reduce model accuracy and macro-F1,
which is bad for utility.

## Datasets and Scope

BloodMNIST is the primary dataset and first experimental target. It is an
8-class blood cell classification dataset from MedMNIST, used at 28x28 or
224x224 depending on experiment feasibility.

Additional datasets are required to validate findings and avoid conclusions
that are only BloodMNIST-specific. The current repo already includes PathMNIST
and DermaMNIST data/runs; use these as default validation datasets unless a
later thesis decision chooses different datasets.

Keep experiments laptop-scale by default. The working machine is a MacBook Air
2024, so prefer CPU/MPS-aware, reproducible experiments over large exhaustive
sweeps that are unlikely to finish locally. The thesis timeline is short
approximately 3 months from the current project state, and early professor
updates are important.

## Current Repository State

The repository currently contains:

- AIJack FedAvg training for BloodMNIST.
- Generalized MedMNIST training and attack scripts for additional datasets.
- IID and Dirichlet non-IID client splits.
- DP-style defense experiments using clipping/noise and sigma-based configs.
- Gradient inversion attack evaluation as a separate stage after FL training.
- Attack-parameter sweeps and exploratory regression/feature-importance
  analysis.
- Privacy-utility frontier plotting from aggregated DP matrix summaries.
- Historical experiment summaries and logs with both successful and failed
  runs.

Important scripts:

- `scripts/fedavg_bloodmnist_aijack.py`: BloodMNIST FedAvg baseline training.
- `scripts/fedavg_bloodmnist_aijack_dp.py`: BloodMNIST FedAvg with DP-style
  defense.
- `scripts/fedavg_medmnist_aijack.py`: generalized MedMNIST FedAvg training.
- `scripts/gradient_inversion_bloodmnist_aijack.py`: BloodMNIST gradient
  inversion attack evaluation.
- `scripts/gradient_inversion_medmnist_aijack.py`: generalized MedMNIST
  gradient inversion attack evaluation.
- `scripts/exploratory_reconstruction_parameter_analysis.py`: standalone
  AIJack sweep-and-analysis script for studying which attacker/design
  parameters affect reconstruction success. With `--run-aijack-sweep`, it
  executes the AIJack-based `gradient_inversion_medmnist_aijack.py` over a
  controlled grid of experiment directories, clients, samples, attack batch
  sizes, iteration counts, trials, attack learning rates, and AIJack distance
  metrics, saving per-cell outputs and an `aijack_sweep_manifest.csv`. For new
  attack-condition evidence, prefer `--sweep-design matched-ofat` over the
  default full-factorial design: it creates one-factor-at-a-time comparison
  blocks with explicit `comparison_id`, `varied_parameter`, and
  `comparison_level` metadata so each parameter can be compared while the other
  attack settings are held fixed. Use `--ofat-anchor-clients` and
  `--ofat-anchor-sample-indices` to repeat each comparison across nuisance
  client/sample contexts for more robust matched contrasts. It can also run in
  analysis-only mode on
  existing metric CSVs and/or
  `attack_metrics.json` files. It retains failed/no-MSE rows in the normalized
  dataset, reports group summaries, held-out cross-validated ridge permutation
  screening, numeric Spearman screening, and matched within-setting contrasts.
  Treat its model-based importance as predictive prioritization, not causal
  evidence; prefer matched contrasts and repeated validation across datasets
  before making thesis claims.
- `scripts/attack_parameter_impact_bloodmnist.py`: attack sweep and exploratory
  regression/feature-importance analysis. Random-forest permutation importance
  from this script is a screening statistic, not a causal estimate. Prefer the
  script's `controlled_pairwise_effects.csv` matched contrasts for thesis claims
  when enough matched contrasts are available. The script also saves
  `parameter_importance.png` and `controlled_pairwise_effects.png` for professor
  updates and thesis figure drafts.
- `scripts/run_full_dp_privacy_utility_matrix.py`: orchestration for the DP
  privacy-utility matrix.
- `scripts/run_bloodmnist_fixed_attacker_eval.py`: current BloodMNIST
  fixed-attacker privacy-utility evaluator. It runs the calibrated attacker
  across saved baseline and DP-matrix models, retains failed/no-MSE cells, and
  writes `attack_cell_summary.csv`, `model_privacy_utility_table.csv`, and
  `analysis_report.md`.
- `scripts/run_bloodmnist_block_a_frontier.py`: Block A orchestrator for the
  BloodMNIST sigma frontier. It validates the fixed-attacker protocol, runs the
  fixed evaluator across all existing no-DP and sigma-matrix BloodMNIST models,
  then writes Block-A-specific tables, plots, and report artifacts.
- `scripts/plot_frontier.py`: headline privacy-utility frontier plots.

Archived scripts:

- `scripts/_archive_20260516/plot_day2_screening_results.py`: old professor
  update plotting helper. Its outputs are already preserved under
  `results/current/analysis/attack_parameter_impact/`.
- `scripts/_archive_20260516/build_day3_exploratory_delivery_package.py`: old
  delivery-package helper that depends partly on archived MSE analysis outputs.
  Keep for historical regeneration only.
- The empty `scripts/run_experiment.py` wrapper was removed during the
  2026-05-16 cleanup.

Important result/status references:

- `EXPERIMENTS_RUN_SO_FAR.md`: compact table of completed training runs,
  attack sweeps, and DP matrix status.
- `BLOODMNIST_FRONTIER_EXPERIMENT_MATRIX.md`: current detailed table of
  planned BloodMNIST federated privacy-utility frontier experiments. It fixes
  the calibrated attacker and varies FL-side parameters including sigma,
  clipping norm, local epochs, number of clients, training batch size, split
  heterogeneity, seeds, and validation datasets.
- `EXPERIMENT_LOG.md`: historical manual log of utility and attack experiments.
- `results/current/`: current thesis-facing outputs. Training runs are under
  `results/current/training/`, attack-parameter analyses are under
  `results/current/analysis/attack_parameter_impact/`, the standalone
  exploratory reconstruction-parameter analysis is under
  `results/current/analysis/exploratory_reconstruction_parameter_analysis/`,
  and privacy-utility matrix summaries are under `results/current/privacy_utility/`.
- `results/current/privacy_utility/bloodmnist_block_a_frontier_v1/`: completed
  2026-05-17 BloodMNIST Block A sigma-frontier fixed-attacker evaluation across
  24 models and 216 retained attack cells. Key outputs are
  `block_a_report.md`, `block_a_model_privacy_utility_table.csv`,
  `attack_cell_summary.csv`, and the `figures/` privacy-utility plots,
  including macro-F1 vs median MSE, macro-F1 vs sigma, median MSE vs sigma,
  and attack success rate vs sigma. The macro-F1 vs median MSE plot is now a
  two-panel figure: measured MSE frontier on the left and all trained models
  with no-positive-MSE attack status on the right, to avoid hiding failed/no-MSE
  attack cells.
  The run retained 78 positive-MSE cells and 138 failed/no-MSE cells. Current
  interpretation: no DP sigma setting preserved baseline macro-F1 within 10
  percentage points for IID, alpha=0.5, or alpha=1 splits; alpha=0.1 at
  sigma 0.25 and 0.5 stayed within 15 percentage points but not within 10.
  Treat zero attack success at some DP settings cautiously, because many cells
  failed with no MSE rather than producing successful high-MSE reconstructions.
- `results/_archive_low_value_20260516/`: older pilots, partial duplicates,
  individual attack-output folders, run logs, generated cache/system files, and
  other low-value generated artifacts. They were archived rather than deleted.
- `configs/current/`: active baseline and sigma-based DP matrix configs.
  The fixed BloodMNIST attacker protocol selected from calibration is
  `configs/current/attack_protocols/bloodmnist_fixed_attacker_v1.yaml`.
  Legacy fixed-noise configs were moved to
  `configs/_archive_legacy_noise_20260516/`.

Failed trainings and failed attacks are informative project state. Do not hide
or silently discard them. When they affect an analysis, report them explicitly
and explain whether they are excluded, retried, or treated as failed cells.

## Working Rules for Agents

- Treat `AGENTS.md` as authoritative for project intent and workflow.
- Read `README.md` only as background unless the user explicitly asks to update
  it.
- Do not edit README or experiment logs unless the user asks for that specific
  documentation change.
- Preserve AIJack as the required framework for FL, attacks, and defenses where
  feasible.
- Prefer reproducible scripts, configs, and saved outputs over ad hoc notebook
  work.
- Keep changes scoped to the requested research or engineering task.
- Do not remove existing experiment artifacts unless the user explicitly asks.
- When adding experiments, save the resolved config, metrics, plots, and summary
  artifacts consistently under `results/`.
- Use clear experiment names that encode dataset, split, defense, attack, and
  key parameters.
- When a major change is made, update this file immediately.

## Experiment Priorities

Prioritize work that helps answer:

> Which FL configurations offer the best privacy-utility trade-off under a
> strong calibrated gradient inversion attack?

Use a two-stage design:

1. Attack calibration: use exploratory matched sweeps, regression, and feature
   importance to choose strong attacker settings. Candidate attack-side factors
   include client id, sample index, attack batch size, attack iterations, number
   of trials, attack learning rate, and distance metric.
2. Privacy-utility evaluation: train FL models with different design/defense
   choices, then attack all of them with the calibrated strong adversary.
   Candidate FL-side factors include dataset, client split, Dirichlet alpha,
   number of clients, local epochs, training batch size, defense type, clipping
   norm, DP sigma/noise multiplier, and epsilon where available.

Do not mix the two roles. Attacker parameters calibrate the evaluation protocol;
FL/design/defense parameters define the thesis privacy-utility comparisons.

Measure reconstruction success quantitatively where possible:

- MSE: lower usually means stronger reconstruction and more leakage.
- SSIM: higher usually means stronger reconstruction and more leakage.
- Add other metrics only when they are useful and consistently available.

Use qualitative image grids when they help interpret reconstruction behavior,
but do not rely on qualitative inspection alone for thesis claims.

Measure utility with:

- Test accuracy.
- Test macro-F1, especially because class imbalance can make accuracy
  misleading.

For privacy-utility frontier work, compare leakage metrics against utility
metrics across heterogeneity levels and defense strengths. Make the direction of
each leakage metric explicit in plots and summaries.

## Reproducibility Expectations

Training runs should save at least:

- `config.yaml`
- `history.csv`
- `test_metrics.csv`
- `client_distributions.csv`
- `confusion_matrix.png`
- `final_model.pt`

Attack runs should save original/reconstructed visual outputs plus machine
readable metrics, normally `attack_metrics.json` or a clearly named equivalent.

Aggregated analyses should save CSV/JSON summaries alongside plots so thesis
figures can be regenerated or checked later.

For attack-parameter impact analysis, report both:

- predictive screening results, for example grouped permutation importance; and
- controlled matched contrasts that compare one varied attack/design parameter
  while holding the other selected sweep parameters fixed.

Make the leakage direction explicit. For MSE, lower values indicate stronger
reconstruction/leakage; for SSIM-like scores, higher values indicate stronger
reconstruction/leakage.

Before running large sweeps, prefer a dry run or capped screening design when
available. Avoid launching broad experiments that do not fit the MacBook Air
constraint unless the user explicitly accepts the runtime.

## Professor Update Bias

The project needs usable updates for the professor as soon as possible. When
choosing between equally valid next tasks, prefer work that produces a clear
summary, table, plot, or defensible intermediate finding over work that only
expands infrastructure.

Good update artifacts include:

- A concise table of completed runs and failed runs.
- A short explanation of the most important attack parameters found so far.
- A privacy-utility plot or preliminary frontier.
- A small set of representative reconstructions with matching quantitative
  metrics.
