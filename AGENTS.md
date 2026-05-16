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

The primary research question is attack-focused:

> How do attack conditions affect reconstruction success?

The broader thesis framing is the privacy-utility trade-off in federated
learning: FL and defense mechanisms can reduce reconstruction quality, which is
good for privacy, but they can also reduce model accuracy and macro-F1, which is
bad for utility.

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
- `scripts/attack_parameter_impact_bloodmnist.py`: attack sweep and exploratory
  regression/feature-importance analysis. Random-forest permutation importance
  from this script is a screening statistic, not a causal estimate. Prefer the
  script's `controlled_pairwise_effects.csv` matched contrasts for thesis claims
  when enough matched contrasts are available. The script also saves
  `parameter_importance.png` and `controlled_pairwise_effects.png` for professor
  updates and thesis figure drafts.
- `scripts/run_full_dp_privacy_utility_matrix.py`: orchestration for the DP
  privacy-utility matrix.
- `scripts/plot_frontier.py`: headline privacy-utility frontier plots.

Important result/status references:

- `EXPERIMENTS_RUN_SO_FAR.md`: compact table of completed training runs,
  attack sweeps, and DP matrix status.
- `EXPERIMENT_LOG.md`: historical manual log of utility and attack experiments.
- `results/current/`: current thesis-facing outputs. Training runs are under
  `results/current/training/`, attack-parameter analyses are under
  `results/current/analysis/attack_parameter_impact/`, and privacy-utility
  matrix summaries are under `results/current/privacy_utility/`.
- `results/_archive_low_value_20260516/`: older pilots, partial duplicates,
  individual attack-output folders, run logs, generated cache/system files, and
  other low-value generated artifacts. They were archived rather than deleted.
- `configs/current/`: active baseline and sigma-based DP matrix configs.
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

> How do attack conditions affect reconstruction success?

Use exploratory regression and feature importance to decide which parameters
deserve deeper experiments. Candidate factors include dataset, client split,
Dirichlet alpha, client id, sample index, attack batch size, attack iterations,
number of trials, attack learning rate, distance metric, defense type, DP sigma,
and epsilon where available.

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
