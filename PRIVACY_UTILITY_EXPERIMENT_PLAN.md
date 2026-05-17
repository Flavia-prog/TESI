# Privacy-Utility Experiment Plan

This file defines the current thesis experiment direction. Historical logs and
previous exploratory results remain useful background, but this plan should guide
new experiments.

## Goal

Assess the privacy-utility trade-off in federated learning under strong gradient
inversion attacks.

The thesis contribution is not only to show that attacks work. The contribution
is to use AIJack to understand which FL design and defense choices reduce
reconstruction leakage while preserving useful model performance.

## Experimental Logic

Use a two-stage design.

### Stage 1: Calibrate a Strong Attack

Purpose: choose a strong, defensible adversary for later privacy-utility
evaluation.

Vary attacker-side settings on a small set of trained baseline models:

- `attack_batch_size`
- `attack_iters`
- `num_trials`
- `attack_lr`
- `distance`
- `client_id`
- `sample_index`

Use matched one-factor-at-a-time comparisons where possible. Treat regression
and permutation importance as screening tools only. The output of this stage is
a fixed or small worst-case attack protocol, for example the settings that give
the strongest leakage across a pilot grid.

Report both:

- median leakage across clients/samples; and
- worst observed leakage across clients/samples/attack settings.

### Stage 2: Evaluate FL Design and Defenses

Purpose: measure privacy and utility for FL configurations built from the start
with different design/defense choices.

Train separate FL models for each design choice. Then attack every trained model
using the calibrated strong attack protocol from Stage 1.

Candidate FL-side variables:

- dataset
- IID vs Dirichlet non-IID split
- Dirichlet `alpha`
- number of clients
- local epochs
- training batch size
- DP clipping norm
- DP noise multiplier / sigma
- gradient compression, if feasible
- Soteria-style defense, if feasible

Do not treat these as attack-only parameters. If a variable changes the FL
training process, it requires a separate training run and saved utility metrics.

## Metrics

Utility:

- test accuracy
- test macro-F1
- test loss and confusion matrix when available

Privacy:

- reconstruction MSE, where lower means stronger leakage
- `leakage_score = -log10(reconstruction_mse)`, where higher means stronger
  leakage
- SSIM-like metrics only if implemented consistently
- failure/no-reconstruction rate, reported explicitly
- representative image grids as qualitative support only

## Main Comparison

For each FL configuration, produce one row with:

- training configuration metadata
- utility metrics
- median leakage under the calibrated attack
- worst-case leakage under the calibrated attack
- number of successful and failed attacks

Plot privacy-utility frontiers:

- x-axis: macro-F1 or accuracy
- y-axis: leakage metric, with direction made explicit
- separate facets or markers for split, defense, dataset, or number of clients

## Scope Control

Keep the design laptop-scale.

Recommended sequence:

1. Run a capped matched attack-calibration pilot.
2. Select a strong attack protocol.
3. Run the main BloodMNIST privacy-utility matrix.
4. Validate only the main conclusions on PathMNIST and DermaMNIST.
5. Add more defenses only after the baseline and DP frontier are stable.

Do not delete or overwrite old experiment artifacts. If old results are no
longer thesis-facing, leave them archived or clearly mark them as historical.
