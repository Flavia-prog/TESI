# Version 2 - Multi-Round FedAVG Snapshot Attack

## Goal
Improve reconstruction by attacking multiple communication rounds and selecting the best snapshot.

## Method
- FL framework: AIJack FedAVG
- Communication rounds: 3
- Auto attack on `receive`: disabled
- Explicit attack loop: run after each round's `receive`
- Per-round selection: best trial per round
- Global selection: best round across all rounds
- Stage schedule per trial:
  - Coarse stage: stronger priors
  - Refine stage: weaker priors with warm-start (`init_x`)

## Key Hyperparameters
- Optimizer: Adam
- Distance: l2
- Restarts per round: 3
- Coarse: `num_iter=700`, `lr=0.08`, `tv=4e-3`, `l2=5e-6`
- Refine: `num_iter=1400`, `lr=0.04`, `tv=6e-4`, `l2=1e-6`

## Observed Outcome
- Best round: round 1
- Best round loss (run shown): about 0.258118
- Predicted class: correct
- Visual quality: clearer global structure than Version 1, still with color/pixel noise

## Main Limitation
Quality depends strongly on round-specific gradient snapshots; some rounds improve reconstruction while others degrade it.
