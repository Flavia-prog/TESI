# Version 1 - Single-Round FedAVG + Restarts

## Goal
Run gradient inversion in the FedAVG pipeline with a stable baseline configuration.

## Method
- FL framework: AIJack FedAVG
- Communication rounds: 1
- Client setup: 2 clients, each constrained to 1 sample (`batch_size=1`)
- Attack target: client 0 uploaded gradient
- Attack model: `GradientInversionAttackServerManager`
- Label handling: `optimize_label=False`, pass true label index

## Key Hyperparameters
- Optimizer: Adam
- Distance: l2
- Iterations: 2000
- Attack restarts: 5
- TV regularization: 1e-3
- L2 regularization: 1e-6
- Clamp range: [-1, 1]

## Observed Outcome
- Best attack loss (run shown): about 0.824676
- Predicted class: correct
- Visual quality: recognizable class structure, but still noisy/pixelated

## Main Limitation
Even with `batch_size=1`, the FedAVG uploaded signal is still tied to local-update dynamics, which weakens pixel-level recovery quality compared with direct per-batch gradients.
