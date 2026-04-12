# Version 3 - Direct Per-Batch DLG/iDLG-Style Inversion

## Goal
Bypass FL aggregation artifacts and attack exact single-batch gradients directly.

## Method
- No FedAVG round aggregation in the attack path
- Compute exact gradients from one forward/backward pass on one sample:
  - `loss = CrossEntropy(model(x_true), y_true)`
  - `received_gradients = autograd.grad(loss, model.parameters())`
- Use `GradientInversion_Attack` directly on captured gradients
- Multi-restart + 2-stage schedule with warm-start

## Key Hyperparameters
- Optimizer: Adam
- Distance: l2
- Restarts: 5
- Coarse: `num_iter=900`, `lr=0.08`, `tv=5e-3`, `l2=6e-6`
- Refine: `num_iter=1500`, `lr=0.04`, `tv=7e-4`, `l2=1e-6`

## Observed Outcome
- Best attack loss (run shown): 0.009765
- Reconstruction MSE (normalized space): 0.010187
- Predicted class: correct
- Visual quality: clearly best among the three versions; strong shape/color recovery with much less noise

## Conclusion
Version 3 gives the strongest reconstruction because it attacks exact per-batch gradients, avoiding the information mixing introduced by FL update/aggregation mechanics.
