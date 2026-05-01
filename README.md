# AIJack FL Privacy Attack (MNIST, CPU-only)

This repository contains a minimal, thesis-ready pipeline for:
- FedAvg baseline training with AIJack
- Gradient inversion attack experiments across batch sizes
- Robust attack summary statistics and plots

The experiment logic, model, data, and hyperparameters are unchanged.

## Active source structure

- `src/main_fedavg.py`: orchestrates baseline + batch-size attack experiments + summary plots
- `src/model.py`: MNIST `SmallCNN`
- `src/data.py`: MNIST loading, deterministic subset, IID client split, loaders
- `src/attack_gradient_inversion.py`: AIJack gradient inversion attack routine and per-attack artifacts
- `src/utils.py`: CPU `DEVICE`, seeding, directory creation, accuracy metric

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.main_fedavg
```

## Reproduced outputs

Main outputs:
- `results/baseline_metrics.csv`
- `results/gradient_inversion/batch_size_summary.csv`
- `results/gradient_inversion/batch_size_vs_mse.png`
- `results/gradient_inversion/batch_size_vs_median_mse.png`

Per batch-size outputs (`batch_size_1`, `batch_size_4`, `batch_size_8`):
- `attack_metrics.csv`
- `reconstruction_grid.png`
- per-attack original/reconstructed/comparison PNGs

## Fixed experiment setup (unchanged)

- Dataset: MNIST (`./data`)
- Train subset: 2000 samples
- Clients: 2 (IID split)
- FedAvg: 3 rounds, 1 local epoch, SGD lr `0.1`
- Batch-size experiments: `1`, `4`, `8`
- Attacks per batch size: `10`
- Gradient inversion iterations: `60`
- Device: CPU only
