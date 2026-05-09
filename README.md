## Centralized BloodMNIST Baseline

Run the first centralized baseline experiment (no federated learning) with:

```bash
source .venv/bin/activate
python scripts/baseline_bloodmnist.py --epochs 20 --batch-size 64 --lr 0.001
```

The script will:
- train a centralized CNN on BloodMNIST official train split,
- validate on the official validation split each epoch,
- evaluate on the official test split using the best validation model,
- save outputs under `results/`:
  - `baseline_bloodmnist_best.pt`
  - `baseline_bloodmnist_metrics.csv`
  - `baseline_bloodmnist_confusion_matrix.png`
