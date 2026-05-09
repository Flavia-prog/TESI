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

## AIJack FedAvg BloodMNIST Baseline

Run the first federated baseline experiment (FedAvg, no attacks, no defenses):

```bash
source .venv/bin/activate

CPPFLAGS="-I$(brew --prefix boost)/include" \
CXXFLAGS="-I$(brew --prefix boost)/include" \
LDFLAGS="-L$(brew --prefix boost)/lib" \
python -m pip install -r requirements.txt

python scripts/fedavg_bloodmnist_aijack.py \
  --num-clients 5 \
  --num-rounds 20 \
  --local-epochs 1 \
  --batch-size 64 \
  --lr 0.001
```
