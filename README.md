# FL Privacy Thesis Pipeline

Repository for thesis experiments on privacy-utility tradeoffs in Federated Learning under gradient inversion attacks.

## Canonical Entrypoints

- `src/LHS.py`: generate experiment plans (`experiment_plan.csv`)
- `src/run_thesis_experiments.py`: run image + text pipelines and produce unified tables
- `src/pipeline_a_bloodmnist.py`: image modality pipeline
- `src/pipeline_b_text.py`: text modality baseline pipeline

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Runbook

1. Generate design points:
```bash
python -m src.LHS
```

2. Run thesis experiments:
```bash
python -m src.run_thesis_experiments
```

3. Curate final outputs:
- Keep thesis-cited files in `results/final/`
- Temporary or exploratory files go to `results/tmp/`

## Output Convention

- `results/tradeoff/tradeoff_master.csv`: merged image/text analysis table
- `results/tradeoff/mlr_coefficients_accuracy.csv`: OLS terms for utility
- `results/tradeoff/mlr_coefficients_reconstruction.csv`: OLS terms for reconstruction
