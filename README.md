# FL Thesis Experiments

This repository contains federated learning experiments for BloodMNIST/PathMNIST using AIJack.

## Package-Based Workflow (New)

The codebase now includes a package-style layout under `src/thesis/`:

- `thesis.data`: dataset loaders and client split utilities
- `thesis.models`: CNN registry (`small_cnn`, `medium_cnn`, `lenet`)
- `thesis.federated`: reusable FedAvg and FedAvg+DP training functions
- `thesis.attacks`: reusable in-process gradient inversion attack runner
- `thesis.metrics`: reconstruction metrics (MSE, windowed SSIM, optional LPIPS)
- `thesis.experiments`: dataclass configs and in-process sweep runner
- `thesis.utils`: seed/device/io/provenance helpers

Thin CLI wrappers are available in `scripts/`:

- `python scripts/train_fedavg.py --config ...`
- `python scripts/train_fedavg_dp.py --config ...`
- `python scripts/run_attack.py --experiment-dir ...`
- `python scripts/run_sweep.py --config ...`

The goal is to study how to build federated learning models while balancing utility and privacy. The current stage focuses on building reproducible FedAvg baselines under IID and non-IID client data distributions.

Legacy standalone scripts (for backward compatibility) still exist in `scripts/`, but the package wrappers above are the recommended entry points.

## Environment Setup

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
CPPFLAGS="-I$(brew --prefix boost)/include" \
CXXFLAGS="-I$(brew --prefix boost)/include" \
LDFLAGS="-L$(brew --prefix boost)/lib" \
python -m pip install -r requirements.txt
```

AIJack requires Boost, CMake, and Ninja because it builds native C++ extensions during installation.

## Current Experiment: AIJack FedAvg on BloodMNIST

The current baseline is an AIJack-based federated learning experiment using FedAvg on BloodMNIST.

Experiment setup:

- Dataset: BloodMNIST
- Framework: AIJack
- Algorithm: FedAvg
- Model: CNN classifier
- Optimizer: SGD
- No attacks
- No defenses

The script uses the core AIJack FedAvg components:

- `FedAVGClient`
- `FedAVGServer`
- `FedAVGAPI`

## Reproducible IID Baseline

Run the IID baseline using the YAML config:

```bash
python scripts/train_fedavg.py \
  --config configs/iid_baseline.yaml
```

The configuration is stored in:

```text
configs/iid_baseline.yaml
```

Results are saved to:

```text
results/iid_baseline/
```

The results folder contains:

```text
config.yaml
history.csv
test_metrics.csv
client_distributions.csv
confusion_matrix.png
final_model.pt
```

## Non-IID Experiments

Non-IID experiments use Dirichlet label-skew splitting to simulate realistic federated learning settings where clients have different class distributions.

In this setup:

- IID means clients receive random, similar class distributions.
- Dirichlet non-IID means clients receive different class proportions.
- Smaller `alpha` means stronger heterogeneity.
- `alpha = 1.0` is mild non-IID.
- `alpha = 0.5` is moderate non-IID.
- `alpha = 0.1` is strong non-IID.

Run the non-IID experiments:

```bash
python scripts/train_fedavg.py \
  --config configs/noniid_alpha_1.yaml

python scripts/train_fedavg.py \
  --config configs/noniid_alpha_05.yaml

python scripts/train_fedavg.py \
  --config configs/noniid_alpha_01.yaml
```

## PathMNIST Equivalents

Use the same training script with dataset override:

```bash
python scripts/train_fedavg.py \
  --config configs/iid_baseline.yaml \
  --dataset pathmnist \
  --experiment-name pathmnist_iid_baseline

python scripts/train_fedavg.py \
  --config configs/noniid_alpha_01.yaml \
  --dataset pathmnist \
  --experiment-name pathmnist_noniid_alpha_01
```

Results are saved to:

```text
results/noniid_alpha_1/
results/noniid_alpha_05/
results/noniid_alpha_01/
```

Each folder contains:

```text
config.yaml
history.csv
test_metrics.csv
client_distributions.csv
confusion_matrix.png
final_model.pt
```

## Manual Run Without YAML

The experiment can also be run manually without a YAML config:

```bash
python scripts/train_fedavg.py \
  --num-clients 5 \
  --num-rounds 20 \
  --local-epochs 1 \
  --batch-size 64 \
  --lr 0.01 \
  --split-type iid \
  --seed 42 \
  --device cpu
```

## Experiment Outputs

Each experiment saves:

- `config.yaml`: the final resolved configuration used for the run
- `history.csv`: validation metrics after each communication round
- `test_metrics.csv`: final test metrics
- `client_distributions.csv`: class distribution for each client
- `confusion_matrix.png`: test confusion matrix
- `final_model.pt`: saved final model weights

The `client_distributions.csv` file is especially important for comparing IID and non-IID experiments because it shows how the BloodMNIST classes are distributed across clients.

## Current Baseline Result

The working IID FedAvg baseline achieved approximately:

```text
Test accuracy: 0.666
Test macro-F1: 0.579
```

Configuration:

```text
Dataset: BloodMNIST
Framework: AIJack
Algorithm: FedAvg
Clients: 5
Rounds: 20
Local epochs: 1
Batch size: 64
Learning rate: 0.01
Optimizer: SGD
Split: IID
Seed: 42
Device: CPU
```

This confirms that the AIJack FedAvg pipeline is working and that the CNN global model learns meaningful BloodMNIST class distinctions.

This baseline is intentionally lower than SOTA centralized BloodMNIST benchmarks. The thesis focus is privacy attack behavior and privacy-utility tradeoffs under controlled FL conditions, not utility maximization.

## Next Steps

Planned next steps:

1. Run controlled attack batch-size sweeps and quantify reconstruction quality changes.
2. Run controlled architecture sweeps across `small_cnn`, `medium_cnn`, and `lenet`.
3. Run training-stage sweeps to measure vulnerability across FL rounds.
4. Replicate key attack-condition findings on PathMNIST.
5. Run DP-SGD privacy-utility-defense sweeps as supporting experiments.

## Separate Attack Evaluation Script

Gradient inversion is implemented as a separate stage from FL training.

Training stage:

- `scripts/train_fedavg.py` trains FedAvg and saves artifacts in `results/<experiment_name>/`.

Attack stage:

- `scripts/run_attack.py` loads a trained experiment folder and evaluates privacy leakage.
- It uses AIJack `GradientInversionAttackServerManager` to attach a malicious FedAvg server and reconstruct private client images from shared gradients.

IID example:

```bash
python scripts/run_attack.py \
  --experiment-dir results/iid_baseline \
  --client-id 0 \
  --sample-index 0 \
  --attack-batch-size 1 \
  --attack-iters 300 \
  --num-trials 3 \
  --device cpu
```

Non-IID example:

```bash
python scripts/run_attack.py \
  --experiment-dir results/noniid_alpha_05 \
  --client-id 0 \
  --sample-index 0 \
  --attack-batch-size 1 \
  --attack-iters 300 \
  --num-trials 3 \
  --device cpu
```

The attack script:

1. Loads `config.yaml` and `final_model.pt` from the experiment folder.
2. Recreates the BloodMNIST client split using the saved training config.
3. Selects one client and one sample (or a very small attack batch).
4. Builds an AIJack malicious FedAvg server using `GradientInversionAttackServerManager`.
5. Runs one communication round with `FedAVGAPI(..., use_gradients=True)` to trigger inversion.
6. Saves original and reconstructed images.
7. Saves attack metrics to JSON.

## DP Defense Experiments

DP-style defense experiments are available in:

- `scripts/train_fedavg_dp.py`

This script runs the same BloodMNIST FedAvg pipeline with per-client update:

- L2 clipping
- Gaussian noise

Implementation detail: it uses AIJack DP components (`DPSGDManager` + `DPSGDClientManager`) together with `FedAVGClient`, `FedAVGServer`, and `FedAVGAPI`.

Run a DP experiment:

```bash
python scripts/train_fedavg_dp.py --config configs/iid_dp_noise_001.yaml
```

Run gradient inversion attack on the trained DP model:

```bash
python scripts/run_attack.py \
  --experiment-dir results/iid_dp_noise_001 \
  --client-id 0 \
  --sample-index 0 \
  --attack-batch-size 1 \
  --attack-iters 1000 \
  --num-trials 5 \
  --attack-lr 0.1 \
  --distance cossim \
  --device cpu \
  --output-dir results/iid_dp_noise_001/attacks/batch1_cossim_1000iters_5trials_lr01_client0_sample0
```
