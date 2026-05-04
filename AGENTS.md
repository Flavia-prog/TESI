# AGENTS.MD

## 1) Thesis Context (from proposal)
- Topic: adversarial attacks on Federated Learning (FL) using AIJack.
- Core question: to what extent shared gradients/updates allow private information reconstruction.
- Current repository focus: image-domain experiments (mainly MNIST, plus archived work on MedMNIST and COVID radiography).

## 2) Repository Scope and What Is Active
This repository has two layers:
- Active/refactored layer: `src/`, `notebooks/`, `results/`, `data/`.
- Historical/archival layer: `archive/` with many old notebooks and a very large radiography dataset snapshot.

Practical entry points for current work:
- `src/pipeline_a_bloodmnist.py` (image modality DP + inversion)
- `src/pipeline_b_text.py` (text modality baseline DP + inversion)
- `src/run_thesis_experiments.py` (unified image/text runner + master outputs)
- `src/LHS.py` (Latin Hypercube experiment plan generation)

## 3) Execution Architecture (how code is organized)
- `src/pipeline_a_bloodmnist.py`: BloodMNIST image modality pipeline with per-sample DP clipping/noise, inversion (cosine + SignedAdam), SSIM, utility accuracy, and epsilon approximation.
- `src/pipeline_b_text.py`: synthetic clinical-text modality pipeline (bag-of-words), per-sample DP clipping/noise, inversion (cosine + SignedAdam), reconstruction similarity, utility accuracy, and epsilon approximation.
- `src/run_thesis_experiments.py`: orchestration entrypoint that runs both modalities from one plan, creates `tradeoff_master.csv`, adds interaction terms, and exports OLS coefficient tables.
- `src/LHS.py`: DOE plan generator with modality balancing support.

## 4) Full File Map (useful, exhaustive by file or exact pattern)

### 4.1 Root files
- `.gitignore`: ignores Python/cache artifacts and temporary results (`results/tmp/`).
- `.DS_Store`: macOS metadata.
- `AGENTS.MD`: this mapping document.

### 4.2 `src/` (core code)
- `src/__init__.py`: package marker (empty).
- `src/main_fedavg.py`: MNIST FedAVG + attack baselines.
- `src/attack_gradient_inversion.py`: MNIST inversion utilities (cosine + SignedAdam).
- `src/attack_membership_inference.py`: confidence-threshold MIA pipeline.
- `src/model.py`: MNIST CNN model(s).
- `src/data.py`: MNIST loading/splitting helpers.
- `src/utils.py`: seed, accuracy, filesystem helpers.
- `src/pipeline_a_bloodmnist.py`: active image-modality thesis pipeline.
- `src/pipeline_b_text.py`: active text-modality thesis pipeline.
- `src/run_thesis_experiments.py`: unified thesis runner.
- `src/LHS.py`: experiment-plan generation.

### 4.3 `notebooks/` (active notebooks)
- `notebooks/01_baseline.ipynb`: baseline pipeline invocation (4 code cells).
- `notebooks/02_experiments.ipynb`: grid runs + result visualization (4 code cells).

#### `notebooks/data/MNIST/raw/` (duplicated local MNIST cache)
- `notebooks/data/MNIST/raw/train-images-idx3-ubyte`
- `notebooks/data/MNIST/raw/train-images-idx3-ubyte.gz`
- `notebooks/data/MNIST/raw/train-labels-idx1-ubyte`
- `notebooks/data/MNIST/raw/train-labels-idx1-ubyte.gz`
- `notebooks/data/MNIST/raw/t10k-images-idx3-ubyte`
- `notebooks/data/MNIST/raw/t10k-images-idx3-ubyte.gz`
- `notebooks/data/MNIST/raw/t10k-labels-idx1-ubyte`
- `notebooks/data/MNIST/raw/t10k-labels-idx1-ubyte.gz`

### 4.4 `data/` (project MNIST cache)
#### `data/MNIST/raw/`
- `data/MNIST/raw/train-images-idx3-ubyte`
- `data/MNIST/raw/train-images-idx3-ubyte.gz`
- `data/MNIST/raw/train-labels-idx1-ubyte`
- `data/MNIST/raw/train-labels-idx1-ubyte.gz`
- `data/MNIST/raw/t10k-images-idx3-ubyte`
- `data/MNIST/raw/t10k-images-idx3-ubyte.gz`
- `data/MNIST/raw/t10k-labels-idx1-ubyte`
- `data/MNIST/raw/t10k-labels-idx1-ubyte.gz`

### 4.5 `results/` (current outputs)
- `results/final/`: thesis-curated final tables/figures.
- `results/tmp/`: exploratory/intermediate run artifacts.
- `results/tradeoff/`: unified image-text experiment tables and regressions (created by `src/run_thesis_experiments.py`).

### 4.6 `archive/` (historical artifacts)
- `archive/baselin.ipynb`: consolidated historical notebook with baseline + multiple experiment sections.
- `archive/results_refactored/mnist_dp_batchsize_tradeoff.csv`: archived copy of refactored results CSV.

### 4.7 `archive/prove vecchie/` (legacy notebooks, scripts, datasets, old outputs)

#### Legacy notebooks/scripts (all files)
- `archive/prove vecchie/12marzo.ipynb`
- `archive/prove vecchie/13marzo.ipynb`
- `archive/prove vecchie/FL1 copy.ipynb`
- `archive/prove vecchie/FL1.ipynb`
- `archive/prove vecchie/aaa.ipynb`
- `archive/prove vecchie/baselin copy.ipynb`
- `archive/prove vecchie/covid2.ipynb`
- `archive/prove vecchie/covid_Fl.ipynb`
- `archive/prove vecchie/createbaseline.ipynb`
- `archive/prove vecchie/esempio1_MNIST.ipynb`
- `archive/prove vecchie/experimentsuround.ipynb`
- `archive/prove vecchie/full_gem.ipynb`
- `archive/prove vecchie/full_gemi.ipynb`
- `archive/prove vecchie/medmnist.ipynb`
- `archive/prove vecchie/new_version.ipynb`
- `archive/prove vecchie/prova.ipynb`
- `archive/prove vecchie/prova2.ipynb`
- `archive/prove vecchie/script_con_gan.ipynb`
- `archive/prove vecchie/script_con_gan_sistemato.ipynb`
- `archive/prove vecchie/tentative1.ipynb`
- `archive/prove vecchie/experiments/exp_baseline_fedavg.py` (standalone MedMNIST FedAVG + inversion demo)
- `archive/prove vecchie/build_log.txt`
- `archive/prove vecchie/dlg_mnist_example.png`

#### Legacy MNIST raw copy
- `archive/prove vecchie/raw/train-images-idx3-ubyte`
- `archive/prove vecchie/raw/train-images-idx3-ubyte.gz`
- `archive/prove vecchie/raw/train-labels-idx1-ubyte`
- `archive/prove vecchie/raw/train-labels-idx1-ubyte.gz`
- `archive/prove vecchie/raw/t10k-images-idx3-ubyte`
- `archive/prove vecchie/raw/t10k-images-idx3-ubyte.gz`
- `archive/prove vecchie/raw/t10k-labels-idx1-ubyte`
- `archive/prove vecchie/raw/t10k-labels-idx1-ubyte.gz`

#### Legacy CSV outputs
- `archive/prove vecchie/results_activation_comparison.csv`
- `archive/prove vecchie/results_batch_size_comparison.csv`
- `archive/prove vecchie/results_complexity_comparison.csv`
- `archive/prove vecchie/results_dp_comparison.csv`
- `archive/prove vecchie/results_privacy_heatmap.csv`
- `archive/prove vecchie/results_privacy_utility_equilibrium.csv`
- `archive/prove vecchie/results_refactored_old/mnist_dp_batchsize_tradeoff.csv`

#### Legacy summary folders
- `archive/prove vecchie/results_old/baseline/attack_vs_round_multisecret_raw.csv`
- `archive/prove vecchie/results_old/baseline/attack_vs_round_multisecret_summary.csv`
- `archive/prove vecchie/results_old/baseline/attack_vs_round_multisecret_summary_by_secret.csv`
- `archive/prove vecchie/results_old/baseline/attack_vs_round_multisecret_summary_robust.csv`
- `archive/prove vecchie/results_old/baseline/attack_vs_round_multisecret_summary_robust_by_secret.csv`
- `archive/prove vecchie/results_old/baseline/attack_vs_round_raw.csv`
- `archive/prove vecchie/results_old/baseline/attack_vs_round_summary.csv`
- `archive/prove vecchie/results_old/baseline/metrics.csv`
- `archive/prove vecchie/results_old/inversion_versions/version1_single_round_restarts.md`
- `archive/prove vecchie/results_old/inversion_versions/version2_multi_round_snapshot.md`
- `archive/prove vecchie/results_old/inversion_versions/version3_direct_batch_dlg.md`

#### Radiography archive note
- The historical `archive/prove vecchie/COVID-19_Radiography_Dataset/` snapshot has been removed to reduce repository size.

## 5) File Inventory Totals (excluding `.git/` and `.venv/` internals)
- Total files in repository workspace: `311`
- Non-PNG files: `116`
- PNG files: `195`

## 6) Practical Guidance for Future Agents
- Treat `src/` + `notebooks/` + `results/` as the canonical active pipeline.
- Treat `archive/` as historical reference unless explicitly needed for replication.
- Be aware there are duplicated MNIST raw caches in:
  - `data/MNIST/raw/`
  - `notebooks/data/MNIST/raw/`
  - `archive/prove vecchie/raw/`
- If extending experiments, prefer adding new scripts/modules under `src/` and keep notebooks thin wrappers.
