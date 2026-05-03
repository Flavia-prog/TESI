# AIJack Environment Setup and First-Run Debugging

## What went wrong on the first attempt

When the experiment was first launched with:

```bash
python -m src.main_fedavg
```

it failed before training because the active interpreter was not the project virtual environment:

- Active Python was `3.13` from Conda/base.
- `torch` and `aijack` were not available in that interpreter.
- `aijack` build/installation is fragile on newer Python/toolchain combinations and can fail with native C++ build errors.

Two recurring issues appeared:

1. Wrong interpreter (`base` Conda instead of project `.venv`).
2. Native build mismatch for `aijack` (Python/CMake/pybind11/Boost toolchain mismatch).


## How it was fixed

The successful path was:

1. Use a dedicated project virtual environment with **Python 3.10**.
2. Install dependencies inside that venv (not globally, not in Conda base).
3. Run the project explicitly with the venv interpreter (`.venv/bin/python`), so shell state cannot silently switch interpreters.


## Reproducible setup (recommended for anyone replicating)

Run from repository root:

```bash
cd /Users/flaviafuscaldi/Desktop/fl_thesis
```

### 1) Create a clean venv with Python 3.10

```bash
/opt/homebrew/bin/python3.10 -m venv .venv
source .venv/bin/activate
python3 --version
```

Expected: `Python 3.10.x`

### 2) Upgrade packaging tools

```bash
python3 -m pip install --upgrade pip setuptools wheel
```

### 3) Install project dependencies

```bash
python3 -m pip install torch torchvision aijack
```

If `aijack` fails to build, install native build tools and retry:

```bash
brew install boost cmake ninja
python3 -m pip install --no-build-isolation --no-cache-dir aijack==0.0.1b2
```

### 4) Verify imports

```bash
python3 - <<'PY'
import torch, torchvision, aijack
print("OK")
PY
```

### 5) Run the experiment

```bash
python3 -m src.main_fedavg
```


## Important pitfalls to avoid

- Do not run from Conda `base` if your project uses `.venv`.
- If prompt shows both `(.venv)` and `(base)`, deactivate Conda until `(base)` disappears.
- Always verify interpreter before running:

```bash
which python3
python3 --version
```

- Most robust command (bypasses shell confusion):

```bash
/Users/flaviafuscaldi/Desktop/fl_thesis/.venv/bin/python3 -m src.main_fedavg
```


## Minimal replication checklist

1. Python `3.10.x`
2. Fresh `.venv` in repo root
3. `pip/setuptools/wheel` upgraded
4. `torch`, `torchvision`, `aijack` installed in that venv
5. `python3 -m src.main_fedavg` runs without import errors


## Output files to confirm successful run

- `results/gradient_inversion/num_clients_2/`
- `results/gradient_inversion/num_clients_5/`
- `results/gradient_inversion/num_clients_summary.csv`
- `results/gradient_inversion/num_clients_vs_median_mse.png`

