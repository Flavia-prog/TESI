
set -euo pipefail


VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
KERNEL_NAME="fl_thesis"
KERNEL_DISPLAY_NAME="Python (fl_thesis)"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN is not installed or not in PATH."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-baselin.txt


python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo ""
echo "Environment ready."
echo "In VS Code/Jupyter, select kernel: $KERNEL_DISPLAY_NAME"
