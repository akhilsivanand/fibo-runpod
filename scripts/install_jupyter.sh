#!/usr/bin/env bash
set -Eeuo pipefail

# Keep notebook dependencies separate from the exact inference environment.
# The lock contains the full Python 3.11 Linux dependency set and wheel hashes.
JUPYTER_SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
/usr/local/bin/python3.11 -m venv /opt/jupyter
/opt/jupyter/bin/python -m pip install --no-cache-dir --require-hashes \
  --only-binary=:all: --no-deps --requirement "$JUPYTER_SOURCE_ROOT/requirements/jupyter.lock"
/opt/jupyter/bin/python -m pip check
ln -sfn /opt/jupyter/bin/jupyter /usr/local/bin/jupyter
ln -sfn /opt/jupyter/bin/jupyter-lab /usr/local/bin/jupyter-lab
/opt/jupyter/bin/jupyter lab --version
