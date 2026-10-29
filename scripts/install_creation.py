#!/usr/bin/env python3
"""Add the optional Dev creation model and saved workflow to an existing Pod."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fibo_runtime.assets import prepare_models
from fibo_runtime.workflows import atomic_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/ComfyUI"))
    args = parser.parse_args()
    data = args.data_root.resolve()
    # Creation shares the encoder and VAE from the working edit installation.
    base = json.loads((ROOT / "config/models.json").read_text())
    for model in base["models"]:
        if model["path"].startswith(("text_encoders/", "vae/")):
            path = data / "models" / model["path"]
            if not path.is_file() or path.stat().st_size != model["size"]:
                raise SystemExit("Prepare the base edit installation first: " + model["path"])
    result = prepare_models(ROOT / "config/creation-models.json", data)
    source = ROOT / "workflows/ui/fibo2-create-dev.json"
    content = source.read_bytes()
    json.loads(content)
    target = data / "user/default/workflows/FIBO" / source.name
    if target.is_symlink() or not target.parent.resolve().is_relative_to(data):
        raise SystemExit("Preserving an existing workflow path outside the data directory")
    if target.exists() and (not target.is_file() or target.read_bytes() != content):
        raise SystemExit("Model is ready; preserving the user's existing creation workflow")
    if not target.exists():
        atomic_write(target, content)
    print(json.dumps({"models": result, "workflow": str(target),
                      "workflow_sha256": hashlib.sha256(content).hexdigest(),
                      "restart_performed": False}))


if __name__ == "__main__":
    main()
