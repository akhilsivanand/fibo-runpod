#!/usr/bin/env python3
"""Deterministically archive only public source; validate optional GPU evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile


VERSION = "0.1.0"
ROOT_FILES = {
    "Dockerfile", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.txt",
    ".dockerignore", ".gitignore", "handler.py", "pyproject.toml",
}
DIRECTORIES = {
    ".runpod": {".json", ".md"},
    "config": {".json"},
    "deploy": {".json"},
    "docs": {".md"},
    "requirements": {".txt", ".in", ".lock"},
    "scripts": {".py", ".sh"},
    "tests": {".py"},
    "src": {".py"},
    "custom_nodes": {".py"},
    "workflows": {".json"},
}
EXCLUDED_PARTS = {
    ".git", "__pycache__", ".pytest_cache", ".cache", ".venv", "venv",
    "private", "local", "validation-local", "workflow-validation", "logs",
    "history", "models", "input", "output", "outputs", "dist", "build",
}


def load_checker(root):
    spec = importlib.util.spec_from_file_location("fibo_release_checker", root / "scripts/check_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def public_files(root, include_validation=False):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            continue
        if any(part.lower() in EXCLUDED_PARTS or part.startswith(".env")
               or part.lower().startswith(("private-", "private_", "local-", "local_", "validation-local"))
               or part.lower().rsplit(".", 1)[0] in {"private", "local"}
               or ".local." in part.lower() or ".private." in part.lower()
               for part in relative.parts):
            continue
        if path.name.startswith(".") and relative.as_posix() not in ROOT_FILES:
            continue
        parts = relative.parts
        allowed = relative.as_posix() in ROOT_FILES
        if len(parts) >= 2 and parts[0] in DIRECTORIES:
            allowed = path.suffix in DIRECTORIES[parts[0]]
        if len(parts) == 3 and parts[:2] == (".github", "workflows"):
            allowed = path.suffix in {".yml", ".yaml"}
        if relative.as_posix() == "docs/validation.json":
            allowed = include_validation
        if allowed:
            yield path, relative


def package_release(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    checker = load_checker(root)
    checker.check_release(root)
    if output.is_relative_to(root):
        raise ValueError("Archive output must be outside the source repository")
    if output.exists() and not output.is_file():
        raise ValueError("Archive output must be a file path")
    include_validation = False
    if (root / "docs/validation.json").is_file():
        # Only the intentionally public report can enter the ZIP. Never infer
        # release proof from arbitrary local validation files.
        checker.check_gpu_report(root, checker.read_json(root / "deploy/pod-template.json"))
        include_validation = True
    files = list(public_files(root, include_validation))
    if not files:
        raise ValueError("No public release files were selected")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".fibo-release-", suffix=".zip", dir=output.parent, delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path, relative in files:
                info = zipfile.ZipInfo("fibo-runpod/" + relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                executable = relative.parts[0] == "scripts" and path.suffix in {".sh", ".py"}
                info.external_attr = (0o100755 if executable else 0o100644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes(), compresslevel=9)
        digest = hashlib.sha256()
        with staged.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result = {"archive": str(output), "sha256": digest.hexdigest(),
                  "file_count": len(files), "size_bytes": staged.stat().st_size,
                  "source_validation": "passed"}
        staged.replace(output)
        return result
    finally:
        staged.unlink(missing_ok=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root.parent / "dist" / f"fibo-runpod-{VERSION}.zip")
    args = parser.parse_args()
    try:
        print(json.dumps(package_release(root, args.output), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Packaging failed: {error}\n")


if __name__ == "__main__":
    main()
