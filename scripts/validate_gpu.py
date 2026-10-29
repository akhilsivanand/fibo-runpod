#!/usr/bin/env python3
"""Run one new Fibo Edit 2 smoke job using an explicitly selected source image.

Without --run this only validates the package catalog and prints the plan.
It does not query a GPU, contact ComfyUI, inspect history, or read saved images.
The resulting record is execution evidence, not a complete release certificate
or an assessment of image quality.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image, UnidentifiedImageError

# Also support running this file directly from an unpacked source checkout.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))
from fibo_runtime import __version__
from fibo_runtime.jobs import (ComfyJobRunner, GenerationRequest, MAX_OUTPUT_BYTES,
                               load_catalog, validate_request)


EDIT_PROMPT = "Change the color of the main object to red. Keep the composition and background unchanged."
FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_details() -> dict:
    """Inspect hardware only after the operator explicitly supplies --run."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU must be available before running validation")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 23 * 1024**3:
        raise RuntimeError("Validation requires at least 23 GiB of available device VRAM")
    response = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,driver_version,name,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True,
    )
    devices = []
    for row in csv.reader(response.stdout.splitlines()):
        if len(row) != 4:
            raise RuntimeError("nvidia-smi returned unexpected hardware metadata")
        index, driver, name, memory = (value.strip() for value in row)
        devices.append({"index": int(index), "driver_version": driver,
                        "name": name, "memory_total_mib": int(memory)})
    if not devices:
        raise RuntimeError("nvidia-smi did not return driver metadata")
    return {
        "name": properties.name,
        "vram_bytes": properties.total_memory,
        "vram_gib": round(properties.total_memory / 1024**3, 3),
        "compute_capability": [properties.major, properties.minor],
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "nvidia_smi_devices": devices,
    }


def package_details(root: Path) -> dict:
    details = {
        "version": __version__,
        "workflow_catalog_sha256": hashlib.sha256((root / "config/workflows.json").read_bytes()).hexdigest(),
    }
    commit_file = root / "comfy-commit.txt"
    if commit_file.is_file():
        details["baked_comfy_commit"] = commit_file.read_text().strip()
    provided = {key: os.environ[key] for key in (
        "FIBO_SOURCE_REVISION", "FIBO_IMAGE_DIGEST", "FIBO_IMAGE_REFERENCE",
    ) if os.environ.get(key)}
    if provided:
        # Environment declarations are useful provenance but do not independently
        # prove the running OCI image's identity.
        details["operator_provided_environment"] = provided
    return details


def save_result(result: dict, request: GenerationRequest, destination: Path,
                elapsed_seconds: float) -> dict:
    """Verify and save only the image returned by this newly submitted job."""
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("Elapsed time must be finite and nonnegative")
    if not isinstance(result, dict) or result.get("preset") != request.preset or result.get("seed") != request.seed:
        raise RuntimeError("The worker returned an unexpected preset or seed")
    images = result.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise RuntimeError("The worker must return exactly one image")
    item = images[0]
    encoded = item.get("data")
    if item.get("type") != "base64" or not isinstance(encoded, str):
        raise RuntimeError("The worker did not return a base64 image")
    if len(encoded) > 4 * ((MAX_OUTPUT_BYTES + 2) // 3):
        raise RuntimeError("Returned image exceeds the output-size limit")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RuntimeError("The worker returned invalid image encoding") from error
    if not content or len(content) > MAX_OUTPUT_BYTES:
        raise RuntimeError("Returned image is empty or too large")
    try:
        with Image.open(io.BytesIO(content)) as image:
            dimensions, image_format = image.size, image.format
            if image_format not in FORMATS or dimensions != (request.width, request.height):
                raise RuntimeError("The worker returned an unexpected image format or dimensions")
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise RuntimeError("The worker returned an invalid image") from error
    output = destination.with_suffix(FORMATS[image_format])
    with output.open("xb") as target:
        target.write(content)
    output.chmod(0o600)
    return {
        "preset": request.preset,
        "completed": True,
        "seed": request.seed,
        "width": dimensions[0],
        "height": dimensions[1],
        "steps": request.steps,
        "cfg": request.cfg,
        "client_elapsed_seconds": round(elapsed_seconds, 3),
        "image_sha256": hashlib.sha256(content).hexdigest(),
        "image_bytes": len(content),
        "image_format": image_format,
        "file": output.name,
    }


def _write_record(path: Path, record: dict) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(record, target, indent=2)
        target.write("\n")
    temporary.chmod(0o600)
    temporary.replace(path)


async def validate(output_dir: Path, root: Path, data_dir: Path, input_path: Path, prompt: str) -> dict:
    catalog = load_catalog(root)
    hardware = gpu_details()
    # Refuse an existing path rather than inspecting or overwriting its contents.
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    record_path = output_dir / "record.json"
    record = {
        "schema_version": 1,
        "evidence_kind": "new_preset_job_smoke",
        "status": "running",
        "started_at": _utc_now(),
        "package": package_details(root),
        "gpu": hardware,
        "jobs": [],
        "limits": [
            "Checks execution and returned image structure, not aesthetic quality.",
            "Does not verify Pod freshness, restart persistence, authentication, SDK transport, or Hub acceptance.",
            "Elapsed times are client wall time, not isolated GPU or server execution time.",
        ],
    }
    _write_record(record_path, record)
    runner = ComfyJobRunner(root=root, data_dir=data_dir)
    try:
        payload = {"preset": "edit", "prompt": prompt, "seed": 42,
                   "image_base64": base64.b64encode(input_path.read_bytes()).decode("ascii")}
        request = validate_request(payload, catalog)
        record["input_sha256"] = hashlib.sha256(request.image_bytes).hexdigest()
        started = time.perf_counter()
        result = await runner.run(payload)
        elapsed = time.perf_counter() - started
        record["jobs"].append(save_result(result, request, output_dir / "01-edit", elapsed))
        record.update(status="passed", finished_at=_utc_now())
        _write_record(record_path, record)
        return record
    except BaseException as error:
        record.update(status="failed", finished_at=_utc_now(), error_type=type(error).__name__)
        _write_record(record_path, record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New directory for the new edited image and record.json")
    parser.add_argument("--run", action="store_true",
                        help="Submit one editing job to the configured ComfyUI service")
    parser.add_argument("--input", type=Path, help="Task-selected source PNG/JPEG/WebP (required with --run)")
    parser.add_argument("--prompt", default=EDIT_PROMPT, help="Instruction for the new editing job")
    args = parser.parse_args()
    root = Path(os.environ.get("FIBO_ROOT", SOURCE_ROOT)).resolve()
    data_dir = Path(os.environ.get("FIBO_DATA_DIR", "/workspace/ComfyUI")).resolve()
    load_catalog(root)
    if not args.run:
        print(json.dumps({
            "status": "dry_run", "gpu_jobs_submitted": 0,
            "plan": [{"preset": "edit", "resolution": "catalog defaults", "steps": 4,
                      "cfg": 1, "seed": 42, "input": "explicit --input image"}],
            "requires": "--run, --input and a new --output-dir to execute",
        }, indent=2))
        return
    if args.input is None:
        parser.error("--input is required with --run")
    record = asyncio.run(validate(args.output_dir.expanduser().absolute(), root, data_dir,
                                  args.input.expanduser().absolute(), args.prompt))
    print(json.dumps({"status": record["status"], "completed_new_jobs": len(record["jobs"]),
                      "record": str(args.output_dir / "record.json")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print("New-job GPU validation failed: " + str(error), file=sys.stderr)
        sys.exit(1)
