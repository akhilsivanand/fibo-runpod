#!/usr/bin/env python3
"""Validate public Fibo source and optional exact-image GPU release evidence.

The default check is source-only. --publish additionally requires a dated
report with real Pod and fresh queue-worker edits, authentication and restart
preservation tied to the published image digest. Report structure is checked;
the maintainer must supply observed facts rather than synthetic evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tomllib
from urllib.parse import urlsplit


PRESETS = {"edit"}
REQUIRED = (
    "Dockerfile", "handler.py", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.txt",
    ".dockerignore", ".runpod/hub.json", ".runpod/tests.json", ".runpod/README.md",
    "deploy/pod-template.json", "deploy/pod-console.json", "docs/release.md", "config/models.json",
    "config/workflows.json", "requirements/runtime.lock", "requirements/upgrade.lock",
    "src/fibo_runtime/bootstrap.py", "src/fibo_runtime/jobs.py",
    "custom_nodes/fibo/__init__.py", "scripts/entrypoint.sh",
    "src/fibo_runtime/filebrowser.py", "scripts/install_filebrowser.sh", "scripts/check_service_ports.py",
    "src/fibo_runtime/jupyter.py", "scripts/install_jupyter.sh", "requirements/jupyter.lock",
)
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
MODEL_INPUTS = {"unet_name", "clip_name", "vae_name", "lora_name"}
MODEL_NODES = {"UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly"}
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".yml", ".yaml", ".sh", ".toml", ".lock", ".in"}
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
SENSITIVE = (
    (re.compile(r"/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\"), "private home path"),
    (re.compile(r"https?://[a-z0-9]{8,}-\d+\.proxy\.runpod\.net", re.I), "deployment-specific Pod URL"),
    (re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"), "private key"),
    (re.compile(r"\b(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|rpa_[A-Za-z0-9]{24,})\b"), "credential-like token"),
    (re.compile(r'["\'](?:podId|pod_id)["\']\s*:\s*["\'][a-z0-9]{8,}["\']', re.I), "deployment-specific Pod ID"),
)
IDENTITY_PLACEHOLDER = re.compile(r"(?:ghcr\.io|github\.com)/(?:OWNER|YOUR_[A-Z_]+)(?:/|:)")


class ReleaseError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReleaseError(message)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ReleaseError(f"Cannot read JSON {path.name}: {error}") from error


def package_path(root, relative):
    require(isinstance(relative, str), "Package path must be a string")
    posix = PurePosixPath(relative)
    require(not posix.is_absolute() and ".." not in posix.parts and "\\" not in relative,
            "Package path must remain relative to the release root")
    result = (root / relative).resolve()
    require(result.is_relative_to(root.resolve()) and result.is_file(),
            f"Required package file is missing or escapes root: {relative}")
    return result


def text_files(root):
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_PARTS for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        require(not path.is_symlink(), f"Public package must not contain file symlinks: {path.relative_to(root)}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"Dockerfile", "LICENSE", ".dockerignore", ".gitignore"}:
            try:
                yield path, path.read_text()
            except UnicodeError as error:
                raise ReleaseError(f"Public text file is not UTF-8: {path.relative_to(root)}") from error


def check_manifest(root):
    manifest = read_json(root / "config/models.json")
    models = manifest.get("models")
    require(manifest.get("schema_version") == 1 and isinstance(models, list) and len(models) == 3,
            "Model manifest must contain exactly three active assets")
    names, paths, total = set(), set(), 0
    for model in models:
        require(isinstance(model, dict), "Model entry must be an object")
        relative = model.get("path", "")
        path = PurePosixPath(relative)
        require(isinstance(relative, str) and not path.is_absolute() and ".." not in path.parts
                and "\\" not in relative and path.suffix == ".safetensors", "Invalid model path")
        require(relative not in paths and path.name not in names, "Duplicate model destination")
        paths.add(relative)
        names.add(path.name)
        size = model.get("size")
        require(type(size) is int and size > 0, f"Invalid model size: {relative}")
        require(isinstance(model.get("sha256"), str) and HEX64.fullmatch(model["sha256"]),
                f"Invalid model SHA256: {relative}")
        source = model.get("source", {})
        revision, repository = source.get("revision", ""), source.get("repository", "")
        require(isinstance(revision, str) and HEX40.fullmatch(revision), f"Unpinned model revision: {relative}")
        require(isinstance(repository, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository),
                f"Invalid model repository: {relative}")
        url = urlsplit(model.get("url", ""))
        require(url.scheme == "https" and url.netloc == "huggingface.co" and not url.query and not url.fragment
                and url.path == f"/{repository}/resolve/{revision}/{source.get('path')}",
                f"Model URL must use its exact pinned public Hugging Face revision: {relative}")
        require(source.get("path") == relative, f"Source/destination model path mismatch: {relative}")
        expected_license = "CC-BY-NC-4.0" if relative.startswith("diffusion_models/") else "Apache-2.0"
        require(model.get("license") == expected_license, f"Missing reviewed model license: {relative}")
        total += size
    require(total == manifest.get("total_size_bytes") == 17620493150, "Model download total is inconsistent")
    return names


def walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def check_workflows(root, model_names):
    catalog = read_json(root / "config/workflows.json")
    presets = catalog.get("presets", {})
    require(catalog.get("version") == 1 and set(presets) == PRESETS, "Workflow catalog must expose the supported edit API preset")
    require(catalog.get("default_preset") in PRESETS, "Unknown default preset")
    for name, entry in list(presets.items()) + list(catalog.get("ui_only", {}).items()):
        ui = read_json(package_path(root, entry["ui"]))
        require(isinstance(ui.get("nodes"), list) and isinstance(ui.get("links"), list), f"Invalid browser graph: {name}")
        for node in walk(ui):
            if isinstance(node, dict) and node.get("type") in MODEL_NODES:
                widgets = node.get("widgets_values", [])
                require(isinstance(widgets, list) and widgets and widgets[0] in model_names,
                        f"Browser graph references an unbundled model: {name}")
        if "api" not in entry:
            continue
        exported = read_json(package_path(root, entry["api"]))
        graph = exported.get("prompt")
        require(isinstance(graph, dict) and graph, f"Invalid API graph: {name}")
        require(any(node.get("class_type") == "SaveImage" for node in graph.values()), f"API graph has no image output: {name}")
        for key in entry.get("nodes", {}).values():
            require(key in graph, f"Catalog points to missing API node: {name}")
        for node in graph.values():
            require(isinstance(node, dict) and isinstance(node.get("inputs"), dict), f"Malformed API node: {name}")
            for key, value in node["inputs"].items():
                if key in MODEL_INPUTS:
                    require(value in model_names, f"API graph references an unbundled model: {name}: {value}")
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int:
                    require(value[0] in graph, f"API graph has a dangling node link: {name}")
    return catalog


def check_deployment(root):
    hub = read_json(root / ".runpod/hub.json")
    require(all(isinstance(hub.get(k), str) and hub[k] for k in ("title", "description")), "Hub metadata is incomplete")
    require(hub.get("type") == "serverless" and hub.get("category") == "image", "Hub must describe an image Serverless API")
    config = hub.get("config", {})
    require(config.get("runsOn") == "GPU" and config.get("gpuCount") == 1 and config.get("gpuIds") == "ADA_24"
            and config.get("containerDiskInGb") == 60 and config.get("allowedCudaVersions") == ["13.0", "13.2"], "Hub hardware settings are inconsistent")
    require("endpointType" not in config, "This release is queue-based, not load-balanced")
    require(any(e.get("key") == "FIBO_MODE" and e.get("value") == "serverless" for e in config.get("env", [])), "Hub must select Serverless mode")
    tests = read_json(root / ".runpod/tests.json")
    require(isinstance(tests.get("tests"), list) and tests["tests"], "Hub generation tests are required")
    for case in tests["tests"]:
        require(isinstance(case.get("name"), str) and isinstance(case.get("input"), dict), "Invalid Hub test case")
        require(case["input"].get("preset") in PRESETS and bool(case["input"].get("prompt", "").strip())
                and bool(case["input"].get("image_base64")), "Hub test must submit an edit prompt and source image")
        require(type(case.get("timeout")) is int and case["timeout"] >= 600000, "Hub cold-test timeout is too short")
    require(tests.get("config", {}).get("gpuCount") == 1 and tests["config"].get("allowedCudaVersions") == ["13.0", "13.2"], "Hub test hardware settings are inconsistent")
    pod = read_json(root / "deploy/pod-template.json")
    require(pod.get("category") == "NVIDIA" and pod.get("isServerless") is False, "Pod template must select NVIDIA Pod mode")
    require(pod.get("containerDiskInGb") == 60 and pod.get("volumeInGb") == 60 and pod.get("volumeMountPath") == "/workspace", "Pod needs 60 GB first-use disk and the optional workspace volume setting")
    require(set(pod.get("ports", [])) == {"8188/http", "8080/http", "8888/http", "22/tcp"}, "Pod must expose standard ComfyUI, FileBrowser, JupyterLab and SSH ports")
    require(pod.get("env") == {"FIBO_MODE": "pod"}, "Pod launch must not require a manual password, token or volume-size variable")
    require(pod.get("dockerEntrypoint") == [] and pod.get("dockerStartCmd") == ["/opt/fibo/scripts/entrypoint.sh"], "Pod must explicitly select the baked startup script and retain the image entry point")
    require(isinstance(pod.get("imageName"), str) and pod["imageName"], "Pod image identity is missing")
    require(type(pod.get("isPublic")) is bool, "Pod visibility must be explicit")
    console = read_json(root / "deploy/pod-console.json")
    require(console.get("startJupyter") is True and console.get("startSsh") is True,
            "Native Connect token and Runpod SSH must be enabled")
    require(console.get("allowedCudaVersions") == ["13.0", "13.2"], "Pod host must support CUDA 13.0")
    require(console.get("portsConfig") == [{"name": "ComfyUI", "port": 8188},
            {"name": "JupyterLab", "port": 8888}, {"name": "FileBrowser", "port": 8080}],
            "Connect labels must match the standard service ports")
    return pod


def check_gpu_report(root, pod):
    report = read_json(root / "docs/validation.json")
    require(report.get("schema_version") == 1, "Unsupported GPU validation report schema")
    require(isinstance(report.get("source_revision"), str) and HEX40.fullmatch(report["source_revision"]), "GPU report needs the tested source commit")
    digest = report.get("image_digest", "")
    require(isinstance(digest, str) and digest.startswith("sha256:") and HEX64.fullmatch(digest[7:]), "GPU report needs the tested image digest")
    require("@sha256:" in pod["imageName"], "Publication requires a Pod image pinned to the GPU-tested immutable digest")
    require(pod["imageName"].split("@", 1)[1] == digest, "Pod image differs from the GPU-tested digest")
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    require(report.get("package_version") == version, "GPU report package version differs from this release")
    try:
        timestamp = datetime.fromisoformat(report["validated_at"].replace("Z", "+00:00"))
        require(timestamp.tzinfo is not None, "GPU validation timestamp must include a timezone")
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseError("GPU report needs an ISO 8601 validation timestamp") from error
    gpu = report.get("gpu", {})
    require(isinstance(gpu.get("name"), str) and bool(gpu["name"].strip()) and isinstance(gpu.get("driver"), str)
            and bool(gpu["driver"].strip()) and type(gpu.get("vram_gb")) in (int, float)
            and gpu["vram_gb"] >= 24 and gpu.get("cuda_runtime") == "13.0", "GPU report needs actual 24 GB GPU, driver, and CUDA 13.0 details")
    for mode, flags in {"pod": ("fresh_volume", "auth_verified", "restart_preserved"), "serverless": ("fresh_worker",)}.items():
        evidence = report.get(mode, {})
        require(all(evidence.get(flag) is True for flag in flags), f"GPU report lacks required {mode} checks")
        jobs = evidence.get("jobs")
        require(isinstance(jobs, list) and {job.get("preset") for job in jobs if isinstance(job, dict)} == PRESETS,
                f"GPU report must include a completed edit for {mode}")
        for job in jobs:
            require(job.get("completed") is True and type(job.get("width")) is int and type(job.get("height")) is int
                    and job["width"] > 0 and job["height"] > 0 and isinstance(job.get("image_sha256"), str)
                    and HEX64.fullmatch(job["image_sha256"]), f"GPU report needs completed image evidence for {mode}")


def check_release(root, publish=False):
    root = Path(root).resolve()
    for relative in REQUIRED:
        package_path(root, relative)
    names = check_manifest(root)
    check_workflows(root, names)
    pod = check_deployment(root)
    placeholders = []
    for path, content in text_files(root):
        relative = path.relative_to(root)
        for pattern, label in SENSITIVE:
            require(not pattern.search(content), f"Public file contains a {label}: {relative}")
        if IDENTITY_PLACEHOLDER.search(content):
            placeholders.append(str(relative))
    if publish:
        require(not placeholders, "Publishing requires concrete repository/registry identities; unresolved placeholders in " + ", ".join(placeholders))
        check_gpu_report(root, pod)
    return {"source": "passed", "publish": "passed" if publish else "not_checked",
            "identity_pending": bool(placeholders), "gpu_validation": "report_checked" if publish else "not_checked"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--publish", action="store_true", help="Require resolved identities and real GPU validation evidence")
    args = parser.parse_args()
    try:
        print(json.dumps(check_release(args.root, args.publish), indent=2))
    except (ReleaseError, KeyError, TypeError, OSError) as error:
        print(f"Release check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
