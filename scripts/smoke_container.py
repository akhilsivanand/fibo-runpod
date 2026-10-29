#!/usr/bin/env python3
"""Check the baked ComfyUI, native FIBO catalog, and API schemas on the CPU.

Run inside the built image with its Python environment. No models are fetched,
no prompt is submitted, and no existing workspace, log, or output is inspected.
"""

import argparse
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import ProxyHandler, Request, build_opener


class SmokeError(RuntimeError):
    pass


FILE_SELECTORS = {"unet_name", "clip_name", "vae_name", "lora_name", "image"}
FRONTEND_NODES = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode"}
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def package_file(root, relative):
    if not isinstance(relative, str):
        raise SmokeError("Invalid workflow path in the package catalog")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise SmokeError("A catalog workflow is missing or outside the package")
    return path


def load_catalog(root):
    config = read_json(root / "config/workflows.json")
    presets, ui_only = config.get("presets", {}), config.get("ui_only", {})
    if not isinstance(presets, dict) or not isinstance(ui_only, dict):
        raise SmokeError("Invalid workflow catalog")
    entries = {**presets, **ui_only}
    if set(presets) != {"edit"} or ui_only:
        raise SmokeError("Expected one Fibo Edit 2 API preset and native template")
    names = set()
    for identifier, entry in entries.items():
        if not isinstance(entry, dict):
            raise SmokeError("Invalid workflow catalog entry")
        ui_path = package_file(root, entry.get("ui"))
        if ui_path.suffix != ".json" or ui_path.name in names:
            raise SmokeError("Template filenames must be unique JSON files")
        names.add(ui_path.name)
        if identifier in presets:
            package_file(root, entry.get("api"))
    return config, entries


def _node_inputs(definition, supplied):
    result, required = {}, set()
    for section in ("required", "optional"):
        for name, descriptor in definition.get("input", {}).get(section, {}).items():
            if isinstance(descriptor, (list, tuple)) and descriptor and descriptor[0] == "COMFY_AUTOGROW_V3":
                try:
                    template = descriptor[1]["template"]
                    names = template["names"]
                    children = template["input"]["required"]
                    minimum = template.get("min", 0)
                    if (not isinstance(names, list) or not names or len(children) != 1
                            or type(minimum) is not int or not 0 <= minimum <= len(names)
                            or any(not isinstance(child, str) or not child for child in names)):
                        raise ValueError()
                    child_descriptor = next(iter(children.values()))
                except (KeyError, IndexError, TypeError, AttributeError, ValueError) as error:
                    raise SmokeError(f"Invalid dynamic input schema for {name}") from error
                expanded = {name + "." + child: child_descriptor for child in names}
                present = set(expanded).intersection(supplied)
                if len(present) < minimum:
                    raise SmokeError(f"Dynamic input {name} needs at least {minimum} linked values")
                result.update(expanded)
            else:
                result[name] = descriptor
                if section == "required":
                    required.add(name)
    return result, required


def validate_api_graph(document, definitions, label="API graph"):
    graph = document.get("prompt", document) if isinstance(document, dict) else None
    if not isinstance(graph, dict) or not graph:
        raise SmokeError(f"{label}: empty API graph")
    classes = set()
    for identifier, node in graph.items():
        if not isinstance(node, dict) or node.get("class_type") not in definitions:
            raise SmokeError(f"{label}: node {identifier} is not available in baked ComfyUI")
        classes.add(node["class_type"])
    for identifier, node in graph.items():
        definition = definitions[node["class_type"]]
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            raise SmokeError(f"{label}: node {identifier} has invalid inputs")
        expected, required = _node_inputs(definition, inputs)
        missing = required - inputs.keys()
        if missing:
            raise SmokeError(f"{label}: node {identifier} is missing required input {sorted(missing)[0]}")
        for name, value in inputs.items():
            if name not in expected:
                raise SmokeError(f"{label}: node {identifier} has unsupported input {name}")
            descriptor = expected[name]
            if not isinstance(descriptor, (list, tuple)) or not descriptor:
                raise SmokeError(f"{label}: node {identifier} has an invalid input schema")
            kind = descriptor[0]
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int:
                source, index = value
                if source not in graph:
                    raise SmokeError(f"{label}: node {identifier} links to a missing node")
                outputs = definitions[graph[source]["class_type"]].get("output", [])
                if index < 0 or index >= len(outputs):
                    raise SmokeError(f"{label}: node {identifier} links to a missing output")
                if isinstance(kind, str) and kind != "*" and outputs[index] not in (kind, "*"):
                    raise SmokeError(f"{label}: node {identifier} has an incompatible link for {name}")
                continue
            if kind == "COMBO":
                # V3 nodes expose dropdowns as COMBO plus metadata options;
                # legacy nodes put the options list directly in descriptor[0].
                metadata = descriptor[1] if len(descriptor) > 1 else None
                if not isinstance(metadata, dict) or not isinstance(metadata.get("options"), list):
                    raise SmokeError(f"{label}: node {identifier} has an invalid COMBO options schema for {name}")
                kind = metadata["options"]
            if isinstance(kind, list):
                if name in FILE_SELECTORS:
                    # Models and input images are deliberately absent in this
                    # CPU check. Their selectors are still required to exist.
                    if not isinstance(value, str):
                        raise SmokeError(f"{label}: node {identifier} has invalid file selector {name}")
                elif value not in kind:
                    raise SmokeError(f"{label}: node {identifier} has an unsupported option for {name}")
                continue
            valid = {
                "INT": lambda item: type(item) is int,
                "FLOAT": lambda item: type(item) in (int, float) and math.isfinite(item),
                "STRING": lambda item: isinstance(item, str),
                "BOOLEAN": lambda item: type(item) is bool,
            }
            if kind in valid:
                if not valid[kind](value):
                    raise SmokeError(f"{label}: node {identifier} has an invalid value type for {name}")
                options = descriptor[1] if len(descriptor) > 1 and isinstance(descriptor[1], dict) else {}
                if kind in ("INT", "FLOAT") and (
                    ("min" in options and value < options["min"]) or
                    ("max" in options and value > options["max"])
                ):
                    raise SmokeError(f"{label}: node {identifier} has an out-of-range value for {name}")
            else:
                raise SmokeError(f"{label}: node {identifier} needs a link for {name}")
    return classes


def validate_ui_graph(graph, definitions, label="UI graph", require_subgraph=False):
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not graph["nodes"]:
        raise SmokeError(f"{label}: invalid UI workflow")
    subgraphs = graph.get("definitions", {}).get("subgraphs", [])
    if not isinstance(subgraphs, list) or any(not isinstance(item, dict) for item in subgraphs):
        raise SmokeError(f"{label}: invalid subgraph definitions")
    ids = [item.get("id") for item in subgraphs]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids) or len(set(ids)) != len(ids):
        raise SmokeError(f"{label}: invalid subgraph identifiers")
    subgraph_ids = set(ids)
    root_types = {node.get("type") for node in graph["nodes"] if isinstance(node, dict)}
    if require_subgraph and not root_types.intersection(subgraph_ids):
        raise SmokeError(f"{label}: official nested subgraph is missing")
    classes = set()
    for part in [graph, *subgraphs]:
        nodes = part.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise SmokeError(f"{label}: empty graph or subgraph")
        identifiers = []
        for node in nodes:
            if not isinstance(node, dict):
                raise SmokeError(f"{label}: invalid UI node")
            identifiers.append(node.get("id"))
            kind = node.get("type")
            if kind in subgraph_ids or kind in FRONTEND_NODES:
                continue
            if kind not in definitions:
                raise SmokeError(f"{label}: UI node class is unavailable: {kind}")
            classes.add(kind)
        if len(set(identifiers)) != len(identifiers):
            raise SmokeError(f"{label}: duplicate UI node identifiers")
    return classes


def _fetch(opener, url, method="GET"):
    with opener.open(Request(url, method=method), timeout=5) as response:
        if response.status != 200:
            raise SmokeError("An expected ComfyUI resource did not return HTTP 200")
        body = response.read(MAX_RESPONSE_BYTES + 1) if method == "GET" else b""
        if len(body) > MAX_RESPONSE_BYTES:
            raise SmokeError("A ComfyUI metadata response exceeded the smoke-check limit")
        return response.headers, body


def _fetch_json(opener, url):
    _, body = _fetch(opener, url)
    try:
        return json.loads(body)
    except (ValueError, UnicodeError) as error:
        raise SmokeError("A ComfyUI metadata endpoint did not return JSON") from error


def _wait_ready(process, opener, base_url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SmokeError(f"Baked ComfyUI exited during CPU startup (code {process.returncode})")
        try:
            definitions = _fetch_json(opener, base_url + "/object_info")
            catalog = _fetch_json(opener, base_url + "/api/workflow_templates")
            if isinstance(definitions, dict) and "KSampler" in definitions and isinstance(catalog, dict):
                return definitions, catalog
        except (HTTPError, URLError, TimeoutError, SmokeError):
            pass
        time.sleep(0.25)
    raise SmokeError(f"Baked ComfyUI metadata was not ready within {timeout:g} seconds")


class _UIParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.in_title = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "script" and attributes.get("src"):
            self.scripts.append(attributes["src"])
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data


def _check_ui(opener, base_url):
    headers, body = _fetch(opener, base_url + "/")
    if "text/html" not in headers.get("Content-Type", "").lower():
        raise SmokeError("The native ComfyUI page is not served as HTML")
    parser = _UIParser()
    parser.feed(body.decode("utf-8"))
    if "comfy" not in parser.title.lower() or not parser.scripts:
        raise SmokeError("The native ComfyUI page or its script references are missing")
    local_scripts = [urljoin(base_url + "/", source) for source in parser.scripts]
    local_scripts = [url for url in local_scripts if urlsplit(url).netloc == urlsplit(base_url).netloc]
    if not local_scripts:
        raise SmokeError("The native ComfyUI page has no local JavaScript entrypoint")
    headers, _ = _fetch(opener, local_scripts[0], method="HEAD")
    if "javascript" not in headers.get("Content-Type", "").lower():
        raise SmokeError("The native ComfyUI JavaScript entrypoint is not being served")


def _stop_child(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_smoke(package_root, comfy_root, timeout=60, port=0):
    package_root, comfy_root = Path(package_root).resolve(), Path(comfy_root).resolve()
    if not math.isfinite(timeout) or timeout <= 0:
        raise SmokeError("Startup timeout must be positive and finite")
    if type(port) is not int or not 0 <= port <= 65535:
        raise SmokeError("Port must be zero or an integer from 1 through 65535")
    if not (comfy_root / "main.py").is_file():
        raise SmokeError("The baked ComfyUI main.py is missing")
    config, entries = load_catalog(package_root)
    provider = config.get("template_source", "fibo")
    if provider != "fibo":
        raise SmokeError("Expected the native fibo template provider")
    if not port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as available:
            available.bind(("127.0.0.1", 0))
            port = available.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    # Explicitly bypass inherited HTTP proxies for this private child process.
    opener = build_opener(ProxyHandler({}))
    process = None
    with tempfile.TemporaryDirectory(prefix="fibo-container-smoke-") as temporary:
        data = Path(temporary)
        for name in ("models", "input", "output", "temp", "user", "custom_nodes", "cache"):
            (data / name).mkdir()
        extra = data / "comfy-paths.yaml"
        extra.write_text("fibo_bundled:\n  custom_nodes: " + json.dumps(str(comfy_root / "custom_nodes")) + "\n")
        environment = {
            key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH") if key in os.environ
        }
        environment.update(
            CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
            PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
            HF_HOME=str(data / "cache/huggingface"), XDG_CACHE_HOME=str(data / "cache"),
            TORCH_HOME=str(data / "cache/torch"), PYTHONPYCACHEPREFIX=str(data / "cache/pycache"),
            TMPDIR=str(data / "temp"), FIBO_ROOT=str(package_root), FIBO_DATA_DIR=str(data),
        )
        command = [
            sys.executable, "-u", str(comfy_root / "main.py"), "--cpu", "--offline",
            "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch",
            "--base-directory", str(data), "--user-directory", str(data / "user"),
            "--input-directory", str(data / "input"), "--output-directory", str(data / "output"),
            "--temp-directory", str(data / "temp"), "--extra-model-paths-config", str(extra),
            "--disable-all-custom-nodes", "--whitelist-custom-nodes", "fibo",
        ]
        try:
            process = subprocess.Popen(
                command, cwd=comfy_root, env=environment, start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            definitions, templates = _wait_ready(process, opener, base_url, timeout)
            advertised = templates.get(provider)
            if not isinstance(advertised, list) or any(not isinstance(name, str) for name in advertised):
                raise SmokeError("The native fibo template provider is missing")
            expected_names = {Path(entry["ui"]).stem for entry in entries.values()}
            if {Path(name).stem for name in advertised} != expected_names or len(advertised) != len(entries):
                raise SmokeError("The native fibo template provider must advertise exactly the bundled editing workflow")
            api_count, node_classes = 0, set()
            for identifier, entry in entries.items():
                filename = package_file(package_root, entry["ui"]).name
                served = _fetch_json(opener, base_url + "/api/workflow_templates/fibo/" + quote(filename))
                bundled = read_json(package_file(package_root, entry["ui"]))
                if served != bundled:
                    raise SmokeError("A native template differs from its bundled workflow")
                node_classes.update(validate_ui_graph(served, definitions, identifier, require_subgraph=False))
                if entry.get("api"):
                    graph = read_json(package_file(package_root, entry["api"]))
                    node_classes.update(validate_api_graph(graph, definitions, identifier))
                    api_count += 1
            _check_ui(opener, base_url)
            if process.poll() is not None:
                raise SmokeError("Baked ComfyUI exited before the checks completed")
            return {
                "status": "passed", "device": "cpu", "templates": len(entries),
                "api_graphs": api_count, "node_classes": len(node_classes),
                "native_ui": True, "model_weights": "not_loaded", "inference": "not_run",
            }
        finally:
            _stop_child(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(os.environ.get("FIBO_ROOT", "/opt/fibo")))
    parser.add_argument("--comfy-root", type=Path, default=Path(os.environ.get("COMFY_ROOT", "/opt/comfyui")))
    parser.add_argument("--timeout", type=float, default=60, help="CPU startup deadline in seconds (default: 60)")
    parser.add_argument("--port", type=int, default=0, help="Loopback port, or 0 for an unused port")
    args = parser.parse_args()
    result = run_smoke(args.package_root, args.comfy_root, args.timeout, args.port)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("CPU smoke check interrupted", file=sys.stderr)
        sys.exit(130)
    except (SmokeError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"CPU smoke check failed: {error}", file=sys.stderr)
        sys.exit(1)
