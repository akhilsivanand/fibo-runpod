"""Validated preset jobs for the queue-based Runpod worker.

The client owns one WebSocket prompt and unique input/output directories. It
never queries ComfyUI history or accepts arbitrary graphs, URLs, or file paths.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import threading
from typing import Any
from urllib.parse import urlsplit, urlunsplit
import uuid
import warnings

import aiohttp
from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 7 * 1024 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 2_097_152
MAX_INPUT_PIXELS = 16_777_216
ALLOWED_KEYS = {
    "preset", "prompt", "seed", "width", "height", "steps", "cfg",
    "image_base64",
}
IMAGE_FORMATS = {"PNG": ("png", "image/png"), "JPEG": ("jpg", "image/jpeg"),
                 "WEBP": ("webp", "image/webp")}
_WORKER_LOCK = threading.Lock()


@dataclass(frozen=True)
class GenerationRequest:
    preset: str
    prompt: str
    seed: int
    width: int
    height: int
    steps: int
    cfg: float
    image_bytes: bytes | None = None
    image_format: str | None = None


def _root() -> Path:
    return Path(os.environ.get("FIBO_ROOT", Path(__file__).resolve().parents[2]))


def _package_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("Invalid preset configuration path")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("Preset configuration file is unavailable")
    return resolved


def load_catalog(root: Path) -> dict[str, Any]:
    catalog = json.loads((root / "config/workflows.json").read_text())
    if catalog.get("version") != 1 or not isinstance(catalog.get("presets"), dict):
        raise ValueError("Unsupported workflow catalog")
    for name in ("edit",):
        preset = catalog["presets"].get(name)
        if not isinstance(preset, dict):
            raise ValueError(f"Missing preset configuration: {name}")
        _package_file(root, preset["api"])
    return catalog


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number from {minimum} through {maximum}")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a finite number from {minimum} through {maximum}")
    return value


def _image(value: Any) -> tuple[bytes, str]:
    if not isinstance(value, str) or not value:
        raise ValueError("image_base64 must contain a base64-encoded PNG, JPEG, or WebP image")
    if len(value) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("Input image exceeds 7 MiB")
    try:
        content = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("image_base64 is not valid raw base64") from error
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Input image must be nonempty and at most 7 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                image_format = image.format
                if image_format not in IMAGE_FORMATS:
                    raise ValueError("Input image format must be PNG, JPEG, or WebP")
                if (min(image.size) < 1 or max(image.size) > 8192
                        or image.width * image.height > MAX_INPUT_PIXELS):
                    raise ValueError("Input image dimensions exceed 8192 pixels per side or 16 megapixels")
                if getattr(image, "is_animated", False):
                    raise ValueError("Animated input images are unsupported")
                image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError("Input image is invalid or too large") from error
    return content, image_format


def validate_request(payload: dict, catalog: dict) -> GenerationRequest:
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError("input must be an object with string keys")
    unknown = set(payload) - ALLOWED_KEYS
    if unknown:
        raise ValueError("Unsupported input fields: " + ", ".join(sorted(unknown)))
    preset_name = payload.get("preset", catalog.get("default_preset", "edit"))
    if not isinstance(preset_name, str) or preset_name != "edit":
        raise ValueError("preset must be edit")
    preset = catalog["presets"][preset_name]
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12_000:
        raise ValueError("prompt must be a nonempty string of at most 12000 characters")
    defaults = preset["defaults"]
    width = _integer(payload.get("width", defaults["width"]), "width", 64, MAX_PIXELS // 64)
    height = _integer(payload.get("height", defaults["height"]), "height", 64, MAX_PIXELS // 64)
    if width % 64 or height % 64 or width * height > MAX_PIXELS:
        raise ValueError("width and height must be multiples of 64 with at most 2097152 total pixels")
    seed = _integer(payload.get("seed", secrets.randbits(63)), "seed", 0, 2**63 - 1)
    steps = _integer(payload.get("steps", defaults["steps"]), "steps", 1, 60)
    cfg = _number(payload.get("cfg", defaults["cfg"]), "cfg", 0, 10)
    image_bytes, image_format = _image(payload.get("image_base64"))
    return GenerationRequest(preset_name, prompt.strip(), seed, width, height, steps,
                             cfg, image_bytes, image_format)


def build_prompt(request: GenerationRequest, catalog: dict, root: Path,
                 output_prefix: str, image_name: str | None = None) -> dict:
    preset = catalog["presets"][request.preset]
    graph = copy.deepcopy(json.loads(_package_file(root, preset["api"]).read_text())["prompt"])
    nodes = preset["nodes"]
    expected = {"positive": "TextEncodeFibo2", "sampler": "KSampler",
                "save": "SaveImage", "load_image": "LoadImage"}
    for key, class_type in expected.items():
        if graph[nodes[key]]["class_type"] != class_type:
            raise ValueError(f"Preset {key} node is invalid")
    if not image_name:
        raise ValueError("edit requires an uploaded source image")
    graph[nodes["positive"]]["inputs"].update(
        prompt=request.prompt, width=request.width, height=request.height)
    graph[nodes["sampler"]]["inputs"].update(
        seed=request.seed, steps=request.steps, cfg=request.cfg)
    graph[nodes["save"]]["inputs"]["filename_prefix"] = output_prefix
    graph[nodes["load_image"]]["inputs"]["image"] = image_name
    return graph


class ComfyJobRunner:
    def __init__(self, comfy_url: str | None = None, data_dir: Path | None = None,
                 root: Path | None = None, timeout: float = 600):
        self.root = Path(root) if root is not None else _root()
        self.data_dir = Path(data_dir or os.environ.get("FIBO_DATA_DIR", "/workspace/ComfyUI"))
        self.comfy_url = (comfy_url or os.environ.get("COMFY_URL", "http://127.0.0.1:8189")).rstrip("/")
        parts = urlsplit(self.comfy_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("COMFY_URL must be an HTTP(S) service URL without credentials or query parameters")
        self.ws_url = urlunsplit(("wss" if parts.scheme == "https" else "ws",
                                 parts.netloc, parts.path + "/ws", "", ""))
        self.timeout = timeout
        self._lock = _WORKER_LOCK

    def _private_directory(self, kind: str, job_id: str) -> Path:
        parent = self.data_dir / kind
        parent.mkdir(parents=True, exist_ok=True)
        jobs = parent / "api_jobs"
        jobs.mkdir(exist_ok=True)
        if not jobs.resolve().is_relative_to(parent.resolve()):
            raise RuntimeError("API job storage must stay within its input/output directory")
        directory = jobs / job_id
        directory.mkdir(mode=0o700)
        return directory

    async def _cancel_owned(self, prompt_id: str) -> None:
        # Both APIs are scoped to this one prompt. Never send a global interrupt.
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.post(self.comfy_url + "/queue", json={"delete": [prompt_id]}, allow_redirects=False) as response:
                    await response.read()
                async with session.post(self.comfy_url + "/interrupt", json={"prompt_id": prompt_id}, allow_redirects=False) as response:
                    await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass

    async def _upload(self, session: aiohttp.ClientSession, request: GenerationRequest,
                      job_id: str) -> str:
        extension, mime = IMAGE_FORMATS[request.image_format]
        filename = "input." + extension
        subfolder = "api_jobs/" + job_id
        form = aiohttp.FormData()
        form.add_field("image", request.image_bytes, filename=filename, content_type=mime)
        form.add_field("type", "input")
        form.add_field("subfolder", subfolder)
        form.add_field("overwrite", "false")
        async with session.post(self.comfy_url + "/upload/image", data=form, allow_redirects=False) as response:
            response.raise_for_status()
            result = await response.json()
        if result.get("name") != filename or result.get("subfolder") != subfolder or result.get("type") != "input":
            raise RuntimeError("ComfyUI returned an unexpected upload location")
        return subfolder + "/" + filename

    async def _download(self, session: aiohttp.ClientSession, output: dict,
                        job_id: str, request: GenerationRequest) -> str:
        filename = output.get("filename")
        subfolder = "api_jobs/" + job_id
        if (not isinstance(filename, str) or filename in {"", ".", ".."}
                or "\x00" in filename or Path(filename).name != filename
                or "/" in filename or "\\" in filename
                or output.get("subfolder") != subfolder or output.get("type") != "output"):
            raise RuntimeError("ComfyUI returned an output outside this job's private directory")
        async with session.get(self.comfy_url + "/view", params={
            "filename": filename, "subfolder": subfolder, "type": "output",
        }, allow_redirects=False) as response:
            response.raise_for_status()
            blocks, count = [], 0
            async for block in response.content.iter_chunked(1024 * 1024):
                count += len(block)
                if count > MAX_OUTPUT_BYTES:
                    raise RuntimeError("Generated output exceeds the image response limit")
                blocks.append(block)
        content = b"".join(blocks)
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in IMAGE_FORMATS or image.size != (request.width, request.height):
                    raise RuntimeError("Generated output has an unexpected format or dimensions")
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise RuntimeError("ComfyUI returned an invalid image") from error
        return base64.b64encode(content).decode("ascii")

    async def run(self, payload: dict) -> dict:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A generation is already running on this worker")
        prompt_id = None
        private_dirs = []
        finished = False
        try:
            catalog = load_catalog(self.root)
            request = validate_request(payload, catalog)
            job_id = uuid.uuid4().hex
            private_dirs.append(self._private_directory("output", job_id))
            if request.image_bytes is not None:
                private_dirs.append(self._private_directory("input", job_id))
            client_id = uuid.uuid4().hex
            async with asyncio.timeout(self.timeout):
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, connect=15)) as session:
                    async with session.ws_connect(self.ws_url, params={"clientId": client_id},
                                                  heartbeat=30, max_msg_size=MAX_OUTPUT_BYTES) as websocket:
                        image_name = await self._upload(session, request, job_id) if request.image_bytes is not None else None
                        graph = build_prompt(request, catalog, self.root, "api_jobs/" + job_id + "/image", image_name)
                        async with session.post(self.comfy_url + "/prompt", json={
                            "prompt": graph, "client_id": client_id,
                        }, allow_redirects=False) as response:
                            response.raise_for_status()
                            result = await response.json()
                        prompt_id = result.get("prompt_id")
                        if not isinstance(prompt_id, str) or not prompt_id or result.get("error") or result.get("node_errors"):
                            raise RuntimeError("ComfyUI rejected the preset; verify its installed models and nodes")
                        outputs = []
                        save_node = catalog["presets"][request.preset]["nodes"]["save"]
                        while True:
                            message = await websocket.receive()
                            if message.type == aiohttp.WSMsgType.BINARY:
                                continue
                            if message.type != aiohttp.WSMsgType.TEXT:
                                raise RuntimeError("ComfyUI WebSocket closed before this job completed")
                            event = json.loads(message.data)
                            data = event.get("data", {})
                            if not isinstance(data, dict) or data.get("prompt_id") != prompt_id:
                                continue
                            event_type = event.get("type")
                            if event_type in {"execution_error", "execution_interrupted"}:
                                finished = True
                                raise RuntimeError("ComfyUI generation failed at node " + str(data.get("node_id", data.get("node", "unknown"))))
                            if event_type == "executed" and str(data.get("node")) == str(save_node):
                                outputs = data.get("output", {}).get("images", [])
                            if event_type == "execution_success" or (event_type == "executing" and data.get("node") is None):
                                finished = True
                                break
                        if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
                            raise RuntimeError("This job completed without its expected single image output")
                        encoded = await self._download(session, outputs[0], job_id, request)
            return {"images": [{"type": "base64", "data": encoded}],
                    "seed": request.seed, "preset": request.preset}
        finally:
            try:
                if isinstance(prompt_id, str) and prompt_id and not finished:
                    await self._cancel_owned(prompt_id)
                for directory in private_dirs:
                    if directory.is_dir() and not directory.is_symlink():
                        shutil.rmtree(directory)
            finally:
                self._lock.release()


_default_runner: ComfyJobRunner | None = None


def handle_event(event: dict, runner: ComfyJobRunner | None = None) -> dict:
    """Runpod SDK handler contract; validation/execution errors deliberately raise."""
    global _default_runner
    if not isinstance(event, dict) or not isinstance(event.get("input"), dict):
        raise ValueError("The job must contain an input object")
    if runner is None:
        if _default_runner is None:
            _default_runner = ComfyJobRunner()
        runner = _default_runner
    return asyncio.run(runner.run(event["input"]))
