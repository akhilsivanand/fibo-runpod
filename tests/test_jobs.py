"""CPU-only tests: request boundaries, preset mapping, and owned-job isolation."""

import asyncio
import base64
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image
import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fibo_runtime.jobs import (ComfyJobRunner, build_prompt, handle_event,
                               load_catalog, validate_request, MAX_IMAGE_BYTES)


def image_bytes(size=(64, 64), image_format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", size, "#446655").save(stream, format=image_format)
    return stream.getvalue()


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog(ROOT)
        self.encoded = base64.b64encode(image_bytes()).decode()

    def validate(self, **kwargs):
        return validate_request({"prompt": "Change the object to red", "image_base64": self.encoded,
                                 **kwargs}, self.catalog)

    def test_edit_defaults_and_reference_conditioning_preserved(self):
        request = self.validate(seed=0)
        self.assertEqual((request.preset, request.width, request.height), ("edit", 1024, 1024))
        graph = build_prompt(request, self.catalog, ROOT, "api_jobs/test/image", "api_jobs/test/input.png")
        nodes = self.catalog["presets"]["edit"]["nodes"]
        text = graph[nodes["positive"]]
        sampler = graph[nodes["sampler"]]["inputs"]
        self.assertEqual(text["class_type"], "TextEncodeFibo2")
        self.assertEqual(text["inputs"]["prompt"], "Change the object to red")
        self.assertEqual(text["inputs"]["images.image_1"], [nodes["load_image"], 0])
        self.assertEqual((sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]),
                         (4, 1, "euler", "simple"))
        self.assertEqual(sampler["positive"], [nodes["positive"], 0])
        self.assertEqual(sampler["latent_image"], [nodes["positive"], 2])
        self.assertFalse(any(node["class_type"].startswith("Lora") for node in graph.values()))

    def test_hostile_and_invalid_inputs_are_rejected(self):
        bad = [
            {"prompt": " "}, {"prompt": "x" * 12001}, {"seed": True},
            {"seed": -1}, {"seed": 2**63}, {"width": 65}, {"height": 0},
            {"width": 2048, "height": 2048}, {"steps": 61}, {"cfg": float("nan")},
            {"cfg": float("inf")}, {"cfg": True}, {"preset": "fast"},
            {"prompt": 42}, {"graph": {}}, {"url": "http://example.invalid/image"},
            {"lora_name": "adapter.safetensors"}, {"lora_strength": 1},
        ]
        for fields in bad:
            with self.subTest(fields=repr(fields)[:80]), self.assertRaises(ValueError):
                self.validate(**fields)

    def test_source_image_is_required_and_dimensions_reach_fibo_encoder(self):
        request = self.validate(width=512, height=768)
        graph = build_prompt(request, self.catalog, ROOT, "api_jobs/test/image", "api_jobs/test/input.png")
        nodes = self.catalog["presets"]["edit"]["nodes"]
        self.assertEqual(graph[nodes["load_image"]]["inputs"]["image"], "api_jobs/test/input.png")
        encoder = graph[nodes["positive"]]["inputs"]
        self.assertEqual((encoder["width"], encoder["height"]), (512, 768))
        with self.assertRaisesRegex(ValueError, "uploaded source image"):
            build_prompt(request, self.catalog, ROOT, "api_jobs/test/image")
        for invalid in (None, "http://example.invalid/image.png", base64.b64encode(b"not an image").decode()):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.validate(image_base64=invalid)
        with self.assertRaisesRegex(ValueError, "7 MiB"):
            self.validate(image_base64="A" * (4 * ((MAX_IMAGE_BYTES + 2) // 3) + 4))

    def test_sdk_errors_raise(self):
        with self.assertRaises(ValueError):
            handle_event({"input": "not an object"})


class FakeResponse:
    def __init__(self, value=None, content=b""):
        self.value = value
        self.bytes = content
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self.value

    async def read(self):
        return self.bytes

    async def iter_chunked(self, size):
        yield self.bytes


class FakeWebSocket(FakeResponse):
    def __init__(self, transport):
        self.transport = transport

    async def receive(self):
        if self.transport.mode == "timeout":
            await asyncio.sleep(30)
        if not self.transport.events:
            return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None)
        event = self.transport.events.pop(0)
        return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(event))


class FakeTransport(FakeResponse):
    def __init__(self, mode="success"):
        self.mode = mode
        self.events = []
        self.calls = []

    def __call__(self, *args, **kwargs):
        return self

    def ws_connect(self, url, **kwargs):
        self.calls.append(("WS", url, kwargs))
        return FakeWebSocket(self)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if url.endswith("/upload/image"):
            fields = {metadata.get("name"): value for metadata, _, value in kwargs["data"]._fields}
            return FakeResponse({"name": "input.png", "subfolder":
                                 "foreign" if self.mode == "upload_escape" else fields["subfolder"], "type": "input"})
        if url.endswith("/prompt"):
            graph = kwargs["json"]["prompt"]
            self.subfolder = graph["7"]["inputs"]["filename_prefix"].rsplit("/", 1)[0]
            output = {"filename": "image_00001_.png", "subfolder": self.subfolder, "type": "output"}
            self.events = [{"type": "executed", "data": {"prompt_id": "someone-else", "node": "7", "output": {"images": [{"filename": "private.png", "subfolder": "other", "type": "output"}]}}}]
            if self.mode == "error":
                self.events.append({"type": "execution_error", "data": {"prompt_id": "owned-prompt", "node_id": "5"}})
            else:
                if self.mode == "escape":
                    output["subfolder"] = "someone-else"
                self.events.extend([
                    {"type": "executed", "data": {"prompt_id": "owned-prompt", "node": "7", "output": {"images": [output]}}},
                    {"type": "execution_success", "data": {"prompt_id": "owned-prompt"}},
                ])
            return FakeResponse({"prompt_id": "owned-prompt", "node_errors": {}})
        if url.endswith("/queue") or url.endswith("/interrupt"):
            return FakeResponse()
        raise AssertionError("Unexpected POST endpoint: " + url)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if not url.endswith("/view"):
            raise AssertionError("Unexpected GET endpoint: " + url)
        self.assert_owned(kwargs["params"])
        return FakeResponse(content=image_bytes())

    def assert_owned(self, params):
        if params != {"filename": "image_00001_.png", "subfolder": self.subfolder, "type": "output"}:
            raise AssertionError("Unowned artifact request")


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name)
        self.runner = ComfyJobRunner("http://127.0.0.1:8189", self.data, ROOT, timeout=1)
        self.payload = {"prompt": "Change the leaf to red", "seed": 7, "width": 64, "height": 64,
                        "image_base64": base64.b64encode(image_bytes()).decode()}

    async def test_only_owned_websocket_results_are_fetched_and_private_directories_are_cleaned(self):
        transport = FakeTransport()
        with patch("fibo_runtime.jobs.aiohttp.ClientSession", transport):
            result = await self.runner.run(self.payload)
        self.assertEqual((result["preset"], result["seed"]), ("edit", 7))
        self.assertEqual(base64.b64decode(result["images"][0]["data"]), image_bytes())
        self.assertEqual(result["images"][0]["type"], "base64")
        self.assertEqual(list((self.data / "output/api_jobs").iterdir()), [])
        self.assertEqual(list((self.data / "input/api_jobs").iterdir()), [])
        self.assertFalse(any("/history" in call[1] for call in transport.calls))
        self.assertEqual(sum(call[0] == "GET" for call in transport.calls), 1)

    async def test_execution_error_and_foreign_output_fail_without_false_success(self):
        for mode in ("error", "escape"):
            with self.subTest(mode=mode):
                transport = FakeTransport(mode)
                with patch("fibo_runtime.jobs.aiohttp.ClientSession", transport), self.assertRaises(RuntimeError):
                    await self.runner.run(self.payload)
                self.assertEqual(list((self.data / "output/api_jobs").iterdir()), [])
                self.assertFalse(any(call[0] == "GET" for call in transport.calls))

    async def test_foreign_upload_location_is_rejected_before_prompt_submission(self):
        transport = FakeTransport("upload_escape")
        with patch("fibo_runtime.jobs.aiohttp.ClientSession", transport), self.assertRaisesRegex(RuntimeError, "upload location"):
            await self.runner.run(self.payload)
        self.assertFalse(any(call[1].endswith("/prompt") for call in transport.calls))
        self.assertEqual(list((self.data / "input/api_jobs").iterdir()), [])
        self.assertEqual(list((self.data / "output/api_jobs").iterdir()), [])
        self.assertFalse(self.runner._lock.locked())

    async def test_timeout_cancels_only_its_prompt_and_releases_worker(self):
        transport = FakeTransport("timeout")
        self.runner.timeout = 0.01
        with patch("fibo_runtime.jobs.aiohttp.ClientSession", transport), self.assertRaises(TimeoutError):
            await self.runner.run(self.payload)
        cancellation = [call for call in transport.calls if call[1].endswith("/interrupt")]
        self.assertEqual(cancellation[0][2]["json"], {"prompt_id": "owned-prompt"})
        deletion = [call for call in transport.calls if call[1].endswith("/queue")]
        self.assertEqual(deletion[0][2]["json"], {"delete": ["owned-prompt"]})
        self.assertFalse(self.runner._lock.locked())

    async def test_concurrent_job_is_rejected(self):
        self.runner._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                await self.runner.run(self.payload)
        finally:
            self.runner._lock.release()


class SDKHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_entrypoint_awaits_job_in_running_event_loop(self):
        spec = importlib.util.spec_from_file_location("fibo_sdk_handler_under_test", ROOT / "handler.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = {"images": [{"type": "base64", "data": "fixture"}], "seed": 7, "preset": "edit"}
        runner = SimpleNamespace(run=AsyncMock(return_value=result))
        payload = {"prompt": "A green leaf", "preset": "edit", "seed": 7}
        with patch.object(module, "ComfyJobRunner", return_value=runner):
            # Mirrors runpod 1.12.0: call handler(job) inside its async run_job,
            # then await the returned awaitable.
            self.assertEqual(await module.handler({"input": payload}), result)
        runner.run.assert_awaited_once_with(payload)
        with self.assertRaises(ValueError):
            await module.handler({"input": "invalid"})


if __name__ == "__main__":
    unittest.main()
