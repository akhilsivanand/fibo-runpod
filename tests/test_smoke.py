"""Exercise smoke-check failure detection without starting Torch or networking."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fibo_smoke_container", ROOT / "scripts/smoke_container.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class SmokeSchemaTests(unittest.TestCase):
    def setUp(self):
        self.definitions = {
            "UNETLoader": {
                "input": {"required": {
                    "unet_name": [[], {}], "weight_dtype": [["default", "fp8_e4m3fn"], {}],
                }},
                "output": ["MODEL"],
            },
            "KSampler": {
                "input": {"required": {
                    "model": ["MODEL"], "seed": ["INT", {"min": 0}],
                    "sampler_name": [["euler", "er_sde"], {}],
                    "scheduler": [["simple", "beta"], {}],
                }},
                "output": ["LATENT"],
            },
        }
        self.graph = {
            "1": {"class_type": "UNETLoader", "inputs": {
                "unet_name": "missing-on-purpose.safetensors", "weight_dtype": "default",
            }},
            "2": {"class_type": "KSampler", "inputs": {
                "model": ["1", 0], "seed": 42, "sampler_name": "er_sde", "scheduler": "simple",
            }},
        }

    def test_model_selectors_can_be_empty_but_the_schema_is_checked(self):
        self.assertEqual(smoke.validate_api_graph(self.graph, self.definitions), {"UNETLoader", "KSampler"})

    def test_api_wrapper_is_supported(self):
        self.assertEqual(smoke.validate_api_graph({"prompt": self.graph}, self.definitions), {"UNETLoader", "KSampler"})

    def test_missing_native_node_is_rejected(self):
        self.graph["1"]["class_type"] = "MissingCustomLoader"
        with self.assertRaisesRegex(smoke.SmokeError, "not available"):
            smoke.validate_api_graph(self.graph, self.definitions)

    def test_new_or_misspelled_input_is_rejected(self):
        self.graph["2"]["inputs"]["unexpected_sampler_flag"] = True
        with self.assertRaisesRegex(smoke.SmokeError, "unsupported input"):
            smoke.validate_api_graph(self.graph, self.definitions)

    def test_missing_required_input_is_rejected(self):
        del self.graph["2"]["inputs"]["seed"]
        with self.assertRaisesRegex(smoke.SmokeError, "missing required input"):
            smoke.validate_api_graph(self.graph, self.definitions)

    def test_sampler_and_scheduler_are_checked_against_actual_options(self):
        for field in ("sampler_name", "scheduler"):
            graph = deepcopy(self.graph)
            graph["2"]["inputs"][field] = "unsupported"
            with self.subTest(field=field), self.assertRaisesRegex(smoke.SmokeError, "unsupported option"):
                smoke.validate_api_graph(graph, self.definitions)

    def test_v3_combo_accepts_declared_options_and_rejects_unknown_options(self):
        # Pinned ComfyUI's V3 ImageScaleToTotalPixels uses this COMBO shape.
        self.definitions["ImageScaleToTotalPixels"] = {
            "input": {"required": {
                "upscale_method": ["COMBO", {
                    "options": ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    "multiselect": False,
                }],
                "megapixels": ["FLOAT", {"min": 0.01, "max": 16.0}],
                "resolution_steps": ["INT", {"min": 1, "max": 256}],
            }},
            "output": ["IMAGE"],
        }
        graph = {"8": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "upscale_method": "lanczos", "megapixels": 1, "resolution_steps": 16,
        }}}
        self.assertEqual(smoke.validate_api_graph(graph, self.definitions), {"ImageScaleToTotalPixels"})
        graph["8"]["inputs"]["upscale_method"] = "unsupported"
        with self.assertRaisesRegex(smoke.SmokeError, "unsupported option for upscale_method"):
            smoke.validate_api_graph(graph, self.definitions)

    def test_v3_combo_requires_declared_options_metadata(self):
        for descriptor in (["COMBO"], ["COMBO", {}], ["COMBO", {"options": "er_sde"}]):
            definitions = deepcopy(self.definitions)
            definitions["KSampler"]["input"]["required"]["sampler_name"] = descriptor
            with self.subTest(descriptor=descriptor), self.assertRaisesRegex(smoke.SmokeError, "invalid COMBO options schema"):
                smoke.validate_api_graph(self.graph, definitions)

    def test_v3_file_selector_can_have_no_installed_models(self):
        self.definitions["UNETLoader"]["input"]["required"]["unet_name"] = ["COMBO", {"options": []}]
        self.assertEqual(smoke.validate_api_graph(self.graph, self.definitions), {"UNETLoader", "KSampler"})

    def test_link_target_output_and_type_are_checked(self):
        for link, message in [(["missing", 0], "missing node"), (["1", 7], "missing output"), (["2", 0], "incompatible link")]:
            graph = deepcopy(self.graph)
            graph["2"]["inputs"]["model"] = link
            with self.subTest(link=link), self.assertRaisesRegex(smoke.SmokeError, message):
                smoke.validate_api_graph(graph, self.definitions)

    def test_scalar_type_and_limits_are_checked(self):
        for seed in (True, -1, 1.25):
            graph = deepcopy(self.graph)
            graph["2"]["inputs"]["seed"] = seed
            with self.subTest(seed=seed), self.assertRaises(smoke.SmokeError):
                smoke.validate_api_graph(graph, self.definitions)

    def test_official_subgraph_is_resolved_and_its_nodes_are_checked(self):
        graph = {
            "nodes": [{"id": 1, "type": "nested-id"}, {"id": 2, "type": "MarkdownNote"}],
            "definitions": {"subgraphs": [{"id": "nested-id", "nodes": [{"id": 7, "type": "UNETLoader"}]}]},
        }
        self.assertEqual(smoke.validate_ui_graph(graph, self.definitions, require_subgraph=True), {"UNETLoader"})
        graph["definitions"]["subgraphs"][0]["nodes"][0]["type"] = "UnavailableNode"
        with self.assertRaisesRegex(smoke.SmokeError, "unavailable"):
            smoke.validate_ui_graph(graph, self.definitions, require_subgraph=True)

    def test_official_missing_subgraph_is_detected(self):
        graph = {"nodes": [{"id": 1, "type": "UNETLoader"}]}
        with self.assertRaisesRegex(smoke.SmokeError, "nested subgraph is missing"):
            smoke.validate_ui_graph(graph, self.definitions, require_subgraph=True)

    def test_dynamic_fibo_image_inputs_are_validated_without_requiring_root_widget(self):
        definitions = {
            "LoadImage": {"input": {"required": {"image": [[], {}]}}, "output": ["IMAGE", "MASK"]},
            "TextEncodeFibo2": {"input": {"required": {
                "prompt": ["STRING", {}],
                "images": ["COMFY_AUTOGROW_V3", {"template": {
                    "input": {"required": {"image": ["IMAGE", {}]}},
                    "names": ["image_1", "image_2"], "min": 0,
                }}],
            }}, "output": ["CONDITIONING"]},
        }
        graph = {
            "10": {"class_type": "LoadImage", "inputs": {"image": ""}},
            "4": {"class_type": "TextEncodeFibo2", "inputs": {
                "prompt": "Change the object to red", "images.image_1": ["10", 0]}},
        }
        self.assertEqual(smoke.validate_api_graph(graph, definitions), {"LoadImage", "TextEncodeFibo2"})
        graph["4"]["inputs"]["images.image_1"] = ["10", 1]
        with self.assertRaisesRegex(smoke.SmokeError, "incompatible link"):
            smoke.validate_api_graph(graph, definitions)
        graph["4"]["inputs"] = {"prompt": "Edit", "images.image_3": ["10", 0]}
        with self.assertRaisesRegex(smoke.SmokeError, "unsupported input"):
            smoke.validate_api_graph(graph, definitions)

    def test_release_catalog_has_one_editing_template_and_api_preset(self):
        config, entries = smoke.load_catalog(ROOT)
        self.assertEqual(set(entries), {"edit"})
        self.assertEqual(set(config["presets"]), {"edit"})
        graph = smoke.read_json(ROOT / entries["edit"]["ui"])
        self.assertIn("TextEncodeFibo2", {node["type"] for node in graph["nodes"]})

    def test_catalog_cannot_escape_package_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "package"
            root.mkdir()
            outside = parent / "private.json"
            outside.write_text("{}")
            with self.assertRaisesRegex(smoke.SmokeError, "outside the package"):
                smoke.package_file(root, "../private.json")


class SmokeLifecycleTests(unittest.TestCase):
    def test_full_metadata_flow_fetches_the_native_edit_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "package"
            comfy = Path(temporary) / "comfy"
            (package / "config").mkdir(parents=True)
            (package / "workflows").mkdir()
            comfy.mkdir()
            (comfy / "main.py").write_text("")
            config = {"template_source": "fibo", "presets": {}, "ui_only": {}}
            api = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "absent.safetensors"}}}
            documents = {}
            for name in ("edit",):
                ui = {"nodes": [{"id": 1, "type": "UNETLoader"}]}
                documents[name + ".json"] = ui
                (package / "workflows" / (name + ".json")).write_text(json.dumps(ui))
                entry = {"ui": "workflows/" + name + ".json"}
                entry["api"] = "workflows/api-" + name + ".json"
                (package / entry["api"]).write_text(json.dumps(api))
                config["presets"][name] = entry
            (package / "config/workflows.json").write_text(json.dumps(config))
            definitions = {"KSampler": {}, "UNETLoader": {"input": {"required": {"unet_name": [[], {}]}}, "output": ["MODEL"]}}
            process = Mock()
            process.poll.return_value = None
            fetched = []
            def template(opener, url):
                prefix = "http://127.0.0.1:8187/api/workflow_templates/fibo/"
                self.assertTrue(url.startswith(prefix))
                filename = url.removeprefix(prefix)
                fetched.append(filename)
                return documents[filename]
            with patch.object(smoke.subprocess, "Popen", return_value=process), patch.object(smoke, "_wait_ready", return_value=(definitions, {"fibo": list(documents)})), patch.object(smoke, "_fetch_json", side_effect=template), patch.object(smoke, "_check_ui"), patch.object(smoke, "_stop_child"):
                result = smoke.run_smoke(package, comfy, port=8187)
            self.assertEqual(set(fetched), set(documents))
            self.assertEqual(result["templates"], 1)
            self.assertEqual(result["api_graphs"], 1)
            self.assertEqual(result["inference"], "not_run")

    def test_failure_terminates_child_and_cleans_its_temporary_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            comfy = Path(temporary)
            (comfy / "main.py").write_text("")
            process = Mock(pid=12345)
            process.poll.return_value = None
            process.wait.return_value = 0
            with patch.object(smoke.subprocess, "Popen", return_value=process) as launch, patch.object(smoke, "_wait_ready", side_effect=smoke.SmokeError("simulated startup failure")), patch.object(smoke.os, "killpg") as stop, patch.dict(os.environ, {"FIBO_PASSWORD": "must-not-reach-child"}):
                with self.assertRaisesRegex(smoke.SmokeError, "simulated startup failure"):
                    smoke.run_smoke(ROOT, comfy, port=8187)
            command = launch.call_args.args[0]
            workspace = Path(command[command.index("--base-directory") + 1])
            self.assertFalse(workspace.exists())
            self.assertIn("--cpu", command)
            self.assertIn("--offline", command)
            self.assertEqual(command[command.index("--listen") + 1], "127.0.0.1")
            options = launch.call_args.kwargs
            self.assertEqual(options["stdout"], subprocess.DEVNULL)
            self.assertEqual(options["stderr"], subprocess.DEVNULL)
            self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertNotIn("FIBO_PASSWORD", options["env"])
            self.assertTrue(options["start_new_session"])
            stop.assert_called_with(12345, signal.SIGTERM)

    def test_unresponsive_child_is_killed_after_termination_deadline(self):
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired("comfy", 10), 0]
        with patch.object(smoke.os, "killpg") as stop:
            smoke._stop_child(process)
        self.assertEqual([call.args for call in stop.call_args_list], [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])

    def test_native_ui_html_and_local_script_are_checked(self):
        body = b'<!doctype html><html><head><title>ComfyUI</title><script type="module" src="/assets/index.js"></script></head></html>'
        replies = [({"Content-Type": "text/html"}, body), ({"Content-Type": "text/javascript"}, b"")]
        with patch.object(smoke, "_fetch", side_effect=replies) as fetch:
            smoke._check_ui(Mock(), "http://127.0.0.1:8187")
        self.assertEqual(fetch.call_args.kwargs["method"], "HEAD")
        self.assertEqual(fetch.call_args.args[1], "http://127.0.0.1:8187/assets/index.js")

    def test_missing_native_ui_script_is_rejected(self):
        body = b"<html><head><title>ComfyUI</title></head></html>"
        with patch.object(smoke, "_fetch", return_value=({"Content-Type": "text/html"}, body)):
            with self.assertRaisesRegex(smoke.SmokeError, "script references are missing"):
                smoke._check_ui(Mock(), "http://127.0.0.1:8187")


if __name__ == "__main__":
    unittest.main()
