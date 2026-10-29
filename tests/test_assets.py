"""Networkless checks for restart recovery and model-file ownership."""

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fibo_runtime import assets


class ModelAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fibo-assets-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.data = self.base / "data"
        self.manifest = self.base / "models.json"
        self.payloads = {"one": b"first model weights", "two": b"second model weights"}
        self.models = [self.model(name, payload) for name, payload in self.payloads.items()]
        self.write_manifest(self.models)
        self.original_quota = os.environ.pop("FIBO_VOLUME_GB", None)
        self.addCleanup(self.restore_quota)
        self.capacity = patch.object(assets.shutil, "disk_usage", return_value=SimpleNamespace(free=10 ** 12))
        self.capacity.start()
        self.addCleanup(self.capacity.stop)
        self.transfers = []

    def restore_quota(self):
        if self.original_quota is None:
            os.environ.pop("FIBO_VOLUME_GB", None)
        else:
            os.environ["FIBO_VOLUME_GB"] = self.original_quota

    @staticmethod
    def model(identifier, payload):
        return {
            "id": identifier,
            "path": f"diffusion_models/{identifier}.safetensors",
            "url": f"https://example.invalid/resolve/{'a' * 40}/{identifier}.safetensors",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "license": "Apache-2.0",
            "source": {"repository": "example/models", "revision": "a" * 40},
            "profiles": ["common"],
        }

    def write_manifest(self, models):
        self.manifest.write_text(json.dumps({
            "schema_version": 1,
            "total_size_bytes": sum(item["size"] for item in models),
            "models": models,
        }))

    def destination(self, model):
        return self.data / "models" / model["path"]

    def transfer(self, model, partial, progress):
        offset = partial.stat().st_size
        self.transfers.append((model["id"], offset))
        with partial.open("ab") as output:
            output.write(self.payloads[model["id"]][offset:])
        progress(partial.stat().st_size)

    def prepare(self, **kwargs):
        with patch.object(assets, "_run_curl", side_effect=self.transfer):
            return assets.prepare_models(self.manifest, self.data, **kwargs)

    def make_owned_partial(self, model, payload):
        destination = self.destination(model)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = assets._claim_partial(destination, model)
        partial.write_bytes(payload)
        return partial

    def test_release_manifest_has_three_pinned_editing_assets(self):
        source = Path(__file__).resolve().parents[1] / "config/models.json"
        models = assets._load_manifest(source)
        self.assertEqual(len(models), 3)
        self.assertEqual({model["path"].split("/")[0] for model in models},
                         {"diffusion_models", "text_encoders", "vae"})
        self.assertTrue(all(model["source"]["revision"] in model["url"] for model in models))

    def test_dry_run_does_not_create_data_tree(self):
        result = assets.prepare_models(self.manifest, self.data, download=False)
        self.assertFalse(self.data.exists())
        self.assertEqual(result["bytes_pending"], sum(map(len, self.payloads.values())))
        self.assertTrue(all(item["status"] == "pending" for item in result["models"]))

    def test_first_install_and_restart_only_verify(self):
        events = []
        result = self.prepare(progress_callback=events.append)
        self.assertEqual(result["downloaded"], 2)
        self.assertEqual([item["id"] for item in result["models"]], ["one", "two"])
        for model in self.models:
            self.assertEqual(self.destination(model).read_bytes(), self.payloads[model["id"]])
            self.assertFalse(assets._partial_paths(self.destination(model))[0].exists())
        with patch.object(assets, "_run_curl", side_effect=AssertionError("Unexpected network transfer")):
            restarted = assets.prepare_models(self.manifest, self.data)
        self.assertEqual(restarted["verified"], 2)
        self.assertEqual(restarted["downloaded"], 0)
        self.assertEqual(events[-1]["phase"], "models_ready")
        self.assertEqual(events[-1]["bytes_complete"], result["bytes_total"])
        self.assertTrue(all("https:" not in event["detail"] for event in events))

    def test_interrupt_and_resume_preserve_downloaded_prefix(self):
        def interrupted(model, partial, progress):
            partial.write_bytes(self.payloads[model["id"]][:7])
            raise assets.AssetError("Simulated connection interruption")
        with patch.object(assets, "_run_curl", side_effect=interrupted):
            with self.assertRaises(assets.AssetError):
                assets.prepare_models(self.manifest, self.data)
        result = self.prepare()
        self.assertEqual(self.transfers[0], ("one", 7))
        self.assertEqual(result["bytes_pending"], sum(map(len, self.payloads.values())) - 7)

    def test_corrupt_owned_complete_partial_is_quarantined_then_redownloaded(self):
        model = self.models[0]
        corrupted = b"x" * model["size"]
        partial = self.make_owned_partial(model, corrupted)
        result = self.prepare()
        self.assertEqual(result["downloaded"], 2)
        self.assertEqual(self.transfers[0], ("one", 0))
        quarantined = list(partial.parent.glob(partial.name + ".corrupt.*"))
        contents = [path.read_bytes() for path in quarantined if not path.name.endswith(".json")]
        self.assertEqual(contents, [corrupted])
        self.assertEqual(self.destination(model).read_bytes(), self.payloads["one"])

    def test_bad_network_payload_has_only_one_clean_retry(self):
        self.write_manifest(self.models[:1])
        attempts = []
        def corrupt(model, partial, progress):
            attempts.append(partial.stat().st_size)
            partial.write_bytes(b"x" * model["size"])
        with patch.object(assets, "_run_curl", side_effect=corrupt):
            with self.assertRaisesRegex(assets.AssetError, "one clean retry"):
                assets.prepare_models(self.manifest, self.data)
        self.assertEqual(attempts, [0, 0])
        self.assertFalse(self.destination(self.models[0]).exists())

    def test_corrupt_incomplete_prefix_is_detected_after_resume(self):
        self.make_owned_partial(self.models[0], b"xxxxxxx")
        self.prepare()
        self.assertEqual(self.transfers[:2], [("one", 7), ("one", 0)])
        self.assertEqual(self.destination(self.models[0]).read_bytes(), self.payloads["one"])

    def test_existing_user_file_is_never_overwritten(self):
        destination = self.destination(self.models[0])
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"my custom model")
        with self.assertRaisesRegex(assets.AssetError, "preserving"):
            self.prepare()
        self.assertEqual(destination.read_bytes(), b"my custom model")
        self.assertFalse(self.transfers)

    def test_unowned_partial_is_preserved_without_quarantine(self):
        destination = self.destination(self.models[0])
        destination.parent.mkdir(parents=True)
        partial, _ = assets._partial_paths(destination)
        partial.write_bytes(b"user partial")
        with self.assertRaisesRegex(assets.AssetError, "Unrecognized partial"):
            self.prepare()
        self.assertEqual(partial.read_bytes(), b"user partial")
        self.assertFalse(list(partial.parent.glob("*.corrupt.*")))
        self.assertFalse(self.transfers)

    def test_replaced_partial_inode_is_preserved(self):
        model = self.models[0]
        partial = self.make_owned_partial(model, b"prefix")
        replacement = partial.with_name("replacement")
        replacement.write_bytes(b"user replacement")
        replacement.replace(partial)
        with self.assertRaisesRegex(assets.AssetError, "identity changed"):
            self.prepare()
        self.assertEqual(partial.read_bytes(), b"user replacement")

    def test_final_created_during_transfer_is_never_overwritten(self):
        self.write_manifest(self.models[:1])
        def racing_transfer(model, partial, progress):
            self.transfer(model, partial, progress)
            self.destination(model).write_bytes(b"user file created concurrently")
        with patch.object(assets, "_run_curl", side_effect=racing_transfer):
            with self.assertRaisesRegex(assets.AssetError, "preserving"):
                assets.prepare_models(self.manifest, self.data)
        self.assertEqual(self.destination(self.models[0]).read_bytes(), b"user file created concurrently")

    def test_symlink_destination_or_partial_is_not_followed(self):
        model = self.models[0]
        destination = self.destination(model)
        destination.parent.mkdir(parents=True)
        owner = self.base / "owner-file"
        owner.write_bytes(b"owner data")
        for path in [destination, assets._partial_paths(destination)[0]]:
            with self.subTest(path=path.name):
                path.symlink_to(owner)
                with self.assertRaisesRegex(assets.AssetError, "symlink"):
                    self.prepare()
                self.assertEqual(owner.read_bytes(), b"owner data")
                self.assertTrue(path.is_symlink())
                path.unlink()

    def test_symlink_model_directory_is_not_followed(self):
        self.data.mkdir()
        owner = self.base / "owner-directory"
        owner.mkdir()
        (self.data / "models").symlink_to(owner, target_is_directory=True)
        with self.assertRaisesRegex(assets.AssetError, "symlink"):
            self.prepare()
        self.assertEqual(list(owner.iterdir()), [])

    def test_all_pending_files_are_budgeted_together(self):
        pending = sum(map(len, self.payloads.values()))
        with patch.object(assets.shutil, "disk_usage", return_value=SimpleNamespace(free=assets.HEADROOM_BYTES + pending - 1)):
            with self.assertRaisesRegex(assets.AssetError, "filesystem capacity"):
                self.prepare()
        self.assertFalse(self.transfers)

    def test_partial_credit_reduces_required_capacity(self):
        self.make_owned_partial(self.models[0], self.payloads["one"][:7])
        pending = sum(map(len, self.payloads.values())) - 7
        with patch.object(assets.shutil, "disk_usage", return_value=SimpleNamespace(free=assets.HEADROOM_BYTES + pending)):
            result = self.prepare()
        self.assertEqual(result["bytes_pending"], pending)

    def test_corrupt_complete_partial_needs_full_replacement_capacity(self):
        self.write_manifest(self.models[:1])
        model = self.models[0]
        partial = self.make_owned_partial(model, b"x" * model["size"])
        with patch.object(assets.shutil, "disk_usage", return_value=SimpleNamespace(free=assets.HEADROOM_BYTES + model["size"] - 1)):
            with self.assertRaisesRegex(assets.AssetError, "filesystem capacity"):
                self.prepare()
        self.assertTrue(partial.exists())
        self.assertFalse(self.transfers)

    def test_configured_volume_quota_counts_existing_tree_metadata(self):
        self.data.mkdir()
        (self.data / "user-file").write_bytes(b"x" * 30)
        pending = sum(map(len, self.payloads.values()))
        capacity = 30 + pending + 10 - 1
        with patch.dict(os.environ, {"FIBO_VOLUME_GB": str(capacity / 1e9)}), patch.object(assets, "HEADROOM_BYTES", 10):
            with self.assertRaisesRegex(assets.AssetError, "Configured volume capacity"):
                self.prepare()
        self.assertFalse(self.transfers)

    def test_invalid_quota_is_rejected(self):
        for value in ["0", "-1", "nan", "Infinity", "not a number"]:
            with self.subTest(value=value), patch.dict(os.environ, {"FIBO_VOLUME_GB": value}):
                with self.assertRaisesRegex(assets.AssetError, "FIBO_VOLUME_GB"):
                    assets.prepare_models(self.manifest, self.data, download=False)

    def test_manifest_traversal_rejected_before_creating_data(self):
        for value in ["../escape.safetensors", "/outside.safetensors", "a/../escape", "a//model", "a\\model", "./model"]:
            with self.subTest(value=value):
                model = dict(self.models[0], path=value)
                self.write_manifest([model])
                with self.assertRaises(assets.AssetError):
                    self.prepare()
                self.assertFalse(self.data.exists())

    def test_manifest_urls_cannot_embed_credentials(self):
        model = dict(self.models[0], url="https://user:password@example.invalid/model")
        self.write_manifest([model])
        with self.assertRaisesRegex(assets.AssetError, "public HTTPS"):
            self.prepare()
        self.assertFalse(self.data.exists())

    def test_manifest_destinations_cannot_overlap_as_parent_and_child(self):
        child = dict(self.models[1], path=self.models[0]["path"] + "/nested.safetensors")
        self.write_manifest([self.models[0], child])
        with self.assertRaisesRegex(assets.AssetError, "contain other model"):
            self.prepare()
        self.assertFalse(self.data.exists())

    def test_model_workers_are_serialized(self):
        outcomes, failures = [], []
        def slow_transfer(model, partial, progress):
            time.sleep(0.03)
            self.transfer(model, partial, progress)
        def worker():
            try:
                outcomes.append(assets.prepare_models(self.manifest, self.data))
            except BaseException as error:
                failures.append(error)
        with patch.object(assets, "_run_curl", side_effect=slow_transfer):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(failures, [])
        self.assertEqual(sorted(result["downloaded"] for result in outcomes), [0, 2])
        self.assertEqual(len(self.transfers), 2)

    def test_symlink_lock_file_is_preserved(self):
        state = self.data / "state"
        state.mkdir(parents=True)
        owner = self.base / "owner-lock"
        owner.write_bytes(b"user lock data")
        lock = state / "models.lock"
        lock.symlink_to(owner)
        with self.assertRaisesRegex(assets.AssetError, "symlink"):
            self.prepare()
        self.assertEqual(owner.read_bytes(), b"user lock data")

    def test_curl_is_bounded_and_resumes_after_transfer_failure(self):
        model = self.models[0]
        partial = self.make_owned_partial(model, b"")
        commands = []
        def fake_curl(command, stdout, stderr):
            commands.append(command)
            offset = partial.stat().st_size
            payload = self.payloads[model["id"]]
            stdout.write(payload[offset:7] if len(commands) == 1 else payload[offset:])
            code = 56 if len(commands) == 1 else 0
            return SimpleNamespace(
                poll=lambda: code, returncode=code, communicate=lambda: (None, b""),
                stderr=io.BytesIO(),
            )
        with patch.object(assets.subprocess, "Popen", side_effect=fake_curl), patch.object(assets.shutil, "which", return_value="/usr/bin/curl"):
            assets._run_curl(model, partial, lambda count: None)
        self.assertEqual([command[command.index("--continue-at") + 1] for command in commands], ["0", "7"])
        for command in commands:
            self.assertIn("--connect-timeout", command)
            self.assertIn("--speed-time", command)
            self.assertIn("--max-time", command)
            self.assertIn("--max-filesize", command)
            self.assertEqual(command[command.index("--proto-redir") + 1], "=https")
        self.assertEqual(partial.read_bytes(), self.payloads[model["id"]])

    def test_complete_payload_can_be_verified_after_curl_reports_error(self):
        self.write_manifest(self.models[:1])
        def fake_curl(command, stdout, stderr):
            stdout.write(self.payloads["one"])
            return SimpleNamespace(
                poll=lambda: 56, returncode=56, communicate=lambda: (None, b""),
                stderr=io.BytesIO(),
            )
        with patch.object(assets, "CURL_ATTEMPTS", 1), patch.object(assets.subprocess, "Popen", side_effect=fake_curl), patch.object(assets.shutil, "which", return_value="/usr/bin/curl"):
            result = assets.prepare_models(self.manifest, self.data)
        self.assertEqual(result["downloaded"], 1)
        self.assertEqual(self.destination(self.models[0]).read_bytes(), self.payloads["one"])

    def test_python_guard_terminates_an_unresponsive_curl(self):
        model = self.models[0]
        partial = self.make_owned_partial(model, b"")
        state = {"code": None, "killed": False}
        def kill():
            state.update(code=-9, killed=True)
        process = SimpleNamespace(
            poll=lambda: state["code"], kill=kill, wait=lambda: state["code"],
            stderr=io.BytesIO(),
        )
        with patch.object(assets.subprocess, "Popen", return_value=process), patch.object(assets.shutil, "which", return_value="/usr/bin/curl"), patch.object(assets, "CURL_MAX_SECONDS", 1), patch.object(assets.time, "monotonic", side_effect=[0, 100]):
            with self.assertRaisesRegex(assets.AssetError, "time limit"):
                assets._run_curl(model, partial, lambda count: None)
        self.assertTrue(state["killed"])


if __name__ == "__main__":
    unittest.main()
