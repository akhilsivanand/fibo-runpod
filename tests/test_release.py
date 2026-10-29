"""Release gates reject incomplete evidence; reports here are synthetic unit fixtures."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_release', ROOT / 'scripts/check_release.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ReleaseChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'package'
        shutil.copytree(ROOT, self.root, ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache'))

    def write_fixture_report(self):
        version = tomllib.loads((self.root / 'pyproject.toml').read_text())['project']['version']
        job = {'preset': 'edit', 'completed': True, 'width': 1024, 'height': 1024,
               'image_sha256': 'b' * 64}
        report = {
            'schema_version': 1, 'package_version': version,
            'source_revision': 'a' * 40, 'image_digest': 'sha256:' + 'c' * 64,
            'validated_at': '2026-10-29T00:00:00+00:00',
            'gpu': {'name': 'Synthetic test device', 'vram_gb': 24, 'cuda_runtime': '13.0',
                    'driver': 'test-only'},
            'pod': {'fresh_volume': True, 'auth_verified': True, 'restart_preserved': True,
                    'jobs': [job]},
            'serverless': {'fresh_worker': True, 'jobs': [job]},
        }
        (self.root / 'docs').mkdir(exist_ok=True)
        (self.root / 'docs/validation.json').write_text(json.dumps(report))
        return report

    def test_prepared_source_does_not_require_gpu_report(self):
        (self.root / 'docs/validation.json').unlink(missing_ok=True)
        result = checker.check_release(self.root)
        self.assertEqual(result['source'], 'passed')
        self.assertEqual(result['gpu_validation'], 'not_checked')

    def test_missing_edit_workflow_is_a_source_failure(self):
        catalog = checker.read_json(self.root / 'config/workflows.json')
        (self.root / catalog['presets']['edit']['api']).unlink()
        with self.assertRaisesRegex(checker.ReleaseError, 'missing'):
            checker.check_release(self.root)

    def test_model_url_must_use_its_pinned_revision(self):
        path = self.root / 'config/models.json'
        manifest = checker.read_json(path)
        model = manifest['models'][0]
        model['url'] = model['url'].replace(model['source']['revision'], 'main')
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(checker.ReleaseError, 'pinned'):
            checker.check_manifest(self.root)

    def test_launch_rejects_shared_password_or_token(self):
        path = self.root / 'deploy/pod-template.json'
        original = checker.read_json(path)
        for variable in ('FIBO_PASSWORD', 'JUPYTER_PASSWORD'):
            with self.subTest(variable=variable):
                path.write_text(json.dumps({**original, 'env': {**original['env'], variable: 'shared-token'}}))
                with self.assertRaises(checker.ReleaseError):
                    checker.check_deployment(self.root)

    def test_native_connection_and_baked_startup_are_required(self):
        path = self.root / 'deploy/pod-console.json'
        original = checker.read_json(path)
        path.write_text(json.dumps({**original, 'startJupyter': False}))
        with self.assertRaises(checker.ReleaseError):
            checker.check_deployment(self.root)
        path.write_text(json.dumps(original))
        path = self.root / 'deploy/pod-template.json'
        pod = checker.read_json(path)
        pod['dockerStartCmd'] = []
        path.write_text(json.dumps(pod))
        with self.assertRaises(checker.ReleaseError):
            checker.check_deployment(self.root)

    def test_report_must_match_exact_immutable_image_and_package_version(self):
        report = self.write_fixture_report()
        pod = {'imageName': 'ghcr.io/example-maintainer/fibo-comfyui:staging'}
        with self.assertRaisesRegex(checker.ReleaseError, 'immutable digest'):
            checker.check_gpu_report(self.root, pod)
        pod['imageName'] = 'ghcr.io/example-maintainer/fibo-comfyui@sha256:' + 'd' * 64
        with self.assertRaisesRegex(checker.ReleaseError, 'differs from the GPU-tested digest'):
            checker.check_gpu_report(self.root, pod)
        pod['imageName'] = 'ghcr.io/example-maintainer/fibo-comfyui@' + report['image_digest']
        checker.check_gpu_report(self.root, pod)
        report['package_version'] = '0.0.0'
        (self.root / 'docs/validation.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(checker.ReleaseError, 'package version'):
            checker.check_gpu_report(self.root, pod)

    def test_incomplete_edit_or_missing_api_evidence_is_rejected(self):
        for mutation in ('incomplete', 'missing_api'):
            report = self.write_fixture_report()
            if mutation == 'incomplete':
                report['pod']['jobs'][0]['completed'] = False
            else:
                report['serverless']['jobs'] = []
            (self.root / 'docs/validation.json').write_text(json.dumps(report))
            pod = {'imageName': 'ghcr.io/example-maintainer/fibo-comfyui@' + report['image_digest']}
            with self.subTest(mutation=mutation), self.assertRaises(checker.ReleaseError):
                checker.check_gpu_report(self.root, pod)

    def test_publication_requires_a_new_gpu_report(self):
        (self.root / 'docs/validation.json').unlink(missing_ok=True)
        with self.assertRaises(checker.ReleaseError):
            checker.check_release(self.root, publish=True)

    def test_archive_excludes_unreviewed_json_and_private_outputs(self):
        spec = importlib.util.spec_from_file_location('package_release', self.root / 'scripts/package_release.py')
        packager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(packager)
        (self.root / 'docs/unreviewed-publication.json').write_text('{"private": true}')
        (self.root / 'output').mkdir(exist_ok=True)
        (self.root / 'output/private.png').write_bytes(b'private fixture')
        output = Path(self.temporary.name) / 'source.zip'
        packager.package_release(self.root, output)
        with zipfile.ZipFile(output) as archive:
            self.assertNotIn('fibo-runpod/docs/unreviewed-publication.json', archive.namelist())
            self.assertNotIn('fibo-runpod/output/private.png', archive.namelist())
            self.assertIn('fibo-runpod/handler.py', archive.namelist())
