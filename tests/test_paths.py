"""Model uploads are preserved while exposing the ordinary ComfyUI model directory."""
import errno
from pathlib import Path
from unittest.mock import patch

import pytest

from fibo_runtime import paths


def write(path, content=b'fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_custom_upload_moves_to_canonical_loras_and_baked_path_aliases_it(tmp_path):
    comfy, data = tmp_path / 'baked', tmp_path / 'ComfyUI'
    write(comfy / 'models/loras/put_loras_here', b'')
    upload = write(comfy / 'models/loras/uploaded.safetensors', b'user uploaded weights')
    identity = paths._identity(upload)
    write(data / 'models/loras/bundled.safetensors', b'bundled weights')
    paths.expose_persistent_models(comfy, data)
    target = data / 'models/loras/uploaded.safetensors'
    assert paths._identity(target) == identity
    assert (comfy / 'models').is_symlink()
    assert (comfy / 'models/loras/uploaded.safetensors').samefile(target)
    assert not (data / 'models/loras/put_loras_here').exists()
    assert (data / 'models/loras/bundled.safetensors').read_bytes() == b'bundled weights'
    paths.expose_persistent_models(comfy, data)  # Idempotent restart.


def test_identical_uploaded_copy_is_deduplicated_but_conflicts_move_nothing(tmp_path):
    comfy, data = tmp_path / 'baked', tmp_path / 'ComfyUI'
    write(comfy / 'models/loras/same.safetensors', b'identical')
    target = write(data / 'models/loras/same.safetensors', b'identical')
    identity = paths._identity(target)
    paths.expose_persistent_models(comfy, data)
    assert paths._identity(target) == identity
    assert (comfy / 'models').is_symlink()
    other, destination = tmp_path / 'other', tmp_path / 'destination'
    write(other / 'models/loras/a-new.safetensors', b'new')
    write(other / 'models/loras/conflict.safetensors', b'old')
    write(destination / 'models/loras/conflict.safetensors', b'new')
    with pytest.raises(RuntimeError, match='conflicts'):
        paths.expose_persistent_models(other, destination)
    assert (other / 'models/loras/a-new.safetensors').read_bytes() == b'new'
    assert not (destination / 'models/loras/a-new.safetensors').exists()
    assert (destination / 'models/loras/conflict.safetensors').read_bytes() == b'new'


def test_cross_filesystem_upload_copy_is_verified_before_source_removal(tmp_path):
    comfy, data = tmp_path / 'baked', tmp_path / 'ComfyUI'
    source = write(comfy / 'models/loras/custom.safetensors', b'opaque user weights')
    actual_link = paths.os.link

    def cross_device(path, destination, **kwargs):
        if Path(path) == source:
            raise OSError(errno.EXDEV, 'cross-device fixture')
        return actual_link(path, destination, **kwargs)

    with patch.object(paths.os, 'link', side_effect=cross_device):
        paths.expose_persistent_models(comfy, data)
    assert (data / 'models/loras/custom.safetensors').read_bytes() == b'opaque user weights'
    assert (comfy / 'models').is_symlink()
    assert not list((data / 'models/loras').glob('.fibo-migrate-*'))


def test_unexpected_links_and_same_directory_are_never_deleted(tmp_path):
    comfy, data = tmp_path / 'baked', tmp_path / 'ComfyUI'
    external = write(tmp_path / 'elsewhere/keep.safetensors')
    (comfy / 'models').mkdir(parents=True)
    (comfy / 'models/linked.safetensors').symlink_to(external)
    with pytest.raises(RuntimeError, match='Unexpected model entry'):
        paths.expose_persistent_models(comfy, data)
    assert external.read_bytes() == b'fixture'
    same = tmp_path / 'same'
    write(same / 'models/loras/keep.safetensors')
    paths.expose_persistent_models(same, same)
    assert (same / 'models/loras/keep.safetensors').read_bytes() == b'fixture'
