"""Keep Pod data and uploaded models in the standard ComfyUI workspace."""
import errno
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile


DEFAULT_DATA_ROOT = Path('/workspace/ComfyUI')
PLACEHOLDER = re.compile(r'put_[A-Za-z0-9_]+_here\Z')


def _metadata(path):
    try:
        return Path(path).lstat()
    except FileNotFoundError:
        return None


def _plain_path(path):
    for component in (*reversed(path.parents), path):
        value = _metadata(component)
        if value is not None and stat.S_ISLNK(value.st_mode):
            raise RuntimeError(f'Workspace migration preserves unexpected symlinks: {component}')


def _identity(path):
    value = path.lstat()
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns


def _digest(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise RuntimeError(f'Expected a regular model file: {path}')
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
        return result.hexdigest()


def _move_model(source, destination, identity):
    if _identity(source) != identity:
        raise RuntimeError(f'Model changed during migration; preserving it: {source.name}')
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        # Cross-filesystem uploads must be copied once before the container
        # source is removed. Hashing verifies bytes; no model is deserialized.
        descriptor, name = tempfile.mkstemp(prefix='.fibo-migrate-', dir=destination.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            shutil.copy2(source, temporary, follow_symlinks=False)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            if _identity(source) != identity or _digest(source) != _digest(temporary):
                raise RuntimeError(f'Model changed while copying; preserving it: {source.name}')
            os.link(temporary, destination, follow_symlinks=False)
        finally:
            temporary.unlink(missing_ok=True)
    if _identity(source) != identity:
        raise RuntimeError(f'Model changed during migration; preserving both copies: {source.name}')
    source.unlink()


def expose_persistent_models(comfy_root, data_root):
    """Preserve container uploads, then alias the baked model path to Pod data.

    Preflight the complete model tree before changing any file. Different
    same-name files and unexpected links are preserved by aborting. Identical
    duplicates may be removed only after both stable files hash equally.
    """
    baked, models = Path(comfy_root) / 'models', Path(data_root) / 'models'
    _plain_path(models)
    if _metadata(models) is not None and not models.is_dir():
        raise RuntimeError('Canonical model path is not a directory; preserving it.')
    if baked == models or (baked.exists() and models.exists() and baked.samefile(models)):
        models.mkdir(parents=True, exist_ok=True)
        return
    if baked.is_symlink():
        if baked.resolve() != models.resolve():
            raise RuntimeError('Baked model directory points elsewhere; preserving its existing link.')
        models.mkdir(parents=True, exist_ok=True)
        return
    _plain_path(baked)
    if _metadata(baked) is not None and not baked.is_dir():
        raise RuntimeError('Baked model path is not a directory; preserving it.')
    directories, placeholders, files = [], [], []
    for source in sorted(baked.rglob('*')) if baked.exists() else []:
        metadata = source.lstat()
        destination = models / source.relative_to(baked)
        _plain_path(destination)
        existing = _metadata(destination)
        if stat.S_ISDIR(metadata.st_mode):
            if existing is not None and not stat.S_ISDIR(existing.st_mode):
                raise RuntimeError(f'Model directory conflicts; preserving both: {source.name}')
            directories.append(source)
        elif stat.S_ISREG(metadata.st_mode):
            if metadata.st_size == 0 and PLACEHOLDER.fullmatch(source.name):
                placeholders.append((source, _identity(source)))
                continue
            duplicate = None
            if existing is not None:
                if not stat.S_ISREG(existing.st_mode) or metadata.st_size != existing.st_size:
                    raise RuntimeError(f'Model conflicts; preserving both: {source.name}')
                identities = _identity(source), _identity(destination)
                if _digest(source) != _digest(destination) or identities != (_identity(source), _identity(destination)):
                    raise RuntimeError(f'Model conflicts; preserving both: {source.name}')
                duplicate = identities[1]
            files.append((source, destination, _identity(source), duplicate))
        else:
            raise RuntimeError(f'Unexpected model entry; preserving it: {source.name}')
    models.mkdir(parents=True, exist_ok=True)
    for directory in directories:
        (models / directory.relative_to(baked)).mkdir(exist_ok=True)
    for source, destination, identity, duplicate in files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if duplicate is not None:
            if (_identity(source), _identity(destination)) != (identity, duplicate) or _digest(source) != _digest(destination):
                raise RuntimeError(f'Model changed during deduplication; preserving both: {source.name}')
            source.unlink()
        else:
            _move_model(source, destination, identity)
    for placeholder, identity in placeholders:
        if _identity(placeholder) != identity:
            raise RuntimeError('A model placeholder changed; preserving the model directory.')
        placeholder.unlink()
    for directory in reversed(directories):
        directory.rmdir()
    if baked.exists():
        baked.rmdir()  # Abort safely if a new upload appeared during migration.
    baked.symlink_to(models, target_is_directory=True)
