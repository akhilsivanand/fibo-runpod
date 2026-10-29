"""Prepare revision-pinned model assets without importing the GPU runtime.

Final files are immutable to this installer. A mismatching file or any symlink
is preserved and reported as an error. Only partials accompanied by this
installer's matching ownership receipt may be resumed or quarantined.
"""

from contextlib import contextmanager
from decimal import Decimal, DecimalException, InvalidOperation
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import time
from urllib.parse import urlsplit
import uuid


HEADROOM_BYTES = 1024 ** 3
CURL_ATTEMPTS = 3
CURL_MAX_SECONDS = 3600


class AssetError(RuntimeError):
    """Asset preparation failed without replacing a user-owned final file."""


def _load_manifest(path):
    try:
        with Path(path).open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (ValueError, UnicodeError) as error:
        raise AssetError("The model manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise AssetError("Unsupported model manifest schema")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise AssetError("The model manifest must contain models")
    ids, paths = set(), set()
    for model in models:
        if not isinstance(model, dict):
            raise AssetError("Invalid model manifest entry")
        identifier, relative = model.get("id"), model.get("path")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", identifier):
            raise AssetError("Invalid model identifier")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise AssetError(f"Invalid model path for {identifier}")
        parts = relative.split("/")
        if PurePosixPath(relative).is_absolute() or any(
            part in ("", ".", "..") or any(ord(char) < 32 for char in part) for part in parts
        ):
            raise AssetError(f"Unsafe model path for {identifier}")
        if identifier in ids or relative in paths:
            raise AssetError("Duplicate model identifier or destination")
        ids.add(identifier)
        paths.add(relative)
        size = model.get("size")
        if type(size) is not int or size <= 0:
            raise AssetError(f"Invalid model size for {identifier}")
        if not isinstance(model.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", model["sha256"]):
            raise AssetError(f"Invalid model checksum for {identifier}")
        url = model.get("url")
        if not isinstance(url, str):
            raise AssetError(f"Invalid download URL for {identifier}")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or any(ord(char) < 33 for char in url):
            raise AssetError(f"A public HTTPS download URL is required for {identifier}")
        if not isinstance(model.get("license"), str) or not model["license"]:
            raise AssetError(f"Missing model license for {identifier}")
        if not isinstance(model.get("source"), dict):
            raise AssetError(f"Missing model provenance for {identifier}")
        profiles = model.get("profiles")
        if not isinstance(profiles, list) or not profiles or any(not isinstance(item, str) or not item for item in profiles):
            raise AssetError(f"Missing model profiles for {identifier}")
    total = sum(model["size"] for model in models)
    for relative in paths:
        if any(str(parent) in paths for parent in PurePosixPath(relative).parents):
            raise AssetError("Model destinations cannot contain other model destinations")
    if "total_size_bytes" in manifest and manifest["total_size_bytes"] != total:
        raise AssetError("Model manifest total does not match its entries")
    return models


def _checked_stat(path):
    """Reject symlink components, including dangling links, before accessing data."""
    path = Path(path)
    for component in [*reversed(path.parents), path]:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise AssetError(f"Refusing to follow an existing symlink: {component}")
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _mkdir(path):
    _checked_stat(path)
    Path(path).mkdir(parents=True, exist_ok=True)
    metadata = _checked_stat(path)
    if not stat.S_ISDIR(metadata.st_mode):
        raise AssetError(f"Expected a directory: {path}")


def _sha256(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise AssetError(f"Expected a regular file: {path}")
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _verify_final(destination, model):
    metadata = _checked_stat(destination)
    if metadata is None:
        return False
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != model["size"] or _sha256(destination) != model["sha256"]:
        raise AssetError(f"Existing model differs; preserving it: {model['path']}")
    return True


def _partial_paths(destination):
    partial = destination.with_name(destination.name + ".part")
    return partial, partial.with_name(partial.name + ".fibo.json")


def _receipt_matches(receipt, model):
    return isinstance(receipt, dict) and receipt.get("schema_version") == 1 and all(
        receipt.get(key) == model[key] for key in ("id", "path", "size", "sha256")
    ) and type(receipt.get("device")) is int and type(receipt.get("inode")) is int


def _owned_partial(destination, model):
    partial, marker = _partial_paths(destination)
    metadata, marker_metadata = _checked_stat(partial), _checked_stat(marker)
    if metadata is None and marker_metadata is None:
        return None
    if marker_metadata is None or not stat.S_ISREG(marker_metadata.st_mode) or marker_metadata.st_size > 4096:
        raise AssetError(f"Unrecognized partial file; preserving it: {model['path']}")
    try:
        descriptor = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r", encoding="utf-8") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise AssetError(f"Unrecognized partial ownership record: {model['path']}")
            receipt = json.load(source)
    except (ValueError, UnicodeError) as error:
        raise AssetError(f"Unrecognized partial ownership record: {model['path']}") from error
    if not _receipt_matches(receipt, model):
        raise AssetError(f"Unrecognized partial ownership record: {model['path']}")
    if metadata is None:
        return None  # A recognized stale receipt can be replaced when downloading.
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or (
        metadata.st_dev, metadata.st_ino
    ) != (receipt["device"], receipt["inode"]):
        raise AssetError(f"Partial file identity changed; preserving it: {model['path']}")
    return metadata


def _write_new_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".fibo-", delete=False) as source:
            temporary = Path(source.name)
            json.dump(value, source, sort_keys=True)
            source.flush()
            os.fsync(source.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _claim_partial(destination, model):
    partial, marker = _partial_paths(destination)
    existing = _owned_partial(destination, model)
    if existing is not None:
        return partial
    # Remove only a matching, stale receipt. Unrecognized markers were rejected.
    if _checked_stat(marker) is not None:
        marker.unlink()
    descriptor = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    metadata = os.fstat(descriptor)
    os.close(descriptor)
    receipt = {key: model[key] for key in ("id", "path", "size", "sha256")}
    receipt.update(schema_version=1, device=metadata.st_dev, inode=metadata.st_ino)
    try:
        _write_new_json(marker, receipt)
    except BaseException:
        current = partial.lstat()
        if (current.st_dev, current.st_ino) == (metadata.st_dev, metadata.st_ino):
            partial.unlink()
        raise
    return partial


def _scan(models, models_root, on_verify=None):
    plan = []
    for model in models:
        if on_verify is not None:
            on_verify(model["id"])
        destination = models_root / model["path"]
        if _verify_final(destination, model):
            plan.append({"model": model, "destination": destination, "remaining": 0, "verified": True})
            continue
        partial = _owned_partial(destination, model)
        size = partial.st_size if partial is not None else 0
        corrupted = size > model["size"] or (size == model["size"] and _sha256(_partial_paths(destination)[0]) != model["sha256"])
        plan.append({
            "model": model, "destination": destination, "verified": False,
            "remaining": model["size"] if corrupted else model["size"] - size,
        })
    return plan


def _used_tree_bytes(root):
    """Use file metadata only; do not follow links or read user file contents."""
    if not root.exists():
        return 0
    seen, total, directories = set(), 0, [root]
    while directories:
        with os.scandir(directories.pop()) as entries:
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    directories.append(Path(entry.path))
                elif stat.S_ISREG(metadata.st_mode):
                    identity = metadata.st_dev, metadata.st_ino
                    if identity not in seen:
                        seen.add(identity)
                        total += metadata.st_size
    return total


def _check_capacity(plan, data_root):
    filesystems = {}
    for item in plan:
        if not item["remaining"]:
            continue
        probe = item["destination"].parent
        while not probe.exists():
            probe = probe.parent
        _checked_stat(probe)
        device = probe.stat().st_dev
        filesystems.setdefault(device, [probe, 0])[1] += item["remaining"]
    for probe, pending in filesystems.values():
        if shutil.disk_usage(probe).free < pending + HEADROOM_BYTES:
            raise AssetError(f"Insufficient filesystem capacity; pending models need {pending} bytes plus {HEADROOM_BYTES} bytes headroom")
    quota = os.environ.get("FIBO_VOLUME_GB")
    if quota is not None:
        try:
            gigabytes = Decimal(quota)
            if not gigabytes.is_finite() or gigabytes <= 0:
                raise InvalidOperation
            capacity = int(gigabytes * 1_000_000_000)
        except (DecimalException, ValueError, OverflowError) as error:
            raise AssetError("FIBO_VOLUME_GB must be a positive, finite number of decimal GB") from error
        pending = sum(item["remaining"] for item in plan)
        if _used_tree_bytes(data_root) + pending + HEADROOM_BYTES > capacity:
            raise AssetError(f"Configured volume capacity is insufficient; pending models need {pending} bytes plus headroom")


@contextmanager
def _model_lock(data_root):
    state = data_root / "state"
    _mkdir(state)
    lock = state / "models.lock"
    metadata = _checked_stat(lock)
    if metadata is not None and not stat.S_ISREG(metadata.st_mode):
        raise AssetError("Model lock path is not a regular file")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _run_curl(model, partial, progress):
    """Retry failed transfers using the newly written offset, never a shell."""
    if shutil.which("curl") is None:
        raise AssetError("curl is required to download missing models")
    for attempt in range(CURL_ATTEMPTS):
        metadata = _owned_partial(partial.with_name(partial.name[:-5]), model)
        if metadata is None:
            raise AssetError(f"Managed partial disappeared: {model['id']}")
        offset = metadata.st_size
        if offset >= model["size"]:
            return  # The caller verifies complete files, even after a transfer error.
        descriptor = os.open(partial, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "ab", buffering=0) as output:
            opened = os.fstat(output.fileno())
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise AssetError(f"Partial file identity changed: {model['id']}")
            command = [
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--proto", "=https", "--proto-redir", "=https", "--max-redirs", "10",
                "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
                "--max-time", str(CURL_MAX_SECONDS), "--max-filesize", str(model["size"]),
                "--continue-at", str(offset),
                "--output", "-", model["url"],
            ]
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.PIPE)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if time.monotonic() - started > CURL_MAX_SECONDS + 30:
                        process.kill()
                        raise AssetError(f"Download time limit exceeded: {model['id']}")
                    progress(os.fstat(output.fileno()).st_size)
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        continue
                _, _ = process.communicate()
                output.flush()
                os.fsync(output.fileno())
                if process.returncode == 0:
                    return
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if process.stderr is not None:
                    process.stderr.close()
        if attempt + 1 < CURL_ATTEMPTS:
            progress(_owned_partial(partial.with_name(partial.name[:-5]), model).st_size)
    metadata = _owned_partial(partial.with_name(partial.name[:-5]), model)
    if metadata is not None and metadata.st_size >= model["size"]:
        return
    raise AssetError(f"Download failed after {CURL_ATTEMPTS} attempts: {model['id']}")


def _quarantine(destination, model):
    metadata = _owned_partial(destination, model)
    if metadata is None:
        raise AssetError("Cannot quarantine an unrecognized partial")
    partial, marker = _partial_paths(destination)
    quarantine = partial.with_name(partial.name + ".corrupt." + uuid.uuid4().hex)
    os.link(partial, quarantine, follow_symlinks=False)
    partial.unlink()
    os.link(marker, quarantine.with_name(quarantine.name + ".fibo.json"), follow_symlinks=False)
    marker.unlink()


def _publish(destination, model):
    metadata = _owned_partial(destination, model)
    partial, marker = _partial_paths(destination)
    if metadata is None or metadata.st_size != model["size"] or _sha256(partial) != model["sha256"]:
        raise AssetError(f"Model verification failed: {model['id']}")
    partial.chmod(0o644)
    try:
        os.link(partial, destination, follow_symlinks=False)
    except FileExistsError:
        _verify_final(destination, model)
    partial.unlink()
    marker.unlink()


def prepare_models(manifest_path, data_root, progress_callback=None, download=True):
    """Verify/install all manifest models under ``data_root/models``.

    ``download=False`` validates, hashes existing files, and preflights capacity
    without writing files. ``FIBO_VOLUME_GB`` optionally sets a decimal-GB
    capacity for the whole data tree; used bytes are obtained from metadata.
    The callback receives dictionaries safe to show on a startup status page.
    """
    models = _load_manifest(manifest_path)
    data_root = Path(os.path.abspath(Path(data_root).expanduser()))
    _checked_stat(data_root)
    models_root = data_root / "models"
    total = sum(model["size"] for model in models)
    complete = 0

    def emit(phase, detail, current_model=None, current_bytes=None):
        if progress_callback is not None:
            progress_callback({
                "phase": phase, "detail": detail, "bytes_total": total,
                "bytes_complete": min(total, complete if current_bytes is None else current_bytes),
                "current_model": current_model,
            })

    def prepare():
        nonlocal complete
        emit("verifying", "Checking existing model files")
        plan = _scan(models, models_root, lambda identifier: emit("verifying", f"Checking {identifier}", identifier))
        pending = sum(item["remaining"] for item in plan)
        emit("checking_storage", "Checking capacity for all pending model files")
        _check_capacity(plan, data_root)
        result = {
            "models_root": str(models_root), "bytes_total": total, "bytes_pending": pending,
            "models": [], "downloaded": 0, "verified": 0,
        }
        if not download:
            result["models"] = [{"id": item["model"]["id"], "path": str(item["destination"]), "status": "verified" if item["verified"] else "pending"} for item in plan]
            result["verified"] = sum(item["verified"] for item in plan)
            emit("planned", "Model preparation preview complete")
            return result
        for item in plan:
            model, destination = item["model"], item["destination"]
            if item["verified"]:
                complete += model["size"]
                result["verified"] += 1
                status = "verified"
            else:
                _mkdir(destination.parent)
                corruptions = 0
                while True:
                    partial = _claim_partial(destination, model)
                    metadata = _owned_partial(destination, model)
                    if metadata.st_size < model["size"]:
                        emit("downloading", f"Downloading {model['id']}", model["id"], complete + metadata.st_size)
                        _run_curl(model, partial, lambda count: emit("downloading", f"Downloading {model['id']}", model["id"], complete + min(count, model["size"])))
                    metadata = _owned_partial(destination, model)
                    emit("verifying", f"Verifying {model['id']}", model["id"])
                    if metadata.st_size == model["size"] and _sha256(partial) == model["sha256"]:
                        _publish(destination, model)
                        break
                    _quarantine(destination, model)
                    corruptions += 1
                    emit("recovering", f"Quarantined invalid managed partial for {model['id']}", model["id"])
                    if corruptions > 1:
                        raise AssetError(f"Model verification failed after one clean retry: {model['id']}")
                    _check_capacity(_scan(models, models_root), data_root)
                complete += model["size"]
                result["downloaded"] += 1
                status = "installed"
            result["models"].append({"id": model["id"], "path": str(destination), "status": status})
            emit("verified", f"Model ready: {model['id']}", model["id"])
        emit("models_ready", "All model files are verified")
        return result

    if not download:
        return prepare()
    emit("waiting_for_models", "Waiting for the model preparation lock")
    with _model_lock(data_root):
        return prepare()
