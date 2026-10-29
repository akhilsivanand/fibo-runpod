"""Install explicitly bundled workflows, updating only unchanged managed files."""
import hashlib
import json
from pathlib import Path
import os
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as f:
        temporary = Path(f.name)
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    try:
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def install_workflows(source_root, data_root):
    source_root, data_root = Path(source_root), Path(data_root)
    config = json.loads((source_root / 'config/workflows.json').read_text())
    state_file = data_root / 'state/managed-workflows.json'
    previous = json.loads(state_file.read_text()) if state_file.is_file() else {}
    current = dict(previous)
    result = {'installed': [], 'updated': [], 'preserved': []}
    entries = list(config['presets'].values()) + list(config.get('ui_only', {}).values())
    for entry in entries:
        source = (source_root / entry['ui']).resolve()
        source.relative_to(source_root.resolve())
        content = source.read_bytes()
        json.loads(content)
        key = 'user/default/workflows/FIBO/' + source.name
        target = data_root / key
        if target.is_symlink() or target.parent.resolve().is_relative_to(data_root.resolve()) is False:
            result['preserved'].append(key)
            continue
        new_hash = hashlib.sha256(content).hexdigest()
        if target.exists():
            if not target.is_file():
                result['preserved'].append(key)
                continue
            actual = digest(target)
            if actual == new_hash:
                current[key] = new_hash
                continue
            if previous.get(key) != actual:
                result['preserved'].append(key)
                continue
            result['updated'].append(key)
        else:
            result['installed'].append(key)
        atomic_write(target, content)
        current[key] = new_hash
    # A package upgrade never deletes a user's older workflow.
    atomic_write(state_file, (json.dumps(current, indent=2) + '\n').encode())
    return result
