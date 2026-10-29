"""Run the standard File Browser service with private bootstrap credentials."""
import asyncio
import json
import os
from pathlib import Path

from .workflows import atomic_write

FILEBROWSER_VERSION = "2.63.23"
FILEBROWSER_BINARY = "/usr/local/bin/filebrowser"
SECRET_VARIABLES = {
    "FILEBROWSER_PASSWORD", "JUPYTER_PASSWORD", "JUPYTER_TOKEN", "JUPYTER_DISABLE_AUTH", "FIBO_PASSWORD",
}


def _password_hash(secret):
    # bcrypt is already part of the pinned runtime inventory. File Browser's
    # FB_PASSWORD option takes a bcrypt hash, rather than a plaintext password.
    import bcrypt
    return bcrypt.hashpw(secret.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def _initial_secret(environment):
    for name in ("FILEBROWSER_PASSWORD", "JUPYTER_PASSWORD", "FIBO_PASSWORD"):
        value = environment.get(name, "")
        if not value:
            continue
        if (not isinstance(value, str) or "RUNPOD_SECRET_" in value.upper()
                or any(ord(character) < 32 or ord(character) == 127 for character in value)):
            raise ValueError(name + " must be a resolved password without control characters.")
        try:
            encoded = value.encode("utf-8")
        except UnicodeError:
            raise ValueError(name + " must contain valid UTF-8 text.") from None
        if not 12 <= len(encoded) <= 72:
            raise ValueError(name + " must contain between 12 and 72 UTF-8 bytes for File Browser.")
        return value
    raise ValueError("File Browser needs FILEBROWSER_PASSWORD or Runpod's JUPYTER_PASSWORD on first setup.")


def prepare_filebrowser(data_root, *, root=None, binary=FILEBROWSER_BINARY, environ=None):
    """Return a secret-free command and a child environment containing only a hash.

    Existing user credentials are retained. FILEBROWSER_PASSWORD, otherwise the
    initial Jupyter token, seeds the admin account only when the DB is new.
    """
    data = Path(data_root).resolve()
    served_root = Path(root).resolve() if root is not None else data.parent
    if not served_root.is_dir():
        raise ValueError("File Browser's workspace root must already exist.")
    state = data / "state" / "filebrowser"
    for directory in (data / "state", state):
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError("Preserving an existing invalid File Browser state path.")
    state.mkdir(parents=True, mode=0o700, exist_ok=True)
    state.chmod(0o700)
    config, database = state / "config.json", state / "filebrowser.db"
    for path in (config, database):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("Preserving an existing invalid File Browser file.")
    source = dict(os.environ if environ is None else environ)
    environment = {key: value for key, value in source.items()
                   if key not in SECRET_VARIABLES and not key.startswith("FB_")}
    if not database.exists():
        environment["FB_PASSWORD"] = _password_hash(_initial_secret(source))
    # An explicit config prevents accidental discovery of a user's unrelated
    # .filebrowser.json. Runtime flags are deliberately fixed to standard ports.
    settings = {
        "address": "0.0.0.0", "port": 8080, "root": str(served_root),
        "username": "admin", "noauth": False, "log": "",
        "disableExec": True, "followExternalSymlinks": False,
    }
    atomic_write(config, (json.dumps(settings, indent=2) + "\n").encode())
    config.chmod(0o600)
    command = [str(binary), "--config", str(config), "--database", str(database)]
    return command, environment


async def start_filebrowser(data_root, *, root=None, binary=FILEBROWSER_BINARY, environ=None):
    """Start File Browser for the bootstrap supervisor to track and terminate."""
    command, environment = prepare_filebrowser(data_root, root=root, binary=binary, environ=environ)
    return await asyncio.create_subprocess_exec(*command, env=environment, umask=0o077)
