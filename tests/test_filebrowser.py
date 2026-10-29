import json
from pathlib import Path

import pytest

from fibo_runtime import filebrowser


def test_new_database_hashes_bootstrap_secret_without_argv_or_plaintext_files(tmp_path, monkeypatch):
    secret = "unique-first-run-token"
    hashed = "$2b$12$example-hashed-credential"
    seen = []
    monkeypatch.setattr(filebrowser, "_password_hash", lambda value: seen.append(value) or hashed)
    data = tmp_path / "ComfyUI"
    command, env = filebrowser.prepare_filebrowser(data, environ={
        "JUPYTER_PASSWORD": secret, "FIBO_PASSWORD": "legacy-private-token",
        "JUPYTER_DISABLE_AUTH": "true", "FB_NOAUTH": "true", "FB_ROOT": "/",
        "PATH": "/usr/bin",
    })
    assert seen == [secret]
    assert secret not in " ".join(command)
    assert env == {"PATH": "/usr/bin", "FB_PASSWORD": hashed}
    config = Path(command[2])
    settings = json.loads(config.read_text())
    assert settings["root"] == str(tmp_path)
    assert settings["address"] == "0.0.0.0" and settings["port"] == 8080
    assert settings["noauth"] is False and settings["disableExec"] is True
    assert settings["followExternalSymlinks"] is False
    assert secret not in config.read_text()
    assert config.stat().st_mode & 0o777 == 0o600
    assert config.parent.stat().st_mode & 0o777 == 0o700


def test_explicit_filebrowser_password_has_priority(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(filebrowser, "_password_hash", lambda value: seen.append(value) or "hash")
    filebrowser.prepare_filebrowser(tmp_path / "ComfyUI", environ={
        "FILEBROWSER_PASSWORD": "explicit-file-password", "JUPYTER_PASSWORD": "provider-connect-token",
    })
    assert seen == ["explicit-file-password"]


def test_existing_database_keeps_user_credentials_and_does_not_require_seed(tmp_path, monkeypatch):
    data = tmp_path / "ComfyUI"
    state = data / "state/filebrowser"
    state.mkdir(parents=True)
    database = state / "filebrowser.db"
    database.write_bytes(b"existing-user-data")
    monkeypatch.setattr(filebrowser, "_password_hash", lambda _: pytest.fail("Existing password was reset"))
    _, env = filebrowser.prepare_filebrowser(data, environ={"FB_PASSWORD": "untrusted-override"})
    assert "FB_PASSWORD" not in env
    assert database.read_bytes() == b"existing-user-data"


@pytest.mark.parametrize("secret", ["short", "x" * 73, "wrong\npassword", "{{ RUNPOD_SECRET_missing }}"])
def test_invalid_first_password_fails_before_hashing(tmp_path, monkeypatch, secret):
    monkeypatch.setattr(filebrowser, "_password_hash", lambda _: pytest.fail("Invalid password was hashed"))
    with pytest.raises(ValueError):
        filebrowser.prepare_filebrowser(tmp_path / "ComfyUI", environ={"JUPYTER_PASSWORD": secret})


def test_symlink_state_is_preserved(tmp_path):
    data = tmp_path / "ComfyUI"
    state = data / "state"
    state.mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    (state / "filebrowser").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="Preserving"):
        filebrowser.prepare_filebrowser(data, environ={})
