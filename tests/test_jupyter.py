import asyncio
import os
from pathlib import Path
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fibo_runtime import jupyter


TOKEN = "provider-generated-jupyter-token"


def workspace(tmp_path):
    data = tmp_path / "workspace" / "ComfyUI"
    data.mkdir(parents=True)
    return data


def test_launch_keeps_secrets_out_of_argv_and_private_config(tmp_path, monkeypatch):
    data = workspace(tmp_path)
    source = {"JUPYTER_PASSWORD": TOKEN, "FIBO_PASSWORD": "private-comfy-password",
              "FILEBROWSER_PASSWORD": "private-browser-password", "JUPYTER_DISABLE_AUTH": "true",
              "JUPYTER_TOKEN": "stale-token", "JUPYTER_RUNTIME_DIR": "/unrelated/runtime",
              "UNRELATED_OPTION": "retained"}
    original = source.copy()
    child = object()
    launch = AsyncMock(return_value=child)
    monkeypatch.setattr(jupyter.asyncio, "create_subprocess_exec", launch)
    assert asyncio.run(jupyter.start_jupyter(data, environ=source)) is child
    args, kwargs = launch.call_args
    assert args[0] == "/opt/jupyter/bin/jupyter-lab" and args[1] == "--config" and len(args) == 3
    config = Path(args[2])
    assert config.is_file() and TOKEN not in config.read_text()
    assert TOKEN not in " ".join(args)
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert stat.S_IMODE(config.parent.stat().st_mode) == 0o700
    runtime = config.parent / "runtime"
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o700
    assert kwargs["env"]["JUPYTER_TOKEN"] == TOKEN
    assert kwargs["env"]["JUPYTER_RUNTIME_DIR"] == str(runtime)
    assert kwargs["env"]["UNRELATED_OPTION"] == "retained"
    assert not (jupyter.SECRET_VARIABLES & kwargs["env"].keys())
    assert kwargs["umask"] == 0o077 and source == original


def test_actual_config_enables_token_auth_on_workspace_and_standard_port(tmp_path, monkeypatch):
    data = workspace(tmp_path)
    command, environment = jupyter.prepare_jupyter(data, environ={"JUPYTER_PASSWORD": TOKEN})
    config = SimpleNamespace(ServerApp=SimpleNamespace(), IdentityProvider=SimpleNamespace())
    monkeypatch.setenv("JUPYTER_TOKEN", environment["JUPYTER_TOKEN"])
    content = Path(command[2]).read_text()
    exec(compile(content, command[2], "exec"), {"get_config": lambda: config})
    assert config.ServerApp.ip == "0.0.0.0" and config.ServerApp.port == 8888
    assert config.ServerApp.port_retries == 0 and config.ServerApp.allow_root is True
    assert config.ServerApp.root_dir == str(data.parent)
    assert config.ServerApp.open_browser is False and config.ServerApp.log_level == "WARNING"
    assert config.IdentityProvider.token == TOKEN
    # Rotating the environment token does not require writing a token to disk.
    monkeypatch.setenv("JUPYTER_TOKEN", "replacement-provider-token")
    exec(compile(content, command[2], "exec"), {"get_config": lambda: config})
    assert config.IdentityProvider.token == "replacement-provider-token"
    assert Path(command[2]).read_text() == content


@pytest.mark.parametrize("environment", [
    {"FIBO_PASSWORD": TOKEN},
    {"JUPYTER_PASSWORD": "short"},
    {"JUPYTER_PASSWORD": "{{ RUNPOD_SECRET_missing_token }}"},
    {"JUPYTER_PASSWORD": "provider-token-with-newline\n"},
])
def test_missing_or_invalid_provider_token_cannot_launch(tmp_path, monkeypatch, environment):
    data = workspace(tmp_path)
    launch = AsyncMock()
    monkeypatch.setattr(jupyter.asyncio, "create_subprocess_exec", launch)
    with pytest.raises(ValueError, match="JUPYTER_PASSWORD"):
        asyncio.run(jupyter.start_jupyter(data, environ=environment))
    launch.assert_not_called()
    assert not (data / "state").exists()


def test_custom_workspace_and_binary_are_passed_as_paths(tmp_path):
    data = workspace(tmp_path)
    served = tmp_path / "custom workspace's root"
    served.mkdir()
    command, _ = jupyter.prepare_jupyter(data, root=served, binary="/isolated/bin/jupyter-lab",
                                         environ={"JUPYTER_PASSWORD": TOKEN})
    assert command[0] == "/isolated/bin/jupyter-lab"
    config = SimpleNamespace(ServerApp=SimpleNamespace(), IdentityProvider=SimpleNamespace())
    previous = os.environ.get("JUPYTER_TOKEN")
    try:
        os.environ["JUPYTER_TOKEN"] = TOKEN
        exec(Path(command[2]).read_text(), {"get_config": lambda: config})
    finally:
        if previous is None:
            os.environ.pop("JUPYTER_TOKEN", None)
        else:
            os.environ["JUPYTER_TOKEN"] = previous
    assert config.ServerApp.root_dir == str(served)


def test_symlinked_state_directory_is_preserved(tmp_path):
    data = workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (data / "state").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        jupyter.prepare_jupyter(data, environ={"JUPYTER_PASSWORD": TOKEN})
    assert (data / "state").is_symlink() and list(outside.iterdir()) == []


def test_symlinked_config_is_preserved_without_changing_target(tmp_path):
    data = workspace(tmp_path)
    state = data / "state" / "jupyter"
    state.mkdir(parents=True)
    target = tmp_path / "user-config.py"
    target.write_text("keep this user config")
    config = state / "jupyter_server_config.py"
    config.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        jupyter.prepare_jupyter(data, environ={"JUPYTER_PASSWORD": TOKEN})
    assert config.is_symlink() and target.read_text() == "keep this user config"


def test_nonregular_config_destination_is_preserved(tmp_path):
    data = workspace(tmp_path)
    config = data / "state" / "jupyter" / "jupyter_server_config.py"
    config.mkdir(parents=True)
    (config / "user-data").write_text("keep")
    with pytest.raises(ValueError, match="invalid Jupyter configuration"):
        jupyter.prepare_jupyter(data, environ={"JUPYTER_PASSWORD": TOKEN})
    assert (config / "user-data").read_text() == "keep"


def test_provider_proxy_origin_is_specific_while_token_auth_remains_enabled(tmp_path, monkeypatch):
    identifier = "portspod1234"
    command, environment = jupyter.prepare_jupyter(workspace(tmp_path), environ={
        "JUPYTER_PASSWORD": TOKEN, "RUNPOD_POD_ID": identifier})
    monkeypatch.setenv("JUPYTER_TOKEN", environment["JUPYTER_TOKEN"])
    config = SimpleNamespace(ServerApp=SimpleNamespace(), IdentityProvider=SimpleNamespace())
    exec(Path(command[2]).read_text(), {"get_config": lambda: config})
    assert config.ServerApp.allow_remote_access is True and config.ServerApp.trust_xheaders is True
    assert config.ServerApp.allow_origin == "https://" + identifier + "-8888.proxy.runpod.net"
    assert config.IdentityProvider.token == TOKEN
