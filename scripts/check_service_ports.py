#!/usr/bin/env python3
"""Exercise the built image's real Pod services on CPU, without model downloads.

Run only in a fresh, disposable container. This check starts the actual Pod
entrypoint, Runpod SSH initialization, supervised JupyterLab, FileBrowser and ComfyUI.
The child-only shim bypasses GPU/model preparation, starts ComfyUI in CPU/offline
mode, and accepts native node readiness without downloaded weights. No prompt
is executed. This verifies service ports, not GPU inference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote
import uuid

import aiohttp
from yarl import URL


PORTS = (22, 8080, 8188, 8189, 8888)
MAX_BODY = 4 * 1024 * 1024
RAW_SECRET_KEYS = ("FIBO_PASSWORD", "JUPYTER_PASSWORD", "JUPYTER_TOKEN", "FILEBROWSER_PASSWORD")


class PortCheckError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise PortCheckError(message)


def _port_preflight():
    for port in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as available:
            try:
                available.bind(("127.0.0.1", port))
            except OSError as error:
                raise PortCheckError(f"Port {port} is already in use; use a fresh disposable container") from error


def _pids():
    return {int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit()}


def _stop(process, previous_pids):
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    # sshd daemonizes into its own session. Only stop sshd processes created
    # after this check's baseline, never a listener that existed before it.
    for pid in _pids() - previous_pids:
        try:
            if Path("/proc", str(pid), "comm").read_text().strip() == "sshd":
                os.kill(pid, signal.SIGTERM)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass


def _child_shim(real_python):
    return f"#!{real_python}\n" + r'''
import asyncio
import json
import os
from pathlib import Path
import sys

REAL_PYTHON = sys.executable
if sys.argv[1:3] != ['-m', 'fibo_runtime.bootstrap']:
    os.execv(REAL_PYTHON, [REAL_PYTHON, *sys.argv[1:]])

from fibo_runtime import bootstrap
bootstrap.check_gpu = lambda: None

def no_models(manifest, data, progress_callback=None, **kwargs):
    if progress_callback:
        progress_callback({'phase':'models_ready', 'detail':'CPU service-port check; no weights loaded',
                           'bytes_total':0, 'bytes_complete':0})
    return {'models':[], 'downloaded':0, 'verified':0, 'bytes_total':0}

bootstrap.prepare_models = no_models
original_spawn = asyncio.create_subprocess_exec
processes = {}

async def cpu_spawn(*command, **kwargs):
    kind = None
    if str(Path(os.environ['COMFY_ROOT']) / 'main.py') in command:
        kind = 'comfy'
        command = (*command, '--cpu', '--offline', '--disable-all-custom-nodes',
                   '--whitelist-custom-nodes', 'fibo')
    elif command and Path(str(command[0])).name == 'filebrowser':
        kind = 'filebrowser'
    process = await original_spawn(*command, **kwargs)
    if kind:
        processes[kind] = process.pid
        Path(os.environ['FIBO_PORT_CHECK_PROCESSES']).write_text(json.dumps(processes))
    return process

asyncio.create_subprocess_exec = cpu_spawn

async def wait_cpu_comfy(process, timeout=180):
    import aiohttp
    deadline = asyncio.get_running_loop().time() + timeout
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                raise RuntimeError('CPU ComfyUI exited before native node readiness')
            try:
                async with session.get('http://127.0.0.1:8189/object_info/KSampler') as response:
                    if response.status == 200 and 'KSampler' in await response.json():
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                pass
            await asyncio.sleep(0.25)
    raise RuntimeError('CPU ComfyUI native node readiness timed out')

bootstrap.wait_for_comfy = wait_cpu_comfy
sys.argv = ['fibo_runtime.bootstrap', *sys.argv[3:]]
bootstrap.main()
'''


async def _request(session, url, method="GET", headers=None, payload=None):
    async with session.request(method, url, headers=headers, json=payload,
                               allow_redirects=False) as response:
        body = await response.content.read(MAX_BODY + 1)
        require(len(body) <= MAX_BODY, "A service response exceeded the check's size limit")
        return response.status, body


async def _ssh_banner():
    reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", 22), 3)
    try:
        banner = await asyncio.wait_for(reader.readline(), 3)
        return banner.startswith(b"SSH-2.0-")
    finally:
        writer.close()
        await writer.wait_closed()


async def _gateway_login(session, token):
    origin = "http://127.0.0.1:8188"
    status, body = await _request(session, origin + "/")
    require(status == 401 and b"/fibo/login" in body and b"<form" in body.lower(),
            "The ComfyUI gateway did not require its token login form")
    async with session.post(origin + "/fibo/login", data={"token": token},
                            headers={"Origin": origin}, allow_redirects=False) as response:
        body = await response.content.read(MAX_BODY + 1)
        require(len(body) <= MAX_BODY, "The token login response exceeded the check's size limit")
        require(response.status == 303 and response.headers.get("Location") == "/",
                "The ComfyUI token login did not redirect to a clean URL")
        cookie = response.cookies.get("__Host-fibo-session")
        require(cookie is not None and cookie["secure"] and cookie["httponly"]
                and cookie["samesite"].lower() == "strict" and cookie["path"] == "/",
                "The ComfyUI login cookie is missing its required protection flags")


async def _services_ready(process, workspace, token, timeout):
    deadline = time.monotonic() + timeout
    relative_probe = "port-check.txt"
    jupyter = "http://127.0.0.1:8888/api/contents/" + relative_probe + "?content=1"
    auth = {"Authorization": "token " + token}
    last_stage = "entrypoint startup"
    cookie_jar = aiohttp.CookieJar(unsafe=True,
        treat_as_secure_origin=[URL("http://127.0.0.1:8188"), URL("ws://127.0.0.1:8188")])
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4),
                                    cookie_jar=cookie_jar, trust_env=False) as session:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise PortCheckError(f"Pod entrypoint exited during startup (code {process.returncode})")
            try:
                last_stage = "ComfyUI gateway readiness on 8188"
                status, body = await _request(session, "http://127.0.0.1:8188/readyz")
                require(status == 200 and json.loads(body).get("ready") is True, last_stage)
                last_stage = "Jupyter token authentication on 8888"
                status, body = await _request(session, jupyter, headers=auth)
                require(status == 200 and json.loads(body).get("content") == "fibo-service-port-check\n", last_stage)
                last_stage = "FileBrowser initialization on 8080"
                status, body = await _request(session, "http://127.0.0.1:8080/api/login", method="POST",
                    payload={"username": "admin", "password": token, "recaptcha": ""})
                require(status == 200 and re.fullmatch(rb"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", body), last_stage)
                browser_token = body.decode("ascii")
                last_stage = "SSH listener on 22"
                require(await _ssh_banner(), last_stage)
                break
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, PortCheckError):
                await asyncio.sleep(0.5)
        else:
            raise PortCheckError(f"Service startup exceeded {timeout:g} seconds at: {last_stage}")
        await _gateway_login(session, token)
        return await _check_services(session, workspace, token, browser_token, jupyter)


async def _check_services(session, workspace, token, browser_token, jupyter_url):
    origin = "http://127.0.0.1:8188"
    status, body = await _request(session, origin + "/")
    require(status == 200 and b"Fibo" in body and b"fibo2-edit-turbo" in body,
            "Port 8188 did not serve the Fibo workflow landing page")
    status, body = await _request(session, origin + "/?editor=1")
    require(status == 200 and b"<html" in body.lower() and b"comfy" in body.lower(),
            "Port 8188 did not proxy the native ComfyUI editor")
    root = Path(os.environ.get("FIBO_ROOT", "/opt/fibo"))
    status, body = await _request(session, origin + "/api/workflow_templates/fibo/fibo2-edit-turbo.json")
    require(status == 200 and json.loads(body) == json.loads((root / "workflows/ui/fibo2-edit-turbo.json").read_text()),
            "Port 8188 did not serve the packaged native workflow")
    status, _ = await _request(session, origin + "/api/prompt", method="POST", headers={"Origin": origin},
                                payload={"prompt": {}, "client_id": uuid.uuid4().hex})
    require(status == 400, "Same-origin HTTP POST did not reach ComfyUI's empty-prompt validation")
    async with session.ws_connect(origin.replace("http:", "ws:") + "/ws?clientId=" + uuid.uuid4().hex,
                                  origin=origin) as websocket:
        require(not websocket.closed, "Port 8188 WebSocket did not upgrade")
    status, _ = await _request(session, origin + "/api/workflow_templates", headers={"Origin": "https://foreign.example"})
    require(status == 403, "The ComfyUI gateway accepted a foreign HTTP Origin")
    try:
        async with session.ws_connect(origin.replace("http:", "ws:") + "/ws", origin="https://foreign.example"):
            raise PortCheckError("The ComfyUI gateway accepted a foreign WebSocket Origin")
    except aiohttp.WSServerHandshakeError as error:
        require(error.status == 403, "A foreign WebSocket Origin returned an unexpected response")
    # Jupyter may set a session cookie after the authenticated readiness probe.
    # Use a separate empty cookie jar to test the anonymous boundary itself.
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4),
                                    cookie_jar=aiohttp.DummyCookieJar(), trust_env=False) as anonymous:
        status, _ = await _request(anonymous, jupyter_url)
        require(status in (401, 403, 302), "Jupyter's contents API did not require authentication")
        status, _ = await _request(anonymous, "http://127.0.0.1:8080/api/resources/port-check.txt")
        require(status == 401, "FileBrowser's resource API did not require authentication")
    status, body = await _request(session, "http://127.0.0.1:8080/api/resources/port-check.txt",
                                  headers={"X-Auth": browser_token})
    require(status == 200 and json.loads(body).get("name") == "port-check.txt",
            "FileBrowser could not access the check's own workspace file")
    return {"8188": "ComfyUI token login/cookie/HTTP/template/POST/WebSocket passed",
            "8888": "JupyterLab token authentication and workspace access passed",
            "8080": "FileBrowser login and authenticated workspace access passed",
            "22": "SSH protocol banner passed", "foreign_origins_rejected": True,
            "gateway_unauthenticated_status": 401,
            "gateway_login_redirect": "303 to clean root URL",
            "gateway_cookie_flags": "Secure HttpOnly SameSite=Strict Path=/",
            "inference": "not_run", "model_downloads": "bypassed_in_child_test_shim",
            "comfyui_device": "cpu"}


def _check_comfy_environment(process_file):
    processes = json.loads(process_file.read_text())
    require("comfy" in processes and "filebrowser" in processes, "Expected service child processes were not observed")
    values = Path("/proc", str(processes["comfy"]), "environ").read_bytes().split(b"\0")
    keys = {value.partition(b"=")[0].decode("ascii", errors="replace") for value in values}
    require(not set(RAW_SECRET_KEYS).intersection(keys), "A login secret reached the ComfyUI child environment")


def _safe_tail(log, token):
    text = log.read_text(errors="replace")[-6000:]
    text = text.replace(token, "[REDACTED]")
    text = re.sub(r"(?i)(token=)[^\s&\"'<>]+", r"\1[REDACTED]", text)
    text = re.sub(r"\$2[aby]\$\d\d\$[./A-Za-z0-9]{53}", "[REDACTED_HASH]", text)
    return text


def run_check(package_root, comfy_root, timeout):
    package_root, comfy_root = Path(package_root).resolve(), Path(comfy_root).resolve()
    require(math.isfinite(timeout) and timeout > 0, "Timeout must be positive and finite")
    require((package_root / "scripts/entrypoint.sh").is_file() and (comfy_root / "main.py").is_file()
            and Path("/start.sh").is_file(), "Run this check inside the built disposable image")
    require(Path("/workspace").is_dir(), "The image must provide /workspace")
    _port_preflight()
    previous_pids = _pids()
    process = None
    token = secrets.token_urlsafe(24)
    with tempfile.TemporaryDirectory(prefix="fibo-port-check-") as temporary, \
            tempfile.TemporaryDirectory(prefix="fibo-port-check-", dir="/workspace") as workspace_dir:
        temporary, workspace = Path(temporary), Path(workspace_dir)
        (workspace / "port-check.txt").write_text("fibo-service-port-check\n")
        shim, entrypoint, log = temporary / "python-shim", temporary / "entrypoint.sh", temporary / "services.log"
        shim.write_text(_child_shim(sys.executable))
        shim.chmod(0o700)
        entrypoint.write_text((package_root / "scripts/entrypoint.sh").read_text()
                              .replace("/opt/venv/bin/python", shlex.quote(str(shim))))
        key = temporary / "ssh-key"
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "fibo-port-check"],
                       check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        environment = os.environ.copy()
        for key_name in (*RAW_SECRET_KEYS, "JUPYTER_DISABLE_AUTH", "RUNPOD_POD_ID", "FIBO_PUBLIC_ORIGIN"):
            environment.pop(key_name, None)
        environment.update(FIBO_MODE="pod", FIBO_ROOT=str(package_root), COMFY_ROOT=str(comfy_root),
            FIBO_DATA_DIR=str(workspace / "ComfyUI"), FIBO_PUBLIC_ORIGIN="http://127.0.0.1:8188",
            JUPYTER_PASSWORD=token, FILEBROWSER_PASSWORD=token, PUBLIC_KEY=key.with_suffix(".pub").read_text().strip(),
            FIBO_PORT_CHECK_PROCESSES=str(temporary / "processes.json"), CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
            PYTHONPYCACHEPREFIX=str(temporary / "pycache"), HF_HOME=str(temporary / "huggingface"),
            TORCH_HOME=str(temporary / "torch"), XDG_CACHE_HOME=str(temporary / "cache"))
        try:
            with log.open("wb") as output:
                process = subprocess.Popen(["bash", str(entrypoint)], cwd=package_root, env=environment,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                result = asyncio.run(_services_ready(process, workspace, token, timeout))
                _check_comfy_environment(temporary / "processes.json")
                result.update(status="passed", actual_entrypoint=True, raw_comfy_login_secrets_absent=True)
                return result
        except BaseException:
            if log.exists():
                tail = _safe_tail(log, token)
                if tail:
                    print("Service-port check diagnostic tail (generated credentials redacted):\n" + tail, file=sys.stderr)
            raise
        finally:
            _stop(process, previous_pids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(os.environ.get("FIBO_ROOT", "/opt/fibo")))
    parser.add_argument("--comfy-root", type=Path, default=Path(os.environ.get("COMFY_ROOT", "/opt/comfyui")))
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(run_check(args.package_root, args.comfy_root, args.timeout), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (PortCheckError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print("Service-port check failed: " + str(error), file=sys.stderr)
        sys.exit(1)
