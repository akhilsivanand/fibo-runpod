"""Start isolated JupyterLab on its standard port without command-line secrets."""
import asyncio
import os
import re
from pathlib import Path
import stat
import tempfile

from .gateway import validate_secret

JUPYTER_BINARY = "/opt/jupyter/bin/jupyter-lab"
SECRET_VARIABLES = {
    "FIBO_PASSWORD", "FILEBROWSER_PASSWORD", "JUPYTER_PASSWORD", "JUPYTER_DISABLE_AUTH",
}


def _checked_stat(path):
    """Reject symlinks in the complete path before changing managed state."""
    for component in [*reversed(path.parents), path]:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("Preserving an existing symlink in the Jupyter state path.")
        if component != path and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Preserving an existing non-directory Jupyter parent path.")
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def prepare_jupyter(data_root, *, root=None, binary=JUPYTER_BINARY, environ=None):
    """Prepare a private config and return argv plus the isolated child environment."""
    source = dict(os.environ if environ is None else environ)
    token = validate_secret(source.get("JUPYTER_PASSWORD", ""), "JUPYTER_PASSWORD")
    pod_id = source.get("RUNPOD_POD_ID")
    public_origin = None
    if pod_id is not None:
        if not re.fullmatch(r"[a-z0-9]{8,58}", pod_id):
            raise ValueError("RUNPOD_POD_ID is not a valid Pod identifier.")
        public_origin = "https://" + pod_id + "-8888.proxy.runpod.net"
    data = Path(os.path.abspath(Path(data_root).expanduser()))
    served_root = Path(os.path.abspath(Path(root).expanduser())) if root is not None else data.parent
    for path in (data, served_root):
        metadata = _checked_stat(path)
        if metadata is None or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Jupyter's data and workspace directories must already exist.")
    state = data / "state" / "jupyter"
    runtime = state / "runtime"
    config = state / "jupyter_server_config.py"
    for directory in (state, runtime):
        metadata = _checked_stat(directory)
        if metadata is not None and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Preserving an existing invalid Jupyter state directory.")
    metadata = _checked_stat(config)
    if metadata is not None and not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Preserving an existing invalid Jupyter configuration file.")
    state.mkdir(parents=True, mode=0o700, exist_ok=True)
    state.chmod(0o700)
    runtime.mkdir(mode=0o700, exist_ok=True)
    runtime.chmod(0o700)
    # The only token reference is an environment lookup. Jupyter's startup
    # URLs are logged at INFO, so WARNING avoids logging token-bearing URLs.
    content = ("import os\n"
               "c = get_config()\n"
               "c.ServerApp.ip = '0.0.0.0'\n"
               "c.ServerApp.port = 8888\n"
               "c.ServerApp.port_retries = 0\n"
               "c.ServerApp.allow_root = True\n"
               "c.ServerApp.allow_remote_access = True\n"
               "c.ServerApp.trust_xheaders = True\n"
               f"c.ServerApp.root_dir = {str(served_root)!r}\n"
               "c.ServerApp.open_browser = False\n"
               "c.ServerApp.log_level = 30\n"
               "c.IdentityProvider.token = os.environ['JUPYTER_TOKEN']\n")
    if public_origin is not None:
        content += f"c.ServerApp.allow_origin = {public_origin!r}\n"
    with tempfile.NamedTemporaryFile(dir=state, prefix=".jupyter-config-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.chmod(0o600)
        temporary.replace(config)
    finally:
        temporary.unlink(missing_ok=True)
    environment = {key: value for key, value in source.items() if key not in SECRET_VARIABLES}
    environment["JUPYTER_TOKEN"] = token
    environment["JUPYTER_RUNTIME_DIR"] = str(runtime)
    return [str(binary), "--config", str(config)], environment


async def start_jupyter(data_root, *, root=None, binary=JUPYTER_BINARY, environ=None):
    """Return the child process so the application supervisor can manage its life."""
    command, environment = prepare_jupyter(data_root, root=root, binary=binary, environ=environ)
    return await asyncio.create_subprocess_exec(*command, env=environment, umask=0o077)
