"""Prepare persistent assets, start ComfyUI, and supervise the selected service."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

import aiohttp
from aiohttp import web

from . import __version__
from .assets import prepare_models
from .gateway import auth_from_environment, create_app
from .paths import DEFAULT_DATA_ROOT, expose_persistent_models
from .workflows import install_workflows, atomic_write
from .filebrowser import start_filebrowser


def check_package(root, comfy_root):
    manifest = json.loads((root / 'config/models.json').read_text())
    assert len(manifest['models']) == 3, 'Expected three Fibo Edit 2 model assets'
    config = json.loads((root / 'config/workflows.json').read_text())
    for item in list(config['presets'].values()) + list(config.get('ui_only', {}).values()):
        json.loads((root / item['ui']).read_text())
        if item.get('api'):
            json.loads((root / item['api']).read_text())
    assert (comfy_root / 'main.py').is_file(), 'Baked ComfyUI is missing'
    import torch
    import comfy_kitchen
    import runpod
    assert torch.__version__ == '2.13.0+cu130', torch.__version__
    print(json.dumps({'package': __version__, 'torch': torch.__version__,
                      'cuda_runtime': torch.version.cuda, 'gpu_inference': 'not_checked'}))


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA 13-compatible NVIDIA GPU is required.')
    if torch.cuda.get_device_properties(0).total_memory < 23 * 1024**3:
        raise RuntimeError('This release requires a GPU with at least 24 GB VRAM.')


async def wait_for_comfy(process, timeout=180):
    deadline = asyncio.get_running_loop().time() + timeout
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f'ComfyUI exited during startup (code {process.returncode}).')
            try:
                async with session.get('http://127.0.0.1:8189/object_info/UNETLoader') as response:
                    info = await response.json()
                    names = info['UNETLoader']['input']['required']['unet_name'][0]
                    if response.status == 200 and 'fibo2_edit_turbo_int8_convrot.safetensors' in names:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError):
                pass
            await asyncio.sleep(1)
    raise RuntimeError('ComfyUI did not become ready within 180 seconds.')


async def serve(root, comfy_root, data_root, mode):
    if mode == 'serverless':
        # Workers may share the persistent model cache. Each gets its own
        # database, inputs, outputs, and runtime lock on ephemeral storage.
        with tempfile.TemporaryDirectory(prefix='fibo-worker-') as workspace:
            await serve_workspace(root, comfy_root, Path(workspace), mode, data_root)
    else:
        await serve_workspace(root, comfy_root, data_root, mode, data_root)


async def serve_workspace(root, comfy_root, data_root, mode, model_data_root):
    auth = auth_from_environment(mode)
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / 'state').mkdir(exist_ok=True)
    with (data_root / 'state/runtime.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A FIBO runtime is already using this data directory.')
        state = {'phase': 'checking_gpu', 'detail': 'Checking the GPU and preparing your workspace.', 'ready': False}
        app = create_app(state, root, mode=mode, auth=auth)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, '0.0.0.0' if mode == 'pod' else '127.0.0.1', auth.port).start()
        children = []
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        try:
            if mode == 'pod':
                children.append(await start_filebrowser(data_root))
            await asyncio.to_thread(check_gpu)
            for name in ('models', 'input', 'output', 'temp', 'user', 'cache', 'custom_nodes'):
                (data_root / name).mkdir(exist_ok=True)
            if mode == 'pod':
                await asyncio.to_thread(expose_persistent_models, comfy_root, data_root)

            def progress(update):
                loop.call_soon_threadsafe(state.update, update)
            await asyncio.to_thread(prepare_models, root / 'config/models.json', model_data_root, progress)
            if mode == 'serverless':
                (data_root / 'models').rmdir()  # This worker's new, empty directory.
                (data_root / 'models').symlink_to(model_data_root / 'models', target_is_directory=True)
            # During downloads, the default TERM behavior exits immediately and
            # leaves a resumable partial. Supervise graceful service shutdown
            # only once preparation has finished.
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
            install_workflows(root, data_root)
            extra = data_root / 'state/comfy-paths.yaml'
            # Include the baked template-only module while writable custom nodes
            # and model directories follow --base-directory on the volume.
            atomic_write(extra, ('fibo_bundled:\n  custom_nodes: ' + json.dumps(str(comfy_root / 'custom_nodes')) + '\n').encode())
            env = os.environ.copy()
            for secret in ('FIBO_PASSWORD', 'JUPYTER_PASSWORD', 'FILEBROWSER_PASSWORD', 'JUPYTER_DISABLE_AUTH'):
                env.pop(secret, None)
            env.update(FIBO_ROOT=str(root), FIBO_DATA_DIR=str(data_root), COMFY_URL='http://127.0.0.1:8189')
            command = [sys.executable, '-u', str(comfy_root / 'main.py'), '--listen', '127.0.0.1', '--port', '8189',
                       '--base-directory', str(data_root), '--user-directory', str(data_root / 'user'),
                       '--input-directory', str(data_root / 'input'), '--output-directory', str(data_root / 'output'),
                       '--temp-directory', str(data_root / 'temp'), '--extra-model-paths-config', str(extra), '--disable-auto-launch']
            if mode == 'serverless':
                command.append('--disable-metadata')
            state.update(phase='starting_comfyui', detail='Starting ComfyUI and loading the workflow catalog.')
            comfy = await asyncio.create_subprocess_exec(*command, cwd=comfy_root, env=env)
            children.append(comfy)
            await wait_for_comfy(comfy)
            if mode == 'serverless':
                children.append(await asyncio.create_subprocess_exec(sys.executable, '-u', str(root / 'handler.py'), env=env))
            state.update(phase='ready', detail='Your workflows are ready.', ready=True)
            print(f'FIBO {__version__}: {mode} service ready.', flush=True)
            tasks = [asyncio.create_task(child.wait()) for child in children]
            tasks.append(asyncio.create_task(stop.wait()))
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if not stop.is_set():
                code = next(task.result() for task in done)
                raise RuntimeError(f'A required service exited (code {code}).')
        finally:
            state['ready'] = False
            for child in children:
                if child.returncode is None:
                    child.terminate()
            for child in children:
                try:
                    await asyncio.wait_for(child.wait(), 15)
                except asyncio.TimeoutError:
                    child.kill()
                    await child.wait()
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-package', action='store_true', help='Check baked imports and assets without downloading or using a GPU')
    args = parser.parse_args()
    root = Path(os.environ.get('FIBO_ROOT', '/opt/fibo')).resolve()
    comfy_root = Path(os.environ.get('COMFY_ROOT', '/opt/comfyui')).resolve()
    if args.check_package:
        check_package(root, comfy_root)
        return
    mode = os.environ.get('FIBO_MODE', 'pod')
    if mode not in ('pod', 'serverless'):
        raise RuntimeError('FIBO_MODE must be pod or serverless.')
    asyncio.run(serve(root, comfy_root, Path(os.environ.get('FIBO_DATA_DIR', str(DEFAULT_DATA_ROOT))).resolve(), mode))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'FIBO startup failed: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
