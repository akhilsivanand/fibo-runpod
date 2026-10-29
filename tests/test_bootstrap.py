import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fibo_runtime.bootstrap import serve
from fibo_runtime import bootstrap


def test_serverless_workers_isolate_runtime_but_share_model_cache(tmp_path):
    paths = []

    async def record(root, comfy, workspace, mode, models):
        assert mode == 'serverless'
        assert models == tmp_path
        assert workspace != models
        assert workspace.is_dir()
        paths.append(workspace)

    async def run():
        with patch('fibo_runtime.bootstrap.serve_workspace', side_effect=record):
            await asyncio.gather(serve(Path('/package'), Path('/comfy'), tmp_path, 'serverless'),
                                 serve(Path('/package'), Path('/comfy'), tmp_path, 'serverless'))
    asyncio.run(run())
    assert paths[0] != paths[1]
    assert all(not path.exists() for path in paths)


def test_pod_uses_persistent_workspace(tmp_path):
    with patch('fibo_runtime.bootstrap.serve_workspace', new_callable=AsyncMock) as start:
        asyncio.run(serve(Path('/package'), Path('/comfy'), tmp_path, 'pod'))
        start.assert_awaited_once_with(Path('/package'), Path('/comfy'), tmp_path, 'pod', tmp_path)


def test_fresh_pod_does_not_move_another_workspaces_files(tmp_path):
    old, current = tmp_path / 'fibo', tmp_path / 'ComfyUI'
    (old / 'models').mkdir(parents=True)
    source = old / 'models/keep.bin'
    source.write_bytes(b'user data')
    with patch.object(bootstrap, 'DEFAULT_DATA_ROOT', current):
        with patch.object(bootstrap, 'serve_workspace', new_callable=AsyncMock) as start:
            asyncio.run(serve(Path('/package'), Path('/comfy'), current, 'pod'))
            assert source.read_bytes() == b'user data'
            assert not current.exists()
            start.assert_awaited_once_with(Path('/package'), Path('/comfy'), current, 'pod', current)
