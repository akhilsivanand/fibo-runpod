"""Queue-based Runpod Hub entry point for the Fibo Edit 2 editing preset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fibo_runtime.jobs import ComfyJobRunner

_runner = None


async def handler(job):
    """Run in the SDK's existing event loop; do not nest asyncio.run()."""
    global _runner
    if not isinstance(job, dict) or not isinstance(job.get("input"), dict):
        raise ValueError("The job must contain an input object")
    if _runner is None:
        _runner = ComfyJobRunner()
    return await _runner.run(job["input"])


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
