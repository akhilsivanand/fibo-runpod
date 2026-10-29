# Fibo 2 Edit Turbo for Runpod

Edit an uploaded image with Bria's Fibo 2 Edit Turbo in ComfyUI, or call the same workflow through a Runpod queue worker. The default uses four Euler/simple steps, CFG 1, the official INT8 diffusion weights, a BF16 Qwen3-VL-4B text encoder and the FLUX.2 VAE.

This package adapts the authenticated gateway and persistent model layout from [MOSS Runpod](https://github.com/akhilsivanand/moss-runpod). It pins ComfyUI v0.43.0 and uses its native Fibo nodes; no image API key or third-party inference node is required.

## License and intended use

**Fibo model weights are licensed CC BY-NC 4.0. Commercial use requires a separate agreement with [Bria](https://platform.bria.ai/licensing/pricing?model=fibo-2).** This repository's original runtime code is MIT-licensed. Downloading through this package does not grant commercial rights. See [third-party notices](THIRD_PARTY_NOTICES.txt) and [model provenance](docs/provenance.md).

## Interactive ComfyUI

Use the published Pod template when available, or create a Runpod GPU Pod from the tested image recorded in `deploy/pod-template.json`:

1. Select a CUDA 13-compatible GPU with at least 24 GB VRAM, a 60 GB container disk, and persistent storage mounted at `/workspace` if you want files retained.
2. Enable SSH and Runpod's native Connect token. Start `/opt/fibo/scripts/entrypoint.sh` with `FIBO_MODE=pod` and expose `8888/http,22/tcp`.
3. Open **Connect → Fibo 2 ComfyUI**. The protected startup page reports preparation while three model files download and their SHA256 hashes are verified. The total download is **17.62 GB**; startup time depends on the host and network.
4. Open **Fibo 2 Edit Turbo**, upload your source image, enter an edit instruction and click **Run**.

The native workflow shortcut is `/?template=fibo2-edit-turbo&source=fibo`. You can use an ordinary edit instruction or supply a structured VGL caption. The supplied workflow is a single-image edit; it does not install the optional prompt-enhancement model from the upstream example.

Models and user files live under `/workspace/ComfyUI/{models,input,output,user}`. `/opt/comfyui/models` refers to the same model tree. Package updates preserve changed user workflows. GPU charges and any retained storage are billed by Runpod; check your actual returned rate and stop the Pod when finished.

## Queue API

Set `FIBO_MODE=serverless`. Each worker has isolated temporary inputs, outputs, user settings and runtime locks; persistent model downloads can be shared. The async handler accepts one preset, `edit`:

```json
{
  "input": {
    "preset": "edit",
    "image_base64": "<raw base64 PNG, JPEG or WebP>",
    "prompt": "Change the red mug to blue, preserving the scene and lighting.",
    "seed": 42,
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "cfg": 1
  }
}
```

Supply your own source image as raw base64, without a data URL prefix. Input files are limited to 7 MiB and 16 megapixels; output dimensions must be multiples of 64 with at most 2,097,152 pixels. The API returns generated PNG data and job metadata. It never accepts arbitrary ComfyUI graphs, paths or remote image URLs. The bundled Hub test creates a real edit from a small original test image.

## Build and validation

The Dockerfile derives from an immutable MOSS runtime digest, replaces its application and ComfyUI checkout, and installs a hash-pinned eight-package dependency delta. The full resulting Python inventory is `requirements/runtime.lock`; the installed delta is `requirements/upgrade.lock`. Models are downloaded at runtime, not baked into the image.

```bash
python -m pip install -r requirements/test.txt
python -m pytest -q
python scripts/check_release.py
docker build -t fibo-comfyui:local .
```

The maintainer dispatches [this repository's build workflow](.github/workflows/container.yml) on the reviewed Fibo source revision. Images are published to the dedicated public package `ghcr.io/akhilsivanand/fibo-comfyui:staging-<commit>`; release promotion reuses the tested manifest.

The staging image `ghcr.io/akhilsivanand/fibo-comfyui@sha256:7487d8ab9784c0f8b4176b21332e0a3f6c762053b597379d3a94e44dfcae46d2` passed CPU package imports and native workflow/schema checks and was verified anonymously pullable. These checks do not establish successful GPU inference. A public release must also pass real image editing, Pod authentication and restart-preservation checks, and a fresh queue-worker edit against that exact image digest. `python scripts/check_release.py --publish` requires those dated results in `docs/validation.json`. GPU validation is pending.

See [release instructions](docs/release.md). Hub submission, automated test success, and Runpod's manual approval are separate states; no approval is implied by this source package.
