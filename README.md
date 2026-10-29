# Fibo 2 Edit Turbo for Runpod

Edit an uploaded image with Bria's Fibo 2 Edit Turbo in ComfyUI, or call the same workflow through a Runpod queue worker. The default uses four Euler/simple steps, CFG 1, the official INT8 diffusion weights, a BF16 Qwen3-VL-4B text encoder and the FLUX.2 VAE.

This package adapts the preparation gateway and persistent model layout from [MOSS Runpod](https://github.com/akhilsivanand/moss-runpod). It pins ComfyUI v0.43.0 and uses its native Fibo nodes; no image API key or third-party inference node is required.

## License and intended use

**Fibo model weights are licensed CC BY-NC 4.0. Commercial use requires a separate agreement with [Bria](https://platform.bria.ai/licensing/pricing?model=fibo-2).** This repository's original runtime code is MIT-licensed. Downloading through this package does not grant commercial rights. See [third-party notices](THIRD_PARTY_NOTICES.txt) and [model provenance](docs/provenance.md).

## Interactive ComfyUI

Deploy the [public Runpod template](https://console.runpod.io/deploy?template=rxfq67i2yt&type=gpu), or create a GPU Pod from the tested image recorded in `deploy/pod-template.json`:

1. Select a CUDA 13-compatible GPU with at least 24 GB VRAM, a 60 GB container disk, and persistent storage mounted at `/workspace` if you want files retained.
2. Enable SSH and Jupyter. Start `/opt/fibo/scripts/entrypoint.sh` with `FIBO_MODE=pod` and expose `8188/http,8080/http,8888/http,22/tcp`.
3. Open **Connect → ComfyUI (8188)** and enter the Jupyter token from Runpod Connect once. The startup page reports preparation while three model files download and their SHA256 hashes are verified. The total download is **17.62 GB**; startup time depends on the host and network.
4. Open **Fibo 2 Edit Turbo**, upload your source image, set **Scale** (1 = input size, 0.5 = half width/height, 2 = double), enter an edit instruction and click **Run**. Width and height are calculated from the uploaded image and rounded down to multiples of 16. To override Scale, enter dimensions in **Resolution override**: `0 × 0` keeps Scale, two nonzero values choose that resolution, and a single nonzero value preserves the input aspect ratio. Setting `1024 × 1024` restores the original square output size. A different aspect ratio center-crops the input during Fibo conditioning.

The scale control is a workflow update applied to the existing live Pod without restarting ComfyUI. It is present in this source checkout and the persistent saved workflow; the published 0.1.0 image still contains the original fixed-resolution workflow. The queue API continues to accept explicit width/height.

The standard service ports are **8188 ComfyUI**, **8888 JupyterLab**, **8080 FileBrowser**, and **22 SSH**, matching [Runpod's official ComfyUI layout](https://github.com/runpod-workers/comfyui-base/blob/main/README.md). Jupyter uses Runpod's generated token and opens at `/workspace`. ComfyUI retains per-Pod token protection on 8188. Enter the Jupyter token from Runpod Connect on its login page; a protected browser cookie keeps subsequent requests signed in. Set `FIBO_PASSWORD` to select Basic authentication instead. The inference backend stays private on loopback port 8189.

FileBrowser opens at `/workspace` with username `admin`. Its initial password is `FILEBROWSER_PASSWORD` when supplied, otherwise the Jupyter token from its first boot. Its database retains the password on persistent storage, including across later token changes. The FileBrowser release and checksum are pinned by the installer.

The native workflow shortcut is `/?template=fibo2-edit-turbo&source=fibo`. You can use an ordinary edit instruction or supply a structured VGL caption. The supplied workflow is a single-image edit; it does not install the optional prompt-enhancement model from the upstream example.

Models and user files live under `/workspace/ComfyUI/{models,input,output,user}`. `/opt/comfyui/models` refers to the same model tree. Package updates preserve changed user workflows. GPU charges and any retained storage are billed by Runpod; check your actual returned rate and stop the Pod when finished.

## Optional image creation

The additional **Fibo 2 Dev** workflow creates an image from a text prompt without an input image. Open **Workflows → FIBO → fibo2-create-dev** after installing it. Its default is 1024 × 1024, 50 Euler/simple steps and CFG 5. It uses the same Qwen3-VL encoder and FLUX.2 VAE as editing, plus one **8.41 GB** Dev INT8 checkpoint. The optional upstream prompt enhancer is not installed.

The creation files are an addition to this source checkout and the live Pod; the immutable 0.1.0 image does not include them. To install from this checkout into an existing prepared Fibo workspace, run:

```bash
python scripts/install_creation.py --data-root /workspace/ComfyUI
```

The installer verifies the pinned Dev checksum, preserves an existing changed workflow, and does not restart ComfyUI. Models and the saved workflow remain on persistent storage. This is an interactive ComfyUI workflow; the queue API still exposes the `edit` preset. Dev weights have the same CC BY-NC 4.0 restriction.

On 29 October 2026, a real browser run created a 1024 × 1024 image on the existing RTX 4090 in 43.883 seconds of server execution. The edit Scale test converted a 512 × 512 input to 768 × 768 at 1.5×. Native frontend export, authenticated POST/WebSocket and fresh sampler execution passed. Neither workflow installation restarted the container, ComfyUI, or gateway. An additional manual-override test used Scale 1.5 with an explicit 768 × 512 target and produced 768 × 512. See [creation evidence](docs/workflow-creation-validation.json), [scale evidence](docs/workflow-scale-validation.json), and [override evidence](docs/workflow-override-validation.json).

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

The 0.2.0 Dockerfile derives from the immutable, GPU-tested Fibo 0.1.0 image. ComfyUI, PyTorch/CUDA and the Python dependency inventory remain byte-for-byte inherited; the derivative updates workflows and the standard service layout and installs a checksum-pinned FileBrowser binary. The full Python inventory is `requirements/runtime.lock`; the original eight-package delta remains documented in `requirements/upgrade.lock`. Models download at runtime.

```bash
python -m pip install -r requirements/test.txt
python -m pytest -q
python scripts/check_release.py
docker build -t fibo-comfyui:local .
```

The maintainer dispatches [this repository's build workflow](.github/workflows/container.yml) on the reviewed Fibo source revision. Images are published to the dedicated public package `ghcr.io/akhilsivanand/fibo-comfyui:staging-<commit>`; release promotion reuses the tested manifest.

The image `ghcr.io/akhilsivanand/fibo-comfyui@sha256:7487d8ab9784c0f8b4176b21332e0a3f6c762053b597379d3a94e44dfcae46d2` passed real GPU validation on **29 October 2026** using an RTX 4090 (24 GB class), CUDA 13.0, and PyTorch 2.13.0+cu130. Both the ComfyUI browser and a fresh Runpod queue worker completed 1024 × 1024 edits with four steps and CFG 1. The browser edit took 55.4 seconds; the queue job reported 14.1 seconds execution after 255.3 seconds queue delay. These separate synthetic-image tests establish execution, not a speed or general quality comparison.

For the historical 0.1.0 image, native Connect login, protected browser POST/WebSocket requests, template import/export, and restart retention of models, workflow, marker and output were verified. A fresh browser passed login and template import again after restart. All 84 source tests and container CPU/schema checks passed; anonymous registry access was verified. The temporary validation endpoint was deleted. See the dated [historical 0.1.0 validation evidence](docs/validation-0.1.0.json), tied to the tested runtime revision and image digest. `python scripts/check_release.py --publish` requires a new exact-image GPU report for the current version; the historical 0.1.0 report is retained separately. The observed Pod GPU rate was $0.89/hour, with storage billed separately; actual rates depend on the selected host.

See [release instructions](docs/release.md). Hub submission, automated test success, and Runpod's manual approval are separate states; no approval is implied by this source package.

## Publication status

The [public Pod template](https://console.runpod.io/deploy?template=rxfq67i2yt&type=gpu) is being aligned with the current 0.2.0 service layout. The historical GPU-tested image and [v0.1.0 release](https://github.com/akhilsivanand/fibo-runpod/releases/tag/v0.1.0) are published. The [Hub submission](https://console.runpod.io/hub/listing/akhilsivanand/fibo-runpod) was accepted on 29 October 2026, but its build failed downloading the GitHub source archive with HTTP 401, before Dockerfile execution. Hub produced no image, did not start automated tests, and is not approved or listed. An anonymous download of the exact source archive returned HTTP 200. Runpod needs to investigate this source authorization failure; the public Pod deployment remains usable. See [publication evidence](docs/publication-0.1.0.json).
