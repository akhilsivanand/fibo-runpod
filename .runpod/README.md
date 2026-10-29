# Fibo 2 Edit Turbo API

Upload an image as raw base64 and describe the change. This queue worker executes the native ComfyUI Fibo 2 Edit Turbo workflow with four steps and CFG 1. It uses official INT8 diffusion weights, the BF16 Qwen3-VL-4B text encoder and the FLUX.2 VAE.

**Noncommercial model use only unless you have a separate Bria commercial license.** Model weights are CC BY-NC 4.0; runtime code is MIT. See the repository README and third-party notices for attribution and licensing links.

The first worker prepares and verifies 17.62 GB of public model files. A retained model cache avoids downloading them on later starts. Select a CUDA 13-compatible GPU with at least 24 GB VRAM and 60 GB of disk.

```json
{"input":{"preset":"edit","prompt":"Change the red mug to blue, preserving the scene.","image_base64":"<raw base64 PNG, JPEG or WebP>","seed":42,"width":1024,"height":1024,"steps":4,"cfg":1}}
```

The input image is required. Outputs contain generated PNG data and execution metadata. Source code, pinned assets and actual validation status: https://github.com/akhilsivanand/fibo-runpod
