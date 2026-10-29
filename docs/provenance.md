# Fibo 2 Edit Turbo sources

Reviewed on 2026-10-29. These records establish source provenance and the intended configuration; they do not establish GPU inference, image quality, container readiness, or Runpod publication. Deployment validation is recorded separately.

The package uses **Fibo 2.0 Edit Turbo**, not the original Fibo Edit or Fibo Edit 1.5. Bria describes it as the distilled editing variant of Fibo 2.0, with four inference steps and guidance 1.0. It uses Qwen3-VL-4B-Instruct for conditioning and the FLUX.2 VAE. The complete model workflow is non-commercial under CC BY-NC 4.0 unless covered by a separate Bria agreement. See the [official model card](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) and [Comfy Org repackaging card](https://huggingface.co/Comfy-Org/Fibo-2.0).

## Pinned implementation

- ComfyUI v0.43.0: `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24` ([release](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.43.0)). This release includes native Fibo 2 support and its RoPE/negative-prompt corrections.
- Native conditioning: [`comfy_extras/nodes_fibo2.py`](https://github.com/Comfy-Org/ComfyUI/blob/426f5eea52a7ece1dd599fff4cb6cd1f352ddd24/comfy_extras/nodes_fibo2.py).
- Dynamic input representation: [`comfy_api/latest/_io.py`](https://github.com/Comfy-Org/ComfyUI/blob/426f5eea52a7ece1dd599fff4cb6cd1f352ddd24/comfy_api/latest/_io.py), specifically `Autogrow`, `parse_class_inputs`, `finalize_prefix`, and `build_nested_inputs`.
- Workflow source: `Comfy-Org/workflow_templates` commit `3c2199b9362457b12b9c60c6cc303501526ebd62`, [`templates/image_fibo2_edit_turbo_pe.json`](https://github.com/Comfy-Org/workflow_templates/blob/3c2199b9362457b12b9c60c6cc303501526ebd62/templates/image_fibo2_edit_turbo_pe.json).

The upstream workflow is preserved byte-for-byte in `workflows/upstream/image_fibo2_edit_turbo_pe.json`: Git blob `dbbe930e587994ba3a23e62c269968768dee37ef`, SHA256 `d39c6e6b9228e6186683f60864b39f237d5c09ff7f31f5049b04006309791de8`. Its MIT license is beside it in `workflows/upstream/LICENSE.txt`. It is a provenance copy, not an installed preset; it contains optional enhancer references and its original unpinned model links.

## Model assets

All three files are fetched from `Comfy-Org/Fibo-2.0`, revision `4742d4bec410034142c8e65559e5f00c6f29eccb`. The repository was public and ungated when checked. `config/models.json` contains the immutable download URLs, byte counts and SHA256 hashes verified against Hugging Face's public file metadata. No model weights are bundled in the source package or baked into the image.

| File | Bytes | Component license |
|---|---:|---|
| `diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors` | 8,408,562,474 | CC BY-NC 4.0 |
| `text_encoders/qwen3vl_4b_bf16.safetensors` | 8,875,719,384 | Apache 2.0 |
| `vae/flux2-vae.safetensors` | 336,211,292 | Apache 2.0 |
| **Total** | **17,620,493,150** | |

The INT8 diffusion repack uses Hadamard rotation and per-channel scaling, retaining modulation layers in BF16; quantization can change output from the BF16 source. The encoder remains BF16. This is the precision combination used by the pinned Comfy Org Turbo workflow.

The repackaging repository declares CC BY-NC 4.0 for its model release; that declaration is retained in each manifest entry's `source.repository_license`. The manifest also records the original component licenses: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) declares Apache 2.0, and [Black Forest Labs explicitly licenses the FLUX.2 autoencoder](https://github.com/black-forest-labs/flux2#flux2-autoencoder) under Apache 2.0. These component terms do not remove the Fibo diffusion model's non-commercial restriction. Attribution and license text are in `THIRD_PARTY_NOTICES.txt`.

## Derived preset

`config/workflows.json` selects a single preset, `edit`, with a simple UI graph and an API graph named `fibo2-edit-turbo.json`. The derivative retains the source model combination and Turbo sampler settings, removes optional prompt enhancement, comparison and batch helper nodes, uses one uploaded reference, and calculates output dimensions from its size multiplied by a Scale control (default 1). The immutable 0.1.0 image retains its original 1024 × 1024 default. Model download metadata in the derived UI graph is revision-pinned. Layout, labels and usage notes are revised. No custom inference node or external API is required.

The upstream graph's optional local Qwen3-VL-8B prompt writer is not installed. The native conditioning node accepts plain edit instructions or a VGL JSON caption with an `edit_instruction` key. A detailed structured caption can improve control; the minimal preset does not generate that caption for the user. [Diffusers' Fibo 2 documentation](https://huggingface.co/docs/diffusers/en/api/pipelines/bria_fibo2) documents both prompt forms.

The API graph uses these native nodes:

| ID | Node | Role |
|---|---|---|
| `1` | `UNETLoader` | Fibo 2 Edit Turbo INT8 diffusion |
| `2` | `CLIPLoader`, type `fibo2` | BF16 Qwen3-VL 4B encoder |
| `3` | `VAELoader` | Full FLUX.2 VAE |
| `10` | `LoadImage` | User-provided source image |
| `4` | `TextEncodeFibo2` | Edit prompt, reference conditioning, output dimensions and empty sampling latent |
| `5` | `KSampler` | Euler/simple, four steps, CFG 1, denoise 1, seed 1234 |
| `6` | `VAEDecode` | Decode sampled latent |
| `7` | `SaveImage` | Save edited output |
| `12` | `PrimitiveFloat` | One user-facing scale factor, default 1 |
| `13` | `ImageScaleBy` | Calculate the scaled reference size |
| `14` | `GetImageSize` | Feed calculated width and height to conditioning |

`TextEncodeFibo2` receives the image through the flat API input **`images.image_1`**, not an `images` dictionary. The V3 Autogrow schema expands that dotted input name, and `build_nested_inputs` reconstructs the dictionary for `execute`. Its three outputs are positive conditioning at index 0, negative conditioning at index 1 and latent at index 2. Width, height and batch size are inputs on node `4`; there is no separate empty-latent node.

A source image is mandatory for the edit preset. The bundled graphs leave the LoadImage selection empty for the user or API wrapper to supply. With one reference, ComfyUI resizes and center-crops it to the output dimensions. The current UI connects width and height to the uploaded image size multiplied by Scale. Fibo floors those dimensions to multiples of 16, preserving the aspect ratio apart from rounding. The queue wrapper still replaces these inputs with its explicit width/height settings (1024 × 1024 by default). The preset covers single-image editing. Masked and multiple-reference editing are capabilities of the underlying model, not additional validated presets in this package.

The API graph was derived directly from the nonnested native graph and pinned node schemas. Static link/schema checks do not replace a frontend export check or real GPU inference.

## Optional Dev creation workflow

The creation workflow derives from Comfy Org's [Dev example](https://github.com/Comfy-Org/workflow_templates/blob/3c2199b9362457b12b9c60c6cc303501526ebd62/templates/image_fibo2_edit_dev_pe.json), preserved byte-for-byte in `workflows/upstream/image_fibo2_edit_dev_pe.json` (SHA256 `24be29c7dfdf05aef158fb14754c51cf01f346861a02e1e4914b0c7e1e782aae`). The reference-image and optional prompt-enhancer paths are removed; the native TextEncodeFibo2 node accepts text alone. The derived graph retains the official structured negative, 50 steps, CFG 5, Euler/simple, and uses 1024 × 1024. The existing upstream MIT license applies.

`config/creation-models.json` pins the additional Dev INT8 file to the same Comfy-Org/Fibo-2.0 revision as the edit assets. Bria documents Dev for text-to-image and Edit Turbo for editing. This optional model adds 8,408,562,474 bytes; both workflows share the existing encoder and VAE.
