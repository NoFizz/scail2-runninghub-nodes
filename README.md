# SCAIL-2 RunningHub support nodes

Minimal source repository for the two project-specific ComfyUI node classes used by the SCAIL-2 workflow:

| File | Registered node | Role |
|---|---|---|
| `scail2_load_video.py` | `SCAIL2LoadVideo` | Stream-decodes a `VIDEO`, downsizes/crops frames, resamples to the configured fps and extracts the source audio. |
| `scail2_auto_segments.py` | `SCAIL2AutoSegments` | Orchestrates whole-video chunk scheduling, sampling, decode, overlap/anchor handling, RGB color matching, and concatenation. Its per-chunk SCAIL conditioning and latent construction are delegated to ComfyUI core `WanSCAILToVideo.execute`; this repository contains no copied implementation of that core function. |

## Runtime baseline

- ComfyUI core with SCAIL-2/SAM3 support, including `comfy_extras.nodes_scail.WanSCAILToVideo`. The checked baseline is ComfyUI `v0.44.0`; other releases must be checked for the required API and model support.
- ComfyUI's normal `torch`/`av` (PyAV) runtime. Do not copy a Windows virtual environment to Linux.
- The workflow separately depends on upstream [ComfyUI-Tween](https://github.com/Ethanfel/ComfyUI-Tween) for BiM-VFI. This repository intentionally does not mirror Tween.

## Scope and limitations

This repository contains only the two project node sources and this supporting README/license. It includes **no** model/checkpoint weights, workflow JSON, personal input video/images, ComfyUI environment, credentials, local filesystem paths, or unpublished platform code. It does not itself install the nodes on RunningHub; the platform must accept and integrate them.

The streaming loader assumes constant-frame-rate footage at or above the configured 16fps for correct duration semantics; low-fps duplication and VFR timestamp handling need further work. Audio decode failures can result in a silent output. The segment node's decoded frame tensors still accumulate with video duration, so it is not constant-memory for arbitrarily long clips. Auto-segment orchestration delegates the per-chunk conditioning implementation to core; it still needs the compatible core and self node to schedule all chunks.

## Separate model/dependency permissions

Model and dependency licensing is separate from this repository's node-source license. In particular, BiM-VFI source/checkpoint has upstream research/education-only terms; commercial usage requires the upstream permission described in its README. SAM3.1 uses the SAM License. This repository's MIT license does not grant rights to either model, their weights, ComfyUI/Tween or media assets.

## License

The two project-specific custom-node files in this repository are released under the MIT License; see [`LICENSE`](LICENSE). This license applies only to these two source files. All upstream code, model weights, checkpoints, and other dependencies retain their own licenses and terms. The repository author's grant does not imply a ComfyUI license grant, and no warranty is made about RunningHub acceptance or operation.
