# SCAIL-2 RunningHub support nodes

Minimal source repository for the two project-specific ComfyUI node classes used by the SCAIL-2 workflow:

| File | Registered node | Role |
|---|---|---|
| `scail2_load_video.py` | `SCAIL2LoadVideo` | Stream-decodes a `VIDEO`, downsizes/crops frames, resamples to the configured fps and extracts the source audio. |
| `scail2_auto_segments.py` | `SCAIL2AutoSegments` | Orchestrates the model's whole-video chunk loop, overlap handling, sampling, decode, and frame concatenation. Its chunk conditioning/latent construction delegates to ComfyUI core `WanSCAILToVideo`; this source does not copy that core implementation. |

## Runtime baseline

- ComfyUI core with SCAIL-2/SAM3 support, including `comfy_extras.nodes_scail.WanSCAILToVideo`. The locally checked baseline is ComfyUI `v0.44.0`; other releases must be checked for the required API and model support.
- ComfyUI's normal `torch`/`av` (PyAV) runtime. Do not copy a Windows virtual environment to Linux.
- The workflow separately depends on the upstream [ComfyUI-Tween](https://github.com/Ethanfel/ComfyUI-Tween) nodes for BiM-VFI. This repository intentionally does not mirror Tween.

## Scope and limitations

This repository contains node source only. It includes **no** model/checkpoint weights, workflow JSON, personal input video/images, ComfyUI environment, credentials, local filesystem paths, or unpublished platform code. It does not itself install the nodes on RunningHub; the platform must accept and integrate them.

The streaming loader assumes constant-frame-rate footage at or above the configured 16fps for correct duration semantics; low-fps duplication and VFR timestamp handling need further work. Audio decode failures can result in a silent output. The segment node's decoded frame tensors still accumulate with video duration, so it is not constant-memory for arbitrarily long clips.

Model and dependency licensing is separate from this repository's node-source license. In particular, the BiM-VFI model/component has upstream research/education-only terms; SAM3.1 uses the SAM License. This repository's MIT license does not grant rights to either model, to ComfyUI/Tween, or to media assets.

## License

The two custom-node files in this repository are released under the MIT License; see [`LICENSE`](LICENSE). This license applies only to these files. All upstream code, models, checkpoints, and other dependencies retain their own licenses and terms.
