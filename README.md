# SCAIL-2 RunningHub support nodes

Minimal source repository for the two project-specific ComfyUI node classes used by the SCAIL-2 workflow:

| File | Registered node | Role |
|---|---|---|
| `scail2_load_video.py` | `SCAIL2LoadVideo` | Stream-decodes a `VIDEO`, downsizes/crops frames, resamples to the configured fps and extracts the source audio. |
| `scail2_auto_segments.py` | `SCAIL2AutoSegments` | Orchestrates whole-video chunk scheduling, sampling, decode, overlap/anchor handling, per-mode colour baseline matching, seam brightness transition, and concatenation. Its per-chunk SCAIL conditioning and latent construction are delegated to ComfyUI core `WanSCAILToVideo.execute`; this repository contains no copied implementation of that core function. |

## Runtime baseline

- ComfyUI core with SCAIL-2/SAM3 support, including `comfy_extras.nodes_scail.WanSCAILToVideo`. The checked baseline is ComfyUI `v0.44.0`; other releases must be checked for the required API and model support.
- ComfyUI's normal `torch`/`av` (PyAV) runtime. Do not copy a Windows virtual environment to Linux.
- The workflow separately depends on upstream [ComfyUI-Tween](https://github.com/Ethanfel/ComfyUI-Tween) for BiM-VFI. This repository intentionally does not mirror Tween.

## Colour handling between segments

`SCAIL2AutoSegments` exposes four widgets that control colour continuity across segment boundaries:

| Widget | Default | Range | Meaning |
|---|---|---|---|
| `color_match` | `true` | bool | Master switch for baseline colour matching |
| `color_match_strength` | `0.5` | 0.0–1.0 | How far each segment is moved toward the baseline; `0` = off, `1` = fully aligned |
| `seam_smooth_frames` | `4` | 0–16 | How many leading frames of each segment receive a brightness transition; `0` = off |
| `seam_smooth_strength` | `0.5` | 0.0–1.0 | Strength of that transition |

The baseline is **absolute and mode-dependent**:

- animation mode (`replacement_mode = false`) — every segment is aligned to the reference image;
- replacement mode (`replacement_mode = true`) — every segment is aligned to its own slice of the driving video.

It is deliberately **not** chained from the previous segment's output, so per-segment error does not accumulate over long clips. Only the per-channel mean is shifted; variance is left untouched, so contrast and sharpness are preserved. The seam transition applies a scalar brightness offset to the first few frames after the anchored prefix leaves the frozen region, decaying to zero; colour and spatial detail are not modified.

The defaults are the values used in production; there is normally no reason to change them. Set `color_match = false` to disable baseline matching entirely (the seam transition remains active).

## Scope and limitations

This repository contains only the two project node sources and this supporting README/license. It includes **no** model/checkpoint weights, workflow JSON, personal input video/images, ComfyUI environment, credentials, local filesystem paths, or unpublished platform code. It does not itself install the nodes on RunningHub; the platform must accept and integrate them.

The streaming loader assumes constant-frame-rate footage at or above the configured 16fps for correct duration semantics; low-fps duplication and VFR timestamp handling need further work. Audio decode failures can result in a silent output. The segment node's decoded frame tensors still accumulate with video duration, so it is not constant-memory for arbitrarily long clips. Automatic segmentation still runs each segment with its own seed (`seed + index`), so colour continuity depends on the baseline matching described above rather than on a shared sampler state; arbitrarily long clips are not guaranteed to be drift-free.

## Separate model/dependency permissions

Model and dependency licensing is separate from this repository's node-source license. In particular, BiM-VFI source/checkpoint has upstream research/education-only terms; commercial usage requires the upstream permission described in its README. SAM3.1 uses the SAM License. This repository's MIT license does not grant rights to either model, their weights, ComfyUI/Tween or media assets.

## License

The two project-specific custom-node files in this repository are released under the MIT License; see [`LICENSE`](LICENSE). This license applies only to these two source files. All upstream code, model weights, checkpoints, and other dependencies retain their own licenses and terms. The repository author's grant does not imply a ComfyUI license grant, and no warranty is made about RunningHub acceptance or operation.
