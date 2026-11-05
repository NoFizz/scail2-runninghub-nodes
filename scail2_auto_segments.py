# -*- coding: utf-8 -*-
"""SCAIL-2 全自动分段节点（v1 API）
把「算段数 → 逐段切片 → 采样 → 解码 → 丢重叠帧 → 段间调色 → 拼接」
整条流水线收进一个节点，画布上不再需要复制 Extend 子图。

用法：LoadVideo 整条视频 → 本节点 → CreateVideo → SaveVideo
模式切换沿用节点上的 replacement_mode 布尔。

参数语义与官方 WanSCAILToVideo 一致；分段公式沿用官方模板
`ceil(total / (frame_count - previous_frame_count))`。
"""

import math
import torch

import comfy.sample
import comfy.utils
import comfy.model_management
try:
    # Reuse ComfyUI core's official SCAIL-2 segment-conditioning implementation.
    # The local wrapper below only orchestrates multi-segment sampling/overlap.
    from comfy_extras.nodes_scail import WanSCAILToVideo
except ImportError as _e:
    raise ImportError(
        "SCAIL2AutoSegments 需要包含 WanSCAILToVideo 的 ComfyUI core "
        "（基准为 v0.44.0）。请升级 ComfyUI 后重启；旧版本缺少 SCAIL-2/SAM3 核心节点。"
    ) from _e

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _build_chunk(positive, negative, vae, width, height, length, batch_size,
                 pose_strength, pose_start, pose_end, video_frame_offset,
                 previous_frame_count, replacement_mode,
                 reference_image=None, clip_vision_output=None,
                 pose_video=None, pose_video_mask=None,
                 reference_image_mask=None, previous_frames=None):
    """Delegate one chunk's conditioning/latent construction to ComfyUI core.

    This file retains only the multi-segment orchestration. The SCAIL-2
    conditioning implementation, temporal masks, reference latents and
    previous-frame noise mask are provided by WanSCAILToVideo in core.
    """
    result = WanSCAILToVideo.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        width=width,
        height=height,
        length=length,
        batch_size=batch_size,
        pose_strength=pose_strength,
        pose_start=pose_start,
        pose_end=pose_end,
        video_frame_offset=video_frame_offset,
        previous_frame_count=previous_frame_count,
        replacement_mode=replacement_mode,
        reference_image=reference_image,
        clip_vision_output=clip_vision_output,
        pose_video=pose_video,
        pose_video_mask=pose_video_mask,
        reference_image_mask=reference_image_mask,
        previous_frames=previous_frames,
    )
    if result is None or result.result is None or len(result.result) < 3:
        raise RuntimeError("ComfyUI core WanSCAILToVideo did not return chunk conditioning and latent")
    positive, negative, latent = result.result[:3]
    return positive, negative, latent


def _color_match(tgt, ref_frame):
    """把生成段按锚定帧做均值/方差匹配，压掉段间色差（reinhard 思路的 RGB 近似）

    两个方差都要保护：
      - ts（目标段方差）为 0：没有可拉伸的细节，原样返回
      - rs（参考帧方差）为 0：参考帧是纯色（黑场/白场/淡入淡出/转场帧），
        拿它当标准差的来源会把整段压成单色。此时退化为只做均值平移，
        保留段内原有的对比度——否则一个黑场锚定帧就能毁掉整段画面。
    """
    out = tgt.clone()
    for c in range(out.shape[-1]):
        t = out[..., c]
        r = ref_frame[..., c]
        tm, ts = float(t.mean()), float(t.std())
        rm, rs = float(r.mean()), float(r.std())
        if ts > 1e-5:
            if rs > 1e-5:
                # 正常的均值+方差匹配
                t = (t - tm) / ts * rs + rm
            else:
                # 参考帧近乎纯色：只做均值平移，不压缩对比度
                t = t - tm + rm
        out[..., c] = t.clamp(0.0, 1.0)
    return out


class SCAIL2AutoSegments:
    """整条驱动视频一次跑完所有分段，输出拼接好的完整帧序列。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "pose_video": ("IMAGE",),
                "pose_video_mask": ("IMAGE",),
                # 填 0 = 自动跟随上游 SCAIL2Prep 的尺寸，避免两处手写不同步
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                                  "tooltip": "0 = 自动跟随 pose_video 的尺寸（推荐）。"
                                             "填非 0 则必须与 SCAIL2Prep 的 width 一致，且是 32 的倍数"}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                                   "tooltip": "0 = 自动跟随 pose_video 的尺寸（推荐）"}),
                "frame_count": ("INT", {"default": 81, "min": 9, "max": 256, "step": 4,
                                        "tooltip": "每段生成帧数，官方训练档 81"}),
                "previous_frame_count": ("INT", {"default": 5, "min": 1, "max": 20, "step": 4,
                                                 "tooltip": "段间重叠的锚定帧数，官方训练档 5"}),
                "replacement_mode": ("BOOLEAN", {"default": False,
                                                 "tooltip": "False=动画模式（驱动掩码黑底）；True=替换模式（驱动掩码白底）"}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 20261104, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                # "unipc" 不是合法采样器名（正确写法是 "uni_pc"），
                # 传错名字 KSampler 会静默降级成 euler，用户选了也没效果。
                "sampler_name": (["euler", "euler_ancestral", "dpmpp_2m", "uni_pc"],
                                 {"default": "euler"}),
                "scheduler": (["simple", "normal", "beta", "karras"], {"default": "simple"}),
                "color_match": ("BOOLEAN", {"default": True,
                                            "tooltip": "段间以上一段锚定帧做均值/方差匹配，压色差"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "reference_image_mask": ("IMAGE",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "free_between_segments": ("BOOLEAN", {"default": True,
                                                      "tooltip": "每段结束后清理显存，防止水位逐段上涨"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frames", "total_frames", "report")
    FUNCTION = "run"
    CATEGORY = "SCAIL-2/auto"

    def run(self, model, vae, positive, negative, pose_video, pose_video_mask,
            width, height, frame_count, previous_frame_count, replacement_mode,
            pose_strength, pose_start, pose_end, seed, steps, cfg,
            sampler_name, scheduler, color_match,
            reference_image=None, reference_image_mask=None, clip_vision_output=None,
            free_between_segments=True):

        total = int(pose_video.shape[0])
        if total < 1:
            raise ValueError("pose_video 是空张量，没有可生成的帧")
        # 最短可用输入：至少 9 帧。少于 9 帧时末段取整后 length 会 <5 被跳过，
        # 导致 outputs 为空、最后抛「没有任何有效分段产出」这种看不懂的错。
        # 提前给一句能照着做的提示。
        if total < 9:
            raise ValueError(
                f"驱动视频抽帧后只有 {total} 帧，太短了：SCAIL-2 至少需要 9 帧"
                f"（16fps 下约 0.6 秒）才能生成一段有效画面。"
                f"请换一条至少 1 秒的驱动视频。不要为此改 target_fps——"
                f"帧率链路固定 16fps 生成、30fps 打包，改它会让动作变速。")

        # 尺寸与上游 SCAIL2Prep 的同步：width/height 传 0 表示「自动跟随 pose_video」，
        # 避免用户改了一处忘了另一处，导致画面与 latent 尺寸不一致（不报错、只悄悄变差）。
        if width == 0 or height == 0:
            auto_h, auto_w = int(pose_video.shape[1]), int(pose_video.shape[2])
            width, height = auto_w, auto_h
            print(f"[SCAIL2Auto] width/height 传 0 → 自动跟随 pose_video：{width}x{height}",
                  flush=True)

        # SCAIL-2 官方要求：宽高必须都是 32 的倍数。
        # widget 的 step=32 只是 UI 步进，用户能手打任意值，这里做强校验。
        if width % 32 != 0 or height % 32 != 0:
            # 建议值本身必须合法：向下取整得到 0 时改用向上取整
            sug_w = (width // 32) * 32 or ((width // 32) + 1) * 32
            sug_h = (height // 32) * 32 or ((height // 32) + 1) * 32
            raise ValueError(
                f"width/height 必须是 32 的倍数（SCAIL-2 官方硬性要求），"
                f"当前 {width}x{height}。建议改成 {sug_w}x{sug_h}")

        # previous_frame_count 必须满足 4n+1：VAE 时域压缩 4 倍，
        # 编码 pfc 帧得到的像素帧数是 4*((pfc-1)//4+1)-3，
        # 而丢弃时按 pfc 丢——两者不等就会造成段间跳帧。
        # 例：pfc=20（UI 的 max）实际只锚定 17 帧却丢 20 帧，接缝跳 3 帧。
        if (previous_frame_count - 1) % 4 != 0:
            good = ((previous_frame_count - 1) // 4) * 4 + 1
            raise ValueError(
                f"previous_frame_count 必须是 4n+1（1/5/9/13/17…），"
                f"当前 {previous_frame_count} 会导致段间锚定错位。建议改成 {good}")

        # pfc 必须小于 frame_count，否则步进会被兜成 1，段数爆炸
        if previous_frame_count >= frame_count:
            raise ValueError(
                f"previous_frame_count（{previous_frame_count}）必须小于 "
                f"frame_count（{frame_count}），否则每段只在推进 1 帧，"
                f"会产生大量无效分段")

        # frame_count 也必须是 4n+1：VAE 时域 4 倍压缩决定的。
        # 不是 4n+1 的话每段都要向下取整，等于每段都悄悄丢 1~3 帧。
        if (frame_count - 1) % 4 != 0:
            good_lo = ((frame_count - 1) // 4) * 4 + 1
            raise ValueError(
                f"frame_count 必须是 4n+1（81/121/161…），当前 {frame_count} 会导致"
                f"每段都向下取整、每段丢 1~3 帧。建议改成 {good_lo} 或 {good_lo + 4}。\n"
                f"显存够的话可以把 frame_count 调大以减少段数：81（8G 稳妥）/ "
                f"121 / 161（24G 可试）")

        if pose_video_mask is not None and int(pose_video_mask.shape[0]) < total:
            raise ValueError(
                f"pose_video_mask 只有 {int(pose_video_mask.shape[0])} 帧，"
                f"比 pose_video 的 {total} 帧少，掩码与画面会错位。"
                f"检查 SAM3 追踪是否丢帧。")

        step = max(1, frame_count - previous_frame_count)
        # 段数：首段覆盖 1..frame_count，之后每段推进 step。
        # 第 k 段起点 = step*k，只要 step*k < total 就得再跑一段，
        # 所以是 ceil((total - previous_frame_count) / step)。
        # 旧写法 ceil(total/step) 会多算一段（total=81 时算出 2 段，实际只需 1 段），
        # 多出的那一段只剩 5 帧、锚定占满、采样不出新内容，最后会塞进一帧冻结画面。
        n_seg = max(1, math.ceil((total - previous_frame_count) / step))

        lines = [f"总帧数={total} 每段={frame_count} 重叠={previous_frame_count} "
                 f"步进={step} 段数={n_seg} 模式={'替换' if replacement_mode else '动画'}"]

        prev_anchor = None      # 上一段输出的最后 previous_frame_count 帧（未裁剪，作锚定）
        prev_last = None        # 【上一段】最后一帧，作调色参考（必须在循环末尾更新）
        outputs = []

        pbar = comfy.utils.ProgressBar(n_seg)
        for i in range(n_seg):
            start = step * i
            remain = total - start
            if remain < 1:
                break
            length = min(frame_count, remain)
            # 向下取整到 4n+1（模型要求），最多丢 3 帧，记进报告里
            if (length - 1) % 4 != 0:
                rounded = ((length - 1) // 4) * 4 + 1
                lines.append(f"段{i+1}: 末段 {length} 帧取整到 {rounded}（丢 {length-rounded} 帧）")
                length = rounded
            if length < 5:
                lines.append(f"段{i+1}: 剩余 {length} 帧不足 5，跳过（丢 {length} 帧）")
                break

            pv = pose_video[start:start + length]
            pm = pose_video_mask[start:start + length] if pose_video_mask is not None else None

            pos, neg, latent = _build_chunk(
                positive, negative, vae, width, height, length, 1,
                pose_strength, pose_start, pose_end,
                video_frame_offset=0,
                previous_frame_count=previous_frame_count,
                replacement_mode=replacement_mode,
                reference_image=reference_image,
                clip_vision_output=clip_vision_output,
                pose_video=pv,
                pose_video_mask=pm,
                reference_image_mask=reference_image_mask,
                previous_frames=prev_anchor,
            )

            # seed 上限是 0xffffffffffffffff，直接 +i 会溢出 int64
            # （torch.manual_seed(2**64) 会报 Overflow when unpacking long long）
            seg_seed = (int(seed) + i) & 0xFFFFFFFFFFFFFFFF
            samples = comfy.sample.sample(
                model,
                comfy.sample.prepare_noise(latent["samples"], seg_seed, None),
                steps, cfg, sampler_name, scheduler,
                pos, neg, latent["samples"],
                denoise=1.0,
                noise_mask=latent.get("noise_mask"),
                disable_pbar=True,
                seed=seg_seed,
            )
            decoded = vae.decode(samples)
            # Wan 视频 VAE 的 decode 返回 5 维 [B, T, H, W, C]，必须并批成 4 维 [T, H, W, C]。
            # 官方 VAEDecode 节点在 nodes.py 里有同样一句 "Combine batches"。
            # 漏了这句会连锁出三个错误：
            #   1) decoded[-5:] 切的是 batch 维，拿到整段而不是最后 5 帧
            #   2) trimmed.shape[0] == 1（batch 数），日志会显示"保留=1"
            #   3) pf.shape[0] == 1 → 补帧时 .repeat(8,1,1,1) 维度对不上直接崩
            if len(decoded.shape) == 5:
                decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])

            # 丢与上一段重叠的前 previous_frame_count 帧
            trimmed = decoded[previous_frame_count:] if i > 0 else decoded
            if trimmed.shape[0] < 1:
                # 本段生成的全是和上一段重叠的锚定帧，没有新内容。
                # （length == previous_frame_count 时会走到这里）
                # 不塞重复帧，直接记进报告跳过。
                lines.append(f"段{i+1}: 生成长度 {length} 帧全是重叠锚定，无新内容，跳过")
                prev_anchor = decoded[-previous_frame_count:].clone()
                prev_last = decoded[-1:].clone()
                pbar.update(1)
                continue

            # ★ 用【上一段】的尾帧做调色参考。旧代码在取参考之前就把 prev_last
            #   更新成本段的尾帧了，导致 _color_match 拿自己校自己、恒等变换，
            #   段间色差一帧都没压掉。
            if i > 0 and color_match and prev_last is not None:
                trimmed = _color_match(trimmed, prev_last)

            outputs.append(trimmed)
            lines.append(f"段{i+1}/{n_seg}: 源起点={start} 生成={length} 保留={trimmed.shape[0]}")
            print(f"[SCAIL2Auto] {lines[-1]}", flush=True)

            # ★ 状态在用到之后才更新
            prev_anchor = decoded[-previous_frame_count:].clone()
            prev_last = decoded[-1:].clone()

            if free_between_segments and i < n_seg - 1:
                comfy.model_management.soft_empty_cache()

            pbar.update(1)

        if not outputs:
            raise RuntimeError("没有任何有效分段产出")

        frames = torch.cat(outputs, dim=0)
        got = int(frames.shape[0])
        # 帧数对账：源 N 帧 → 成片 M 帧，差多少明确写出来，不再静默
        delta = got - total
        if delta != 0:
            lines.append(f"⚠ 帧数对账：源 {total} 帧 → 成片 {got} 帧（{delta:+d}）；"
                         f"末段不足 4n+1 时按模型要求向下取整，属预期损耗")
        else:
            lines.append(f"帧数对账：源 {total} 帧 → 成片 {got} 帧（一致）")
        report = "\n".join(lines) + f"\n成片总帧数={got}（16fps ≈ {got/16:.2f}s）"
        print("[SCAIL2Auto] " + report, flush=True)
        return (frames, int(got), report)


NODE_CLASS_MAPPINGS["SCAIL2AutoSegments"] = SCAIL2AutoSegments
NODE_DISPLAY_NAME_MAPPINGS["SCAIL2AutoSegments"] = "SCAIL-2 全自动分段（整条视频）"
