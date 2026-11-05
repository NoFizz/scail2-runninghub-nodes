# -*- coding: utf-8 -*-
"""SCAIL-2 流式视频加载：逐帧解码 + 立即缩放裁切，只保留需要的帧。

为什么要这个节点
────────────────
ComfyUI 原生的 `GetVideoComponents` 会把**整条视频**一次性解码成 float32 张量：

    1080×1920 × 465 帧 × 3 通道 × 4 字节 ≈ 11.6 GB 常驻内存

再叠上 Wan 14B int8 模型在 CPU 上的 15.9 GB 暂存，总占用冲到 31.5 GB / 32 GB（实测 96%）。
后果：**1080P 素材超过约 17 秒就会爆内存**。

本节点按「解一帧 → 立刻缩放到目标尺寸 → 丢掉原帧」的方式处理：

    内存 = 目标尺寸 × 输出帧数 = 512×896 × 249 帧 ≈ 1.4 GB

省掉 10 GB 以上，长视频/高分辨率素材不再受内存限制。

输出帧数、帧率、时长语义与 SCAIL2Prep + GetVideoComponents 组合完全一致。
"""

import math
import torch
import torch.nn.functional as F

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    import av
except Exception:                                    # 没装 pyav 时给出可读提示
    av = None


def _check32(w, h):
    """校验宽高是否为 32 的倍数（SCAIL-2 硬性要求，不做自动取整）。

    latent 空间 = 宽高 ÷ 8；28 通道掩码要三次 ÷2 降采样，
    两者必须严格相等，所以宽高必须能被 32 整除。
    用严格校验而不是自动取整：用户填多少就按多少，填错立刻报清楚。
    """
    if w % 32 == 0 and h % 32 == 0:
        return
    sug_w = max(32, (w // 32) * 32)
    sug_h = max(32, (h // 32) * 32)
    if sug_w < w:
        sug_w += 32
    if sug_h < h:
        sug_h += 32
    raise ValueError(
        f"width/height 必须是 32 的倍数（SCAIL-2 硬性要求），当前 {w}x{h}。"
        f"最接近的合法值是 {sug_w}x{sug_h}。\n"
        f"参考：512x896(512p) / 704x1280(704p) / 768x1344 / 896x1536(24G 可试) / 1088x1920")


def _center_crop_resize_frame(img, width, height):
    """单帧「保持比例缩放 + 中心裁切」，算法与 comfy.utils.common_upscale(..., "center") 一致。

    img: (H, W, C) float 0..1
    返回: (height, width, C)
    """
    H, W, C = img.shape
    if H == height and W == width:
        return img

    old_aspect = W / H
    new_aspect = width / height
    x = y = 0
    if old_aspect > new_aspect:                       # 原图更宽 → 左右裁
        x = round((W - W * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:                     # 原图更高 → 上下裁
        y = round((H - H * (old_aspect / new_aspect)) / 2)
    cw, ch = W - x * 2, H - y * 2
    if cw <= 0 or ch <= 0:
        cw, ch, x, y = W, H, 0, 0

    t = img[y:y + ch, x:x + cw, :].permute(2, 0, 1).unsqueeze(0)   # 1,C,H,W
    t = F.interpolate(t, size=(height, width), mode="bicubic",
                      align_corners=False, antialias=True)
    return t.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0)


def _decode_audio(source, container_factory):
    """单独走一遍只解音频，返回 ComfyUI 的 AUDIO 字典格式。"""
    if av is None:
        return None
    try:
        with container_factory() as c:
            astream = next((s for s in c.streams if s.type == "audio"), None)
            if astream is None:
                return None
            chunks, sr, nch = [], 0, 0
            for packet in c.demux(astream):
                for frame in packet.decode():
                    if sr == 0:
                        sr = int(frame.sample_rate)
                        nch = int(frame.layout.nb_channels)
                    arr = frame.to_ndarray()          # (channels, samples) float32
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    chunks.append(torch.from_numpy(arr.copy()))
            if not chunks or sr == 0:
                return None
            wave = torch.cat(chunks, dim=1).unsqueeze(0)      # (1, C, T)
            return {"waveform": wave, "sample_rate": sr}
    except Exception as e:
        print(f"[SCAIL2LoadVideo] 音频提取失败（不影响画面）：{e}", flush=True)
        return None


class SCAIL2LoadVideo:
    """逐帧流式解码，边解边缩放到目标尺寸，内存只跟输出有关。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "接 LoadVideo 的输出（保留上传按钮）"}),
                # 0 = 跟随源尺寸 / 保持原帧率
                # 档位自由填，但必须是 32 的倍数（节点会校验并给出最近的合法值）
                "width": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 32,
                                  "tooltip": "目标宽度，任意档位都行，但必须是 32 的倍数；"
                                             "填 0 = 用源宽度"}),
                "height": ("INT", {"default": 896, "min": 0, "max": 16384, "step": 32,
                                   "tooltip": "目标高度，任意档位都行，但必须是 32 的倍数；"
                                              "填 0 = 用源高度"}),
                "target_fps": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 240.0, "step": 0.01,
                                         "tooltip": "模型原生 16fps；填 0 = 保持源帧率"}),
                "with_audio": ("BOOLEAN", {"default": True,
                                           "tooltip": "是否一并提取原音（合成成片用）"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "report")
    FUNCTION = "run"
    CATEGORY = "SCAIL-2/auto"

    def run(self, video, width, height, target_fps=16.0, with_audio=True):
        if av is None:
            raise RuntimeError("缺少 PyAV 依赖，无法流式解码。请改用 LoadVideo + 拆视频帧 + SCAIL2Prep 那条链路")

        source = video.get_stream_source()
        if isinstance(source, str):
            factory = lambda: av.open(source, mode="r")
        else:                                        # BytesIO：每次重新 seek
            def factory():
                source.seek(0)
                return av.open(source, mode="r")

        # ---------- 先读元信息（不解码画面） ----------
        with factory() as c:
            vs = next((s for s in c.streams if s.type == "video"), None)
            if vs is None:
                raise ValueError("这个文件里没有视频流")
            src_w, src_h = int(vs.codec_context.width), int(vs.codec_context.height)
            try:
                src_fps = float(vs.average_rate) if vs.average_rate else 30.0
            except Exception:
                src_fps = 30.0
            if src_fps <= 0.01:
                src_fps = 30.0                      # VFR/识别失败时的兜底
            dur = 0.0
            if c.duration is not None:
                dur = c.duration / av.time_base
            elif vs.duration is not None and vs.time_base is not None:
                dur = float(vs.duration * vs.time_base)
            total_src = int(vs.frames) if vs.frames else int(round(dur * src_fps))

        if src_w <= 0 or src_h <= 0:
            raise ValueError("读不到视频尺寸，文件可能损坏")

        W = width if width > 0 else src_w
        H = height if height > 0 else src_h
        # 严格校验 32 的倍数（不自动取整）；填错直接报清楚并给最接近的合法值
        _check32(W, H)

        fps_out = target_fps if target_fps and target_fps > 0 else src_fps

        # ---------- 算出要保留哪些源帧（时间轴重采样，保持时长） ----------
        if dur <= 0 and total_src > 0:
            dur = total_src / src_fps
        n_out = max(1, int(round(dur * fps_out)))
        keep = []
        for i in range(n_out):
            src_idx = int(round((i / fps_out) * src_fps))
            keep.append(src_idx)
        keep_set = sorted(set(keep))
        keep_map = {v: k for k, v in enumerate(keep_set)}

        # ---------- 逐帧解码：只要的帧才转张量+缩放，其余立刻丢弃 ----------
        out_frames = []
        buf = {}
        idx = 0
        with factory() as c:
            vs = next((s for s in c.streams if s.type == "video"), None)
            vs.thread_type = "AUTO"
            for frame in c.decode(vs):
                if idx in keep_map:
                    arr = frame.to_ndarray(format="rgb24")           # (H,W,3) uint8
                    t = torch.from_numpy(arr).float() / 255.0
                    buf[idx] = _center_crop_resize_frame(t, W, H)
                idx += 1
                if idx > keep_set[-1]:
                    break

        if not buf:
            raise RuntimeError(f"没有解出任何帧（源帧数 {idx}，期望 {n_out}）")

        out_frames = [buf[i] for i in sorted(buf)]
        images = torch.stack(out_frames, dim=0)                      # (T,H,W,C)
        got = int(images.shape[0])

        audio = _decode_audio(source, factory) if with_audio else None

        # ---------- 帧数规整到 4n+1（向上取整，补末帧） ----------
        pad_note = ""
        if (got - 1) % 4 != 0:
            legal = ((got - 1) // 4) * 4 + 1
            if legal < got:
                legal += 4
            pad = legal - got
            images = torch.cat([images, images[-1:].repeat(pad, 1, 1, 1)], dim=0)
            pad_note = f"，末尾补 {pad} 帧重复帧凑 4n+1 → {legal} 帧"
            got = legal

        warn = ""
        if got < 9:
            warn = f"；⚠ 只有 {got} 帧，太短，下游可能无法生成"
        if target_fps and src_fps < target_fps:
            warn += f"；⚠ 源帧率 {src_fps:.2f} 低于目标 {fps_out:.0f}，存在复制帧"

        report = (f"流式解码：源 {src_w}x{src_h} @ {src_fps:.2f}fps ({idx}帧, {dur:.2f}s) "
                  f"-> {W}x{H} @ {fps_out:.0f}fps ({got}帧, {got/fps_out:.2f}s){pad_note}{warn}")
        print("[SCAIL2LoadVideo] " + report, flush=True)
        return (images, audio, float(fps_out), int(got), report)


NODE_CLASS_MAPPINGS["SCAIL2LoadVideo"] = SCAIL2LoadVideo
NODE_DISPLAY_NAME_MAPPINGS["SCAIL2LoadVideo"] = "SCAIL-2 流式加载（逐帧解码+缩放裁切）"
