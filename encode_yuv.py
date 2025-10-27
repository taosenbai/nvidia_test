#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

def find_gst_launch() -> Optional[str]:
    for name in ("gst-launch-1.0", "gst-launch"):
        p = shutil.which(name)
        if p:
            return p
    return None

def parse_resolution_from_name(name: str) -> Optional[Tuple[int, int, str]]:
    m = re.search(r"(\d{3,5})[xX](\d{3,5})", name)
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    pair = {w, h}
    if pair == {1920, 1080}:
        return w, h, "1080p"
    if pair == {3840, 2160}:  # 仅 3840x2160/2160x3840 视为 4K
        return w, h, "4k"
    return None

def collect_yuv_files(base: Path, recursive: bool) -> list[Path]:
    if recursive:
        cands = list(base.rglob("*.yuv")) + list(base.rglob("*.YUV"))
    else:
        cands = list(base.glob("*.yuv")) + list(base.glob("*.YUV"))
    seen: set[str] = set()
    out: list[Path] = []
    for p in cands:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out

def build_common_src_segment(yuv: Path, pixel_format: str, w: int, h: int, fps: int) -> list[str]:
    # 解析原始文件使用 rawvideoparse；随后统一用 videoconvert 转为 I420 以便编码器稳定协商
    fmt_map = {
        "I420": "i420",
        "NV12": "nv12",
        "YV12": "yv12",
    }
    src_fmt = fmt_map.get(pixel_format.upper())
    if not src_fmt:
        raise ValueError(f"不支持的像素格式: {pixel_format}")
    return [
        "filesrc", f"location={str(yuv)}",
        "!", "rawvideoparse",
        f"format={src_fmt}", f"width={w}", f"height={h}", f"framerate={fps}/1",
        "!", "videoconvert",
        "!", "video/x-raw,format=I420",  # 统一为 I420，避免协商失败
        "!", "queue"
    ]

def build_pipeline_cmd(
    gst: str,
    src_seg: list[str],
    enc_name: str,
    codec: str,  # "h264" or "h265"
    bitrate_kbps: int,
    profile: str,
    extra_props: list[str],
    out_path: Path,
) -> list[str]:
    parse_elem = "h264parse" if codec == "h264" else "h265parse"
    caps_profile = f"video/x-{codec},profile={profile}"
    caps_stream = f"video/x-{codec},stream-format=byte-stream,alignment=au"

    enc_props: list[str] = [f"bitrate={bitrate_kbps}"]
    # 某些环境无 profile 属性；对 x264enc/x265enc 不直接设 profile，交给 caps 约束
    enc_props += extra_props

    return [
        gst, "-e", "-q",
        *src_seg,
        "!", enc_name, *enc_props,
        "!", caps_profile,                     # 通过 caps 约束 profile
        "!", parse_elem, "config-interval=1",  # 插入 SPS/PPS
        "!", caps_stream,                      # ES + AU 对齐
        "!", "filesink", f"location={str(out_path)}"
    ]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def encode_one(
    gst: str,
    yuv: Path,
    w: int,
    h: int,
    size_class: str,
    fps: int,
    pixel_format: str,
    overwrite: bool,
) -> None:
    h264_map = {"1080p": 5000, "4k": 20000}  # kbps
    h265_map = {"1080p": 3000, "4k": 12000}  # kbps

    src_seg = build_common_src_segment(yuv, pixel_format, w, h, fps)

    plans = [
        ("x264enc",  "h264", "x264enc",   h264_map[size_class], "high",            [],            ".h264"),
        ("nvh264enc","h264", "nvh264enc", h264_map[size_class], "high",            ["preset=hq"], ".h264"),
        ("x265enc",  "h265", "x265enc",   h265_map[size_class], "main",            [],            ".h265"),
        ("nvh265enc","h265", "nvh265enc", h265_map[size_class], "main",            ["preset=hq"], ".h265"),
    ]

    for out_subdir, codec, enc_name, br, profile, extra, ext in plans:
        out_dir = yuv.parent / out_subdir
        ensure_dir(out_dir)
        out_path = out_dir / yuv.with_suffix(ext).name
        if out_path.exists() and not overwrite:
            print(f"[跳过] 目标已存在: {out_path}")
            continue

        cmd = build_pipeline_cmd(
            gst=gst,
            src_seg=src_seg,
            enc_name=enc_name,
            codec=codec,
            bitrate_kbps=br,
            profile=profile,
            extra_props=extra,
            out_path=out_path,
        )
        try:
            subprocess.run(cmd, check=True)
            print(f"[完成] {yuv} -> {out_path} ({enc_name}, {profile}, {br}kbps)")
        except FileNotFoundError:
            print(f"[失败] 未找到 {enc_name}（插件可能未安装/不支持平台）: {yuv}")
        except subprocess.CalledProcessError as exc:
            print(f"[失败] {enc_name} 编码失败（退出码 {exc.returncode}）: {yuv}")

def main() -> None:
    ap = argparse.ArgumentParser(description="批量将 YUV 编码为 H.264/H.265 ES（按 1080p/4K 预设码率）")
    ap.add_argument("directory", nargs="?", default=".", help="待遍历目录（默认当前目录）")
    ap.add_argument("--format", dest="pixel_format", default="I420", help="源 YUV 像素格式（默认 I420）")
    ap.add_argument("--fps", type=int, default=30, help="帧率（默认 30）")
    ap.add_argument("--recursive", dest="recursive", action="store_true", default=True, help="递归（默认是）")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false", help="不递归")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在输出文件")
    args = ap.parse_args()

    base = Path(args.directory).resolve()
    if not base.exists() or not base.is_dir():
        print(f"目录不存在或不是目录: {base}", file=sys.stderr)
        sys.exit(2)

    gst = find_gst_launch()
    if not gst:
        print("未找到 gst-launch-1.0，请安装 GStreamer。", file=sys.stderr)
        sys.exit(3)

    files = collect_yuv_files(base, args.recursive)
    if not files:
        print(f"未找到 .yuv 文件: {base}")
        return

    total, done = len(files), 0
    for yuv in files:
        parsed = parse_resolution_from_name(yuv.name)
        if not parsed:
            print(f"[跳过] 无法从文件名解析分辨率或不支持（仅 1080p / 3840x2160）: {yuv.name}")
            continue
        w, h, cls = parsed
        try:
            encode_one(
                gst=gst,
                yuv=yuv,
                w=w,
                h=h,
                size_class=cls,
                fps=args.fps,
                pixel_format=args.pixel_format,
                overwrite=args.overwrite,
            )
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"[失败] 未处理异常: {yuv} ({e})")

    print(f"处理完成：编码成功 {done}/{total} 个文件")

if __name__ == "__main__":
    main()
