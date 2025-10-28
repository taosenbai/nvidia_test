#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import shlex
from pathlib import Path
from typing import Optional, Tuple

RES_1080 = {(1920, 1080), (1080, 1920)}
RES_4K = {(3840, 2160), (2160, 3840)}  # 严格 4K（不兼容 3820×2160）

RES_PATTERN = re.compile(r'(\d{3,5})[xX](\d{3,5})')

def which_or_raise(bin_name: str):
    if shutil.which(bin_name) is None:
        raise RuntimeError(f"未找到依赖程序: {bin_name}，请先安装 GStreamer（gst-launch-1.0）。")

def find_resolutions_in_name(name: str):
    return [(int(m.group(1)), int(m.group(2))) for m in RES_PATTERN.finditer(name)]

def choose_resolution(name: str) -> Optional[Tuple[int, int]]:
    cands = find_resolutions_in_name(name)
    if not cands:
        return None
    for w, h in cands:
        if (w, h) in RES_4K or (w, h) in RES_1080:
            return (w, h)
    return max(cands, key=lambda wh: wh[0] * wh[1])

def decide_bitrate_kbps(w: int, h: int, default_kbps: int) -> int:
    if (w, h) in RES_1080:
        return 3000
    if (w, h) in RES_4K:
        return 12000
    return default_kbps

def build_pipeline_args(
    src: Path,
    dst: Path,
    width: int,
    height: int,
    bitrate_kbps: int,
    fps: int,
    pix_format: str,
    profile: str = "main",
):
    return [
        "gst-launch-1.0",
        "-e",
        "filesrc", f"location={str(src)}",
        "!",
        "rawvideoparse",
        f"format={pix_format}",
        f"width={width}",
        f"height={height}",
        f"framerate={fps}/1",
        "!",
        "x265enc",
        f"bitrate={bitrate_kbps}",
        "!",
        f"video/x-h265,profile={profile}",
        "!",
        "filesink", f"location={str(dst)}",
    ]

def encode_one(
    yuv_path: Path,
    out_dir_name: str,
    fps: int,
    pix_format: str,
    default_bitrate_kbps: int,
    overwrite: bool,
    dry_run: bool,
) -> bool:
    res = choose_resolution(yuv_path.name)
    if res is None:
        print(f"[跳过] 未在文件名解析到分辨率: {yuv_path}")
        return False
    w, h = res
    bitrate_kbps = decide_bitrate_kbps(w, h, default_bitrate_kbps)

    out_dir = yuv_path.parent / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = yuv_path.stem + ".h265"
    out_path = out_dir / out_name

    if out_path.exists() and not overwrite:
        print(f"[跳过] 目标已存在（使用 --overwrite 覆盖）: {out_path}")
        return False

    cmd = build_pipeline_args(
        src=yuv_path,
        dst=out_path,
        width=w,
        height=h,
        bitrate_kbps=bitrate_kbps,
        fps=fps,
        pix_format=pix_format,
        profile="main",
    )

    print(f"[编码] {yuv_path.name} -> {out_path.name} | {w}x{h} @{fps}fps | {bitrate_kbps} kbps | i420 main")
    if dry_run:
        print("  命令:", " ".join(shlex.quote(part) for part in cmd))
        return False

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(f"[失败] {yuv_path}\n{proc.stdout}")
        return False
    else:
        print(f"[完成] {out_path}")
        return True

def iter_yuv_files(root: Path, recursive: bool = True):
    if recursive:
        for p in root.rglob("*.yuv"):
            if p.is_file():
                yield p
    else:
        for p in root.glob("*.yuv"):
            if p.is_file():
                yield p

def main():
    parser = argparse.ArgumentParser(
        description="批量使用 GStreamer x265enc 编码目录下的 YUV 文件（输出 ES 裸流 .h265）"
    )
    parser.add_argument("input_dir", help="输入目录路径")
    parser.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    parser.add_argument("--fps", type=int, default=30, help="帧率（默认 30）")
    parser.add_argument("--format", default="i420", help="像素格式（默认 i420）")
    parser.add_argument("--default-bitrate", type=int, default=3000, help="未知分辨率的默认码率 kbps（默认 3000）")
    parser.add_argument("--out-subdir", default="x265enc", help="输出子目录名（默认 x265enc）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令不执行")
    args = parser.parse_args()

    root = Path(args.input_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"输入目录不存在或不可用: {root}")
        raise SystemExit(2)

    try:
        which_or_raise("gst-launch-1.0")
    except RuntimeError as e:
        print(str(e))
        raise SystemExit(3)

    total = 0
    success = 0
    for yuv in iter_yuv_files(root, recursive=not args.no_recursive):
        total += 1
        ok = encode_one(
            yuv_path=yuv,
            out_dir_name=args.out_subdir,
            fps=args.fps,
            pix_format=args.format,
            default_bitrate_kbps=args.default_bitrate,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if ok:
            success += 1

    if total == 0:
        print("未找到任何 .yuv 文件。")
    print(f"[统计] 输入文件: {total}, 成功编码: {success}")

if __name__ == "__main__":
    main()
