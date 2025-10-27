#!/usr/bin/env python3
"""
将指定目录下的所有 MP4 文件使用 GStreamer 解码为原始 YUV（默认 I420）并输出到同级 yuv 子目录。

使用示例：
  python3 scripts/mp4_to_yuv_gst.py /path/to/dir
  python3 scripts/mp4_to_yuv_gst.py . --no-recursive --overwrite --format I420

依赖：
  - GStreamer 1.x（需要 gst-launch-1.0 可执行程序）
  - GStreamer 的基础与解码插件（能够解码 MP4/H.264 等）
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_gst_launch() -> str | None:
    """查找系统上的 gst-launch 可执行文件。"""
    for candidate in ("gst-launch-1.0", "gst-launch"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def build_pipeline_command(
    gst_launch: str, input_path: Path, output_path: Path, pixel_format: str
) -> list[str]:
    """构建将 MP4 解码为原始 YUV 的 GStreamer 命令。"""
    uri = input_path.resolve().as_uri()
    return [
        gst_launch,
        "-e",
        "-q",
        "uridecodebin",
        f"uri={uri}",
        "!",
        "videoconvert",
        "!",
        f"video/x-raw,format={pixel_format}",
        "!",
        "filesink",
        f"location={str(output_path)}",
    ]


def convert_one(
    gst_launch: str,
    mp4_path: Path,
    out_dir_name: str,
    pixel_format: str,
    overwrite: bool,
) -> tuple[bool, str]:
    """将单个 MP4 解码为 YUV 文件。返回 (是否成功, 消息)。"""
    yuv_dir = mp4_path.parent / out_dir_name
    yuv_dir.mkdir(parents=True, exist_ok=True)
    out_path = yuv_dir / mp4_path.with_suffix(".yuv").name

    if out_path.exists():
        if overwrite:
            try:
                out_path.unlink()
            except Exception as exc:  # noqa: BLE001
                return False, f"无法删除已存在文件: {out_path} ({exc})"
        else:
            return False, f"skip exists: {out_path}"

    cmd = build_pipeline_command(gst_launch, mp4_path, out_path, pixel_format)
    try:
        subprocess.run(cmd, check=True)
        return True, str(out_path)
    except FileNotFoundError:
        return False, "未找到 gst-launch-1.0，请先安装 GStreamer"
    except subprocess.CalledProcessError as exc:
        return False, f"转换失败（退出码 {exc.returncode}）"


def collect_mp4_files(base_dir: Path, recursive: bool) -> list[Path]:
    """收集指定目录（可递归）下的所有 .mp4 文件（大小写不敏感）。"""
    if recursive:
        candidates = list(base_dir.rglob("*.mp4")) + list(base_dir.rglob("*.MP4"))
    else:
        candidates = list(base_dir.glob("*.mp4")) + list(base_dir.glob("*.MP4"))

    # 去重（按绝对路径）
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        abspath = str(p.resolve())
        if abspath not in seen:
            seen.add(abspath)
            uniq.append(p)
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 GStreamer 将目录下的 MP4 解码为原始 YUV (默认 I420)"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="待遍历的目录（默认：当前目录）",
    )
    parser.add_argument(
        "--format",
        dest="pixel_format",
        default="I420",
        help="输出 YUV 像素格式（默认：I420）",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="递归遍历子目录（默认：是）",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="不递归遍历子目录",
    )
    parser.add_argument(
        "--out-dir-name",
        default="yuv",
        help="输出的同级子目录名（默认：yuv）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标 YUV 已存在则覆盖",
    )

    args = parser.parse_args()

    base_dir = Path(args.directory).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"目录不存在或不是目录: {base_dir}", file=sys.stderr)
        sys.exit(2)

    gst_launch = find_gst_launch()
    if not gst_launch:
        print(
            "未找到 GStreamer 可执行文件（gst-launch-1.0）。请先安装 GStreamer。",
            file=sys.stderr,
        )
        sys.exit(3)

    mp4_files = collect_mp4_files(base_dir, args.recursive)
    if not mp4_files:
        print(f"未在目录中发现 MP4 文件: {base_dir}")
        return

    total = len(mp4_files)
    success_count = 0
    for mp4_path in mp4_files:
        ok, msg = convert_one(
            gst_launch,
            mp4_path,
            args.out_dir_name,
            args.pixel_format,
            args.overwrite,
        )
        prefix = "完成" if ok else ("跳过" if msg.startswith("skip") else "失败")
        print(f"[{prefix}] {mp4_path} -> {msg}")
        if ok:
            success_count += 1

    print(f"转换完成：成功 {success_count}/{total}")


if __name__ == "__main__":
    main()



