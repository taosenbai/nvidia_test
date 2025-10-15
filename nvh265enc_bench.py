import os
import time
import argparse
import subprocess
import pathlib

def build_encode_args(yuv_path, width=3840, height=2160, total_frames=564,
                      framerate="60/1", bitrate_kbps=15000, preset="hq"):
    # 等价于：
    # gst-launch-1.0 -e \
    #   filesrc location=... ! \
    #   rawvideoparse format=i420 width=3840 height=2160 framerate=60/1 ! \
    #   identity eos-after=564 ! queue ! \
    #   videoconvert ! video/x-raw,format=NV12 ! \
    #   nvh264enc bitrate=25000 preset=hq ! \
    #   video/x-h264,profile=high,stream-format=byte-stream,alignment=au ! \
    #   fakesink sync=false qos=false
    p = pathlib.Path(yuv_path).expanduser().resolve()
    args = [
        "gst-launch-1.0", "-e",
        "filesrc", f"location={str(p)}", "!",
        "rawvideoparse",
        "format=i420", f"width={int(width)}", f"height={int(height)}", f"framerate={framerate}", "!",
        "identity", f"eos-after={int(total_frames)}", "!", "queue", "!",
        "video/x-raw,format=I420", "!",
        "nvh265enc", f"bitrate={int(bitrate_kbps)}", f"preset={preset}", "!",
        "video/x-h265,profile=main,stream-format=byte-stream,alignment=au", "!",
        "fakesink", "sync=false", "qos=false"
    ]
    return args

def run_encode_and_measure(argv, verbose=False):
    env = os.environ.copy()
    env["GST_DEBUG"] = env.get("GST_DEBUG", "0" if not verbose else "2")
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.STDOUT

    t0 = time.time()
    proc = subprocess.Popen(argv, shell=False, stdout=stdout, stderr=stderr, env=env)
    try:
        pid, status, rusage = os.wait4(proc.pid, 0)
    except ChildProcessError:
        proc.wait()
        rusage = None
    t1 = time.time()

    wall = max(0.0, t1 - t0)
    cpu_time = 0.0
    if rusage is not None:
        cpu_time = float(getattr(rusage, "ru_utime", 0.0)) + float(getattr(rusage, "ru_stime", 0.0))

    ncpu = os.cpu_count() or 1
    cpu_pct_agg = 0.0 if wall <= 0 else 100.0 * cpu_time / wall         # 可>100%，类似 top 的“%CPU”
    cpu_pct_norm = 0.0 if wall <= 0 else cpu_pct_agg / ncpu             # 归一到 0~100%
    return wall, cpu_pct_agg, cpu_pct_norm

def main():
    ap = argparse.ArgumentParser(description="NVENC(nvh264enc) I420→NV12（CPU转换）全速编码统计")
    ap.add_argument("yuv", help="输入 I420 YUV 文件路径")
    ap.add_argument("--width", type=int, default=3840, help="宽，默认 3840")
    ap.add_argument("--height", type=int, default=2160, help="高，默认 2160")
    ap.add_argument("--total-frames", type=int, default=564, help="总帧数，默认 564")
    ap.add_argument("--framerate", default="60/1", help="解析帧率（默认 60/1）")
    ap.add_argument("--bitrate-kbps", type=int, default=15000, help="码率 Kbps，默认 15000=15Mbps")
    ap.add_argument("--preset", default="hq", help="nvh264enc preset（默认 hq）")
    ap.add_argument("--print-gst-log", action="store_true", help="打印 GStreamer 输出")
    args = ap.parse_args()

    if args.total_frames <= 0:
        print("总帧数必须为正整数")
        return 2

    argv = build_encode_args(
        yuv_path=args.yuv,
        width=args.width,
        height=args.height,
        total_frames=args.total_frames,
        framerate=args.framerate,
        bitrate_kbps=args.bitrate_kbps,
        preset=args.preset
    )

    wall_s, cpu_pct_agg, cpu_pct_norm = run_encode_and_measure(argv, verbose=args.print_gst_log)

    avg_ms = 1000.0 * wall_s / args.total_frames
    fps = args.total_frames / wall_s if wall_s > 0 else float("nan")

    print(f"总帧数: {args.total_frames}")
    print(f"总耗时: {wall_s:.6f} s")
    print(f"平均每帧编码耗时: {avg_ms:.3f} ms")
    print(f"平均编码FPS: {fps:.2f}")
    print(f"平均CPU占用(聚合/类似top): {cpu_pct_agg:.2f}%")
    print(f"平均CPU占用(归一/0-100%): {cpu_pct_norm:.2f}%")

if __name__ == "__main__":
    main()
