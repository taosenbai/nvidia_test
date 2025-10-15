import os
import time
import argparse
import subprocess
import pathlib

def build_decode_args(input_path, stream_index=0, codec="h264"):
    # 使用 URI 以兼容路径中包含空格
    uri = pathlib.Path(input_path).expanduser().resolve().as_uri()
    parser = "h264parse" if codec == "h264" else "h265parse"
    decoder = "nvh264dec" if codec == "h264" else "nvh265dec"
    # 以 argv 列表启动（shell=False），确保统计到 gst-launch 进程本身
    return [
        "gst-launch-1.0", "-e",
        "urisourcebin", f"uri={uri}", "!",
        "qtdemux", "name=d", f"d.video_{stream_index}", "!", "queue", "!",
        parser, "!", decoder, "!", "queue", "!",
        "fakesink", "sync=false", "qos=false",
    ]

def run_decode_and_measure(argv):
    env = os.environ.copy()
    env["GST_DEBUG"] = env.get("GST_DEBUG", "0")  # 关闭多余日志，减少干扰
    t0 = time.time()
    proc = subprocess.Popen(
        argv,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=env
    )
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

    # 两种口径：
    # - 聚合口径（类似 top，可能 >100%）：cpu_time / wall
    # - 归一口径（0~100%，按逻辑核归一）：聚合 / ncpu
    ncpu = os.cpu_count() or 1
    cpu_pct_agg = 0.0 if wall <= 0 else 100.0 * cpu_time / wall
    cpu_pct_norm = 0.0 if wall <= 0 else cpu_pct_agg / ncpu
    return wall, cpu_pct_agg, cpu_pct_norm

def main():
    ap = argparse.ArgumentParser(description="NVDEC 全速解码基准（固定总帧数，准确统计 gst-launch 进程 CPU）")
    ap.add_argument("input", help="输入 MP4 文件路径")
    ap.add_argument("--total-frames", type=int, default=564, help="总帧数（默认 564）")
    ap.add_argument("--codec", choices=["h264","h265"], default="h264", help="h264 或 h265，默认 h264")
    ap.add_argument("--stream-index", type=int, default=0, help="qtdemux 视频流索引，默认 0")
    args = ap.parse_args()

    if args.total_frames <= 0:
        print("总帧数必须为正整数")
        return 2

    argv = build_decode_args(args.input, args.stream_index, args.codec)
    wall_s, cpu_pct_agg, cpu_pct_norm = run_decode_and_measure(argv)

    avg_ms = 1000.0 * wall_s / args.total_frames
    fps = args.total_frames / wall_s if wall_s > 0 else float("nan")

    print(f"总帧数: {args.total_frames}")
    print(f"总耗时: {wall_s:.6f} s")
    print(f"平均每帧耗时: {avg_ms:.3f} ms")
    print(f"平均FPS: {fps:.2f}")
    print(f"平均CPU占用(聚合/类似top): {cpu_pct_agg:.2f}%")
    print(f"平均CPU占用(归一/0-100%): {cpu_pct_norm:.2f}%")

if __name__ == "__main__":
    main()
