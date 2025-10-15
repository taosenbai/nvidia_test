import os
import time
import argparse
import subprocess
from pathlib import Path

TOTAL_FRAMES = 564  # 写死总帧数

def build_decode_argv(h265_es_path: str):
    # 使用 urisourcebin 以安全处理文件路径（含空格等）
    uri = Path(h265_es_path).expanduser().resolve().as_uri()
    return [
        "gst-launch-1.0", "-e",
        "urisourcebin", f"uri={uri}", "!",
        "h265parse", "!",
        "nvh265dec", "!",
        "fakesink", "sync=false", "qos=false"
    ]

def run_and_measure(argv, verbose=False):
    env = os.environ.copy()
    env["GST_DEBUG"] = env.get("GST_DEBUG", "0" if not verbose else "2")
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.STDOUT

    t0 = time.time()
    proc = subprocess.Popen(argv, shell=False, stdout=stdout, stderr=stderr, env=env)
    try:
        pid, status, rusage = os.wait4(proc.pid, 0)  # 仅统计 gst-launch 进程自身
    except ChildProcessError:
        proc.wait()
        rusage = None
    t1 = time.time()

    wall = max(0.0, t1 - t0)
    cpu_time = 0.0
    if rusage is not None:
        cpu_time = float(getattr(rusage, "ru_utime", 0.0)) + float(getattr(rusage, "ru_stime", 0.0))

    ncpu = os.cpu_count() or 1
    cpu_pct_agg = 0.0 if wall <= 0 else 100.0 * cpu_time / wall     # 聚合口径，可能>100%
    cpu_pct_norm = 0.0 if wall <= 0 else cpu_pct_agg / ncpu         # 归一到0~100%
    return wall, cpu_pct_agg, cpu_pct_norm

def main():
    ap = argparse.ArgumentParser(description="NVDEC(nvh265dec) .h265 ES 全速解码统计（平均耗时 & 进程CPU）")
    ap.add_argument("input", help=".h265 ES 文件路径")
    ap.add_argument("--print-gst-log", action="store_true", help="打印 GStreamer 输出（排查用）")
    args = ap.parse_args()

    argv = build_decode_argv(args.input)
    wall_s, cpu_pct_agg, cpu_pct_norm = run_and_measure(argv, verbose=args.print_gst_log)

    avg_ms = 1000.0 * wall_s / TOTAL_FRAMES
    fps = TOTAL_FRAMES / wall_s if wall_s > 0 else float("nan")

    print(f"总帧数: {TOTAL_FRAMES}")
    print(f"总耗时: {wall_s:.6f} s")
    print(f"平均每帧解码耗时: {avg_ms:.3f} ms")
    print(f"平均解码FPS: {fps:.2f}")
    print(f"平均CPU占用(聚合/类似top): {cpu_pct_agg:.2f}%")
    print(f"平均CPU占用(归一/0-100%): {cpu_pct_norm:.2f}%")

if __name__ == "__main__":
    main()
