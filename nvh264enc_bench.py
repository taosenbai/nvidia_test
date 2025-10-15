import os
import time
import argparse
import subprocess
import pathlib
import threading

def build_encode_args(yuv_path, width=3840, height=2160, total_frames=564,
                      framerate="60/1", bitrate_kbps=25000, preset="hq"):
    # gst-launch-1.0 -e \
    #   filesrc location=... ! \
    #   rawvideoparse format=i420 width=3840 height=2160 framerate=60/1 ! \
    #   identity eos-after=564 ! queue ! \
    #   video/x-raw,format=I420 ! \
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
        "nvh264enc", f"bitrate={int(bitrate_kbps)}", f"preset={preset}", "!",
        "video/x-h264,profile=high,stream-format=byte-stream,alignment=au", "!",
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

# ---------------- GPU 采样 ----------------

def _run_cmd(args):
    try:
        p = subprocess.run(args, shell=False, capture_output=True, text=True)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", ""

def get_gpu_names():
    rc, out, _ = _run_cmd([
        "nvidia-smi", "--query-gpu=index,name",
        "--format=csv,noheader"
    ])
    names = {}
    if rc == 0 and out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                except:
                    continue
                names[idx] = parts[1]
    return names

class GpuSampler:
    def __init__(self, interval_sec=0.2):
        self.interval = max(0.05, float(interval_sec))
        self.stop_evt = threading.Event()
        self.th = None
        self.samples = {}  # idx -> dict of lists
        self.fields = None
        self.have_enc = False

    def _try_query_once(self):
        # 优先带 encoder；若不支持则退化为仅 GPU/MEM
        queries = [
            ["index","utilization.gpu","utilization.encoder","memory.used","memory.total"],
            ["index","utilization.gpu","memory.used","memory.total"]
        ]
        for q in queries:
            rc, out, _ = _run_cmd(["nvidia-smi", "--query-gpu="+",".join(q), "--format=csv,noheader,nounits"])
            if rc == 0 and out:
                return q, out
        return None, ""

    def _parse_lines(self, fields, out):
        rows = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(fields):
                continue
            rows.append(parts)
        return rows

    def _record_row(self, fields, parts):
        try:
            idx = int(parts[0])
        except:
            return
        rec = self.samples.setdefault(idx, {"gpu":[], "enc":[], "mem_used":[], "mem_total":[]})
        for f, v in zip(fields, parts):
            if f == "utilization.gpu":
                try: rec["gpu"].append(float(v))
                except: pass
            elif f == "utilization.encoder":
                try: rec["enc"].append(float(v))
                except: pass
            elif f == "memory.used":
                try: rec["mem_used"].append(float(v))
                except: pass
            elif f == "memory.total":
                try: rec["mem_total"].append(float(v))
                except: pass

    def _loop(self):
        fields, out = self._try_query_once()
        if not fields:
            return
        self.fields = fields
        self.have_enc = ("utilization.encoder" in fields)

        rows = self._parse_lines(fields, out)
        for parts in rows:
            self._record_row(fields, parts)

        while not self.stop_evt.wait(self.interval):
            rc, out, _ = _run_cmd(["nvidia-smi", "--query-gpu="+",".join(fields), "--format=csv,noheader,nounits"])
            if rc != 0 or not out:
                continue
            rows = self._parse_lines(fields, out)
            for parts in rows:
                self._record_row(fields, parts)

    def start(self):
        rc, _, _ = _run_cmd(["nvidia-smi", "-L"])
        if rc != 0:
            return False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return True

    def stop(self):
        self.stop_evt.set()
        if self.th:
            self.th.join(timeout=2.0)

    def stats(self):
        def avg(lst):
            return sum(lst)/len(lst) if lst else None
        res = {}
        for idx, rec in self.samples.items():
            res[idx] = {
                "gpu": avg(rec["gpu"]),
                "enc": avg(rec["enc"]),
                "mem_used_mib": avg(rec["mem_used"]),
                "mem_total_mib": (rec["mem_total"][0] if rec["mem_total"] else None)
            }
        return res, self.have_enc

def main():
    ap = argparse.ArgumentParser(description="NVENC(nvh264enc) I420 全速编码统计（进程CPU & GPU利用率）")
    ap.add_argument("yuv", help="输入 I420 YUV 文件路径")
    ap.add_argument("--width", type=int, default=3840, help="宽，默认 3840")
    ap.add_argument("--height", type=int, default=2160, help="高，默认 2160")
    ap.add_argument("--total-frames", type=int, default=564, help="总帧数，默认 564")
    ap.add_argument("--framerate", default="60/1", help="解析帧率（默认 60/1）")
    ap.add_argument("--bitrate-kbps", type=int, default=25000, help="码率 Kbps，默认 25000=25Mbps")
    ap.add_argument("--preset", default="hq", help="nvh264enc preset（默认 hq）")
    ap.add_argument("--gpu-sample-interval", type=float, default=0.2, help="GPU 采样间隔秒，默认 0.2")
    ap.add_argument("--print-gst-log", action="store_true", help="打印 GStreamer 输出")
    args = ap.parse_args()

    if args.total_frames <= 0:
        print("总帧数必须为正整数")
        return 2

    # 启动 GPU 采样
    sampler = GpuSampler(interval_sec=args.gpu_sample_interval)
    gpu_ok = sampler.start()
    gpu_names = get_gpu_names() if gpu_ok else {}

    # 跑编码并计时/CPU
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

    # 停采样并汇总
    if gpu_ok:
        sampler.stop()
        gpu_stats, have_enc = sampler.stats()
    else:
        gpu_stats, have_enc = {}, False

    avg_ms = 1000.0 * wall_s / args.total_frames
    fps = args.total_frames / wall_s if wall_s > 0 else float("nan")

    print(f"总帧数: {args.total_frames}")
    print(f"总耗时: {wall_s:.6f} s")
    print(f"平均每帧编码耗时: {avg_ms:.3f} ms")
    print(f"平均编码FPS: {fps:.2f}")
    print(f"平均CPU占用(聚合/类似top): {cpu_pct_agg:.2f}%")
    print(f"平均CPU占用(归一/0-100%): {cpu_pct_norm:.2f}%")

    if not gpu_ok:
        print("GPU统计: 未检测到 nvidia-smi，无法采样")
    else:
        print("GPU统计(平均值):")
        for idx in sorted(gpu_stats.keys()):
            s = gpu_stats[idx]
            name = gpu_names.get(idx, f"GPU{idx}")
            gpu = s.get("gpu")
            enc = s.get("enc")
            mu = s.get("mem_used_mib")
            mt = s.get("mem_total_mib")
            if have_enc:
                print(f"- GPU{idx}({name}): GPU={gpu if gpu is not None else 'N/A'}%  "
                      f"ENC={enc if enc is not None else 'N/A'}%  MEM={mu:.0f}/{mt:.0f} MiB")
            else:
                print(f"- GPU{idx}({name}): GPU={gpu if gpu is not None else 'N/A'}%  "
                      f"MEM={mu:.0f}/{mt:.0f} MiB")

if __name__ == "__main__":
    main()
