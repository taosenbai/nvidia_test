import os
import re
import time
import argparse
import subprocess
import threading
from pathlib import Path

TOTAL_FRAMES = 564  # 写死总帧数

def build_decode_argv(h264_es_path: str):
    # 使用 urisourcebin 以安全处理文件路径（含空格等）
    uri = Path(h264_es_path).expanduser().resolve().as_uri()
    return [
        "gst-launch-1.0", "-e",
        "urisourcebin", f"uri={uri}", "!",
        "h264parse", "!",
        "nvh264dec", "!",
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

# ---------------- GPU 统计 ----------------

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
        self.have_encdec = False

    def _try_query_once(self):
        # 优先带 encoder/decoder；失败则退化为仅 GPU/MEM
        queries = [
            ["index","utilization.gpu","utilization.encoder","utilization.decoder","memory.used","memory.total"],
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
        rec = self.samples.setdefault(idx, {"gpu":[], "enc":[], "dec":[], "mem_used":[], "mem_total":[]})
        # 映射字段
        for f, v in zip(fields, parts):
            if f == "utilization.gpu":
                try: rec["gpu"].append(float(v))
                except: pass
            elif f == "utilization.encoder":
                try: rec["enc"].append(float(v))
                except: pass
            elif f == "utilization.decoder":
                try: rec["dec"].append(float(v))
                except: pass
            elif f == "memory.used":
                try: rec["mem_used"].append(float(v))
                except: pass
            elif f == "memory.total":
                try: rec["mem_total"].append(float(v))
                except: pass

    def _loop(self):
        # 第一次检测字段
        fields, out = self._try_query_once()
        if not fields:
            return
        self.fields = fields
        self.have_encdec = ("utilization.encoder" in fields) and ("utilization.decoder" in fields)

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
        # 先检查 nvidia-smi 是否存在
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
        # 返回每 GPU 的平均值
        def avg(lst):
            return sum(lst)/len(lst) if lst else None
        res = {}
        for idx, rec in self.samples.items():
            res[idx] = {
                "gpu": avg(rec["gpu"]),
                "enc": avg(rec["enc"]),
                "dec": avg(rec["dec"]),
                "mem_used_mib": avg(rec["mem_used"]),
                "mem_total_mib": (rec["mem_total"][0] if rec["mem_total"] else None)
            }
        return res, self.have_encdec

def main():
    ap = argparse.ArgumentParser(description="NVDEC(nvh264dec) .h264 ES 全速解码统计（平均耗时 & 进程CPU & GPU利用率）")
    ap.add_argument("input", help=".h264 ES 文件路径")
    ap.add_argument("--print-gst-log", action="store_true", help="打印 GStreamer 输出（排查用）")
    ap.add_argument("--gpu-sample-interval", type=float, default=0.2, help="GPU 采样间隔秒，默认 0.2")
    args = ap.parse_args()

    # 启动 GPU 采样
    sampler = GpuSampler(interval_sec=args.gpu_sample_interval)
    gpu_ok = sampler.start()

    # 跑解码并计时/CPU
    argv = build_decode_argv(args.input)
    wall_s, cpu_pct_agg, cpu_pct_norm = run_and_measure(argv, verbose=args.print_gst_log)

    # 停止 GPU 采样并汇总
    if gpu_ok:
        sampler.stop()
        gpu_stats, have_encdec = sampler.stats()
        gpu_names = get_gpu_names()
    else:
        gpu_stats, have_encdec, gpu_names = {}, False, {}

    avg_ms = 1000.0 * wall_s / TOTAL_FRAMES
    fps = TOTAL_FRAMES / wall_s if wall_s > 0 else float("nan")

    print(f"总帧数: {TOTAL_FRAMES}")
    print(f"总耗时: {wall_s:.6f} s")
    print(f"平均每帧解码耗时: {avg_ms:.3f} ms")
    print(f"平均解码FPS: {fps:.2f}")
    print(f"平均CPU占用(聚合/类似top): {cpu_pct_agg:.2f}%")
    print(f"平均CPU占用(归一/0-100%): {cpu_pct_norm:.2f}%")

    if not gpu_ok:
        print("GPU统计: 未检测到 nvidia-smi，无法采样")
    else:
        print("GPU统计(平均值):")
        # 选出“最可能活跃”的 GPU（平均 decoder 利用率最高）
        primary_idx = None
        best_dec = -1.0
        for idx, s in gpu_stats.items():
            if s.get("dec") is not None and s["dec"] > best_dec:
                best_dec = s["dec"]
                primary_idx = idx
        for idx in sorted(gpu_stats.keys()):
            s = gpu_stats[idx]
            name = gpu_names.get(idx, f"GPU{idx}")
            gpu = s.get("gpu")
            enc = s.get("enc")
            dec = s.get("dec")
            mu = s.get("mem_used_mib")
            mt = s.get("mem_total_mib")
            tag = " <=" if idx == primary_idx else ""
            if have_encdec:
                print(f"- GPU{idx}({name}): GPU={gpu:.1f}%  ENC={enc if enc is not None else 'N/A'}%  "
                      f"DEC={dec if dec is not None else 'N/A'}%  MEM={mu:.0f}/{mt:.0f} MiB{tag}")
            else:
                print(f"- GPU{idx}({name}): GPU={gpu if gpu is not None else 'N/A'}%  "
                      f"MEM={mu:.0f}/{mt:.0f} MiB{tag}")

if __name__ == "__main__":
    main()
