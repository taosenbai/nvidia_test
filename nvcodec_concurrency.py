import os
import re
import time
import math
import argparse
import subprocess
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# ----------- 错误/事件匹配 -----------
ERROR_REGEXES = [
    re.compile(r'\btype\s*:\s*error\b', re.I),
    re.compile(r'\bGstMessageError\b', re.I),
    re.compile(r'\berror=\(', re.I),
    re.compile(r'\bnot[-\s]?negotiated\b', re.I),
    re.compile(r'\b(resource|insufficient|unavailable)\b', re.I),
    re.compile(r'\bNo more NvEnc\b', re.I),
    re.compile(r'\bfailed\b', re.I),
    re.compile(r'\bpermission\b', re.I),
    re.compile(r'\bcould not link\b', re.I),
]
EOS_REGEX = re.compile(r'\btype\s*:\s*eos\b', re.I)
ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# ----------- 构建管线 -----------
def build_decode_argv(codec: str, uri: str, frames: int) -> List[str]:
    if codec == "nvh264dec":
        chain = [
            "urisourcebin", f"uri={uri}", "!",
            "h264parse", "!", "nvh264dec", "!",
            "identity", f"eos-after={frames}", "!",
            "fakesink", "sync=false", "qos=false"
        ]
    elif codec == "nvh265dec":
        chain = [
            "urisourcebin", f"uri={uri}", "!",
            "h265parse", "!", "nvh265dec", "!",
            "identity", f"eos-after={frames}", "!",
            "fakesink", "sync=false", "qos=false"
        ]
    else:
        raise ValueError("decode codec 必须为 nvh264dec 或 nvh265dec")
    return ["gst-launch-1.0", "-m", "-e"] + chain

def build_encode_argv(codec: str, frames: int, width: int, height: int,
                      framerate: str, bitrate_kbps: int, use_nv12: bool,
                      file_input: Optional[str]) -> List[str]:
    if file_input:
        p = Path(file_input).expanduser().resolve()
        pre = [
            "filesrc", f"location={str(p)}", "!",
            "rawvideoparse", "format=i420",
            f"width={width}", f"height={height}", f"framerate={framerate}", "!",
            "identity", f"eos-after={frames}", "!"
        ]
        if use_nv12:
            pre += ["videoconvert", "!",
                    f"video/x-raw,format=NV12,width={width},height={height},framerate={framerate}"]
        else:
            pre += [f"video/x-raw,format=I420,width={width},height={height},framerate={framerate}"]
    else:
        # 合成源：先转换再一次性给出完整 caps（注意 caps 必须是一个带逗号的参数）
        pre = [
            "videotestsrc", f"num-buffers={frames}", "is-live=false", "pattern=smpte", "!",
            "videoconvert", "!",
            (f"video/x-raw,format=NV12,width={width},height={height},framerate={framerate}"
             if use_nv12 else
             f"video/x-raw,format=I420,width={width},height={height},framerate={framerate}")
        ]

    if codec == "nvh264enc":
        enc = ["nvh264enc", f"bitrate={bitrate_kbps}", "preset=hq"]
        post_caps = ["video/x-h264,stream-format=byte-stream,alignment=au"]
    elif codec == "nvh265enc":
        enc = ["nvh265enc", f"bitrate={bitrate_kbps}", "preset=hq"]
        post_caps = ["video/x-h265,stream-format=byte-stream,alignment=au"]
    else:
        raise ValueError("encode codec 必须为 nvh264enc 或 nvh265enc")

    chain = pre + ["!"] + enc + ["!"] + post_caps + ["!", "fakesink", "sync=false", "qos=false"]
    return ["gst-launch-1.0", "-m", "-e"] + chain

def make_argv(args, per_pipe_frames: int) -> List[str]:
    if args.codec in ("nvh264dec", "nvh265dec"):
        if not args.input:
            raise SystemExit("解码模式需要 --input 指定输入文件")
        uri = Path(args.input).expanduser().resolve().as_uri()
        return build_decode_argv(args.codec, uri, per_pipe_frames)
    else:
        return build_encode_argv(
            args.codec, per_pipe_frames, args.width, args.height, args.framerate,
            args.bitrate_kbps, use_nv12=not args.i420_direct, file_input=args.input
        )

# ----------- 进程结果 -----------
class PipeResult:
    def __init__(self, idx: int):
        self.idx = idx
        self.ok = True
        self.err_lines: List[str] = []
        self.exit_code: Optional[int] = None
        self.saw_eos: bool = False

def _reader_thread(proc: subprocess.Popen, res: PipeResult, verbose: bool):
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = ANSI_RE.sub('', raw.rstrip('\n'))
        if verbose:
            print(f"[P{res.idx}] {line}")
        if any(r.search(line) for r in ERROR_REGEXES):
            res.ok = False
            if len(res.err_lines) < 6:
                res.err_lines.append(line)
        if EOS_REGEX.search(line):
            res.saw_eos = True

# ----------- GPU 采样 -----------
def _run_cmd(args: List[str]) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(args, shell=False, capture_output=True, text=True)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", ""

class GpuSampler:
    def __init__(self, interval_sec: float = 0.2):
        self.interval = max(0.05, float(interval_sec))
        self.stop_evt = threading.Event()
        self.th: Optional[threading.Thread] = None
        self.fields: Optional[List[str]] = None
        self.samples: Dict[int, Dict[str, List[float]]] = {}  # idx -> {'gpu':[], 'enc':[], 'dec':[], 'mem_used':[], 'mem_total':[]}

    def _try_query_once(self) -> Tuple[Optional[List[str]], str]:
        queries = [
            ["index","utilization.gpu","utilization.encoder","utilization.decoder","memory.used","memory.total"],
            ["index","utilization.gpu","utilization.encoder","memory.used","memory.total"],
            ["index","utilization.gpu","utilization.decoder","memory.used","memory.total"],
            ["index","utilization.gpu","memory.used","memory.total"],
        ]
        for q in queries:
            rc, out, _ = _run_cmd(["nvidia-smi", "--query-gpu="+",".join(q), "--format=csv,noheader,nounits"])
            if rc == 0 and out:
                return q, out
        return None, ""

    def _parse_lines(self, fields: List[str], out: str) -> List[List[str]]:
        rows = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == len(fields):
                rows.append(parts)
        return rows

    def _record_row(self, fields: List[str], parts: List[str]):
        try:
            idx = int(parts[0])
        except:
            return
        rec = self.samples.setdefault(idx, {"gpu":[], "enc":[], "dec":[], "mem_used":[], "mem_total":[]})
        for f, v in zip(fields, parts):
            try:
                if f == "utilization.gpu":
                    rec["gpu"].append(float(v))
                elif f == "utilization.encoder":
                    rec["enc"].append(float(v))
                elif f == "utilization.decoder":
                    rec["dec"].append(float(v))
                elif f == "memory.used":
                    rec["mem_used"].append(float(v))
                elif f == "memory.total":
                    rec["mem_total"].append(float(v))
            except:
                pass

    def _loop(self):
        fields, out = self._try_query_once()
        if not fields:
            return
        self.fields = fields
        for parts in self._parse_lines(fields, out):
            self._record_row(fields, parts)
        while not self.stop_evt.wait(self.interval):
            rc, out, _ = _run_cmd(["nvidia-smi", "--query-gpu="+",".join(self.fields), "--format=csv,noheader,nounits"])
            if rc != 0 or not out:
                continue
            for parts in self._parse_lines(self.fields, out):
                self._record_row(self.fields, parts)

    def start(self) -> bool:
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

def agg(vals: List[float], mode: str) -> Optional[float]:
    if not vals:
        return None
    if mode == "avg":
        return sum(vals) / len(vals)
    if mode == "max":
        return max(vals)
    if mode == "p95":
        s = sorted(vals)
        k = max(0, min(len(s)-1, int(math.ceil(0.95*len(s))-1)))
        return s[k]
    return sum(vals) / len(vals)

def pick_busy_gpu(samples: Dict[int, Dict[str, List[float]]], preferred_field: str, mode: str, fixed_idx: Optional[int]) -> Tuple[Optional[int], Optional[float], str]:
    field = preferred_field
    # 如果首选字段不存在，回退到 'gpu'
    has_field = any(samples.get(i, {}).get(field) for i in samples.keys())
    if not has_field:
        field = "gpu"
        if not any(samples.get(i, {}).get(field) for i in samples.keys()):
            return None, None, field

    if fixed_idx is not None and fixed_idx in samples:
        val = agg(samples[fixed_idx].get(field, []), mode)
        return fixed_idx, val, field

    best_idx, best_val = None, None
    for i, rec in samples.items():
        val = agg(rec.get(field, []), mode)
        if val is None:
            continue
        if best_val is None or val > best_val:
            best_idx, best_val = i, val
    return best_idx, best_val, field

# ----------- 执行批次 -----------
def run_batch(args, per_pipe_frames: int) -> Tuple[List[PipeResult], bool, Dict[str, Any]]:
    argv = make_argv(args, per_pipe_frames)
    env = os.environ.copy()
    env["GST_DEBUG_NO_COLOR"] = "1"
    env.setdefault("GST_DEBUG", "0")
    if args.gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    # GPU 采样
    sampler = GpuSampler(interval_sec=args.gpu_sample_interval)
    gpu_sampling = sampler.start()

    procs: List[subprocess.Popen] = []
    results: List[PipeResult] = []
    readers: List[threading.Thread] = []

    # 启动并发
    for i in range(args.concurrency):
        res = PipeResult(i)
        results.append(res)
        p = subprocess.Popen(
            argv, shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env
        )
        procs.append(p)
        t = threading.Thread(target=_reader_thread, args=(p, res, args.print_gst_log), daemon=True)
        readers.append(t)
        t.start()

    # 等待完成/超时
    deadline = time.time() + args.timeout if args.timeout > 0 else None
    for i, p in enumerate(procs):
        try:
            if deadline is None:
                ret = p.wait()
            else:
                remaining = max(0.0, deadline - time.time())
                ret = p.wait(timeout=remaining)
            results[i].exit_code = ret
        except subprocess.TimeoutExpired:
            results[i].ok = False
            results[i].err_lines.append("TimeoutExpired")
            try:
                p.kill()
            except Exception:
                pass
            results[i].exit_code = None

    for t in readers:
        t.join(timeout=1.0)

    # 停止采样并评估饱和
    gpu_info = {
        "sampling": gpu_sampling,
        "busy_idx": None,
        "metric": None,
        "value": None,
        "threshold": args.gpu_threshold,
        "saturated": False
    }
    if gpu_sampling:
        sampler.stop()
        # 选择 dec/enc 字段
        preferred_field = "dec" if args.codec.endswith("dec") else "enc"
        busy_idx, val, metric = pick_busy_gpu(sampler.samples, preferred_field, args.gpu_agg, args.gpu_index)
        gpu_info.update({"busy_idx": busy_idx, "metric": metric, "value": val})
        if val is not None and val >= args.gpu_threshold:
            gpu_info["saturated"] = True

    # 判定成功：无错误、退出码为0、（可选）收到EOS、且GPU未饱和（若启用阈值）
    ok_all = True
    for r in results:
        cond = r.ok and (r.exit_code == 0)
        if args.require_eos:
            cond = cond and r.saw_eos
        if not cond:
            ok_all = False
    if args.use_gpu_threshold and gpu_info["saturated"]:
        ok_all = False

    return results, ok_all, gpu_info

# ----------- 主程序 -----------
def main():
    ap = argparse.ArgumentParser(description="NV NVIDIA NVCodec 并发探测器（错误+GPU饱和阈值）")
    ap.add_argument("--codec", required=True,
                    choices=["nvh264dec","nvh265dec","nvh264enc","nvh265enc"])
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--frames-per-pipe", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--gpu-index", type=int, default=None)

    ap.add_argument("--input", help="解码必需；编码可选（I420 原始YUV）")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--framerate", default="30/1")
    ap.add_argument("--bitrate-kbps", type=int, default=5000)
    ap.add_argument("--i420-direct", action="store_true", help="编码直接 I420（enc 不支持则会失败）")

    ap.add_argument("--require-eos", action="store_true", help="收到 EOS 才算成功")
    ap.add_argument("--print-gst-log", action="store_true")

    # GPU 阈值与聚合方式
    ap.add_argument("--use-gpu-threshold", action="store_true",
                    help="启用 GPU 饱和阈值判定（达到阈值视为上限）")
    ap.add_argument("--gpu-threshold", type=float, default=95.0,
                    help="GPU 编解码利用率阈值（%），默认 95")
    ap.add_argument("--gpu-agg", choices=["avg","max","p95"], default="p95",
                    help="GPU 利用率聚合口径，默认 p95")
    ap.add_argument("--gpu-sample-interval", type=float, default=0.2,
                    help="GPU 采样间隔秒，默认 0.2")

    ap.add_argument("--search", action="store_true", help="自动搜索最大并发（错误或GPU饱和即停止增长）")
    ap.add_argument("--search-max", type=int, default=128)
    args = ap.parse_args()

    if args.search:
        low, high = 0, 1
        while high <= args.search_max:
            args.concurrency = high
            print(f"尝试并发: {high}")
            _, ok, gpu = run_batch(args, args.frames_per_pipe)
            if ok:
                low = high
                high *= 2
            else:
                break
        if high > args.search_max and low == args.search_max:
            print(f"在上界 {args.search_max} 内均成功。最大并发 >= {args.search_max}")
            return
        if low == 0 and not ok:
            print("并发 1 即失败，请检查插件/驱动/管线。")
            return

        lo, hi = low, min(high, args.search_max)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            args.concurrency = mid
            print(f"二分尝试并发: {mid}")
            _, ok, gpu = run_batch(args, args.frames_per_pipe)
            if ok:
                lo = mid
            else:
                hi = mid
        print(f"最大稳定并发: {lo}（{args.codec}）")
        return

    results, ok, gpu = run_batch(args, args.frames_per_pipe)
    succ = sum(1 for r in results if r.ok and r.exit_code == 0 and (r.saw_eos or not args.require_eos))
    fail = len(results) - succ
    print(f"结果: 成功 {succ} / 失败 {fail}，并发={args.concurrency}，类型={args.codec}")
    if gpu["sampling"]:
        if gpu["value"] is not None:
            print(f"GPU饱和判定: 指标={gpu['metric']} 聚合={args.gpu_agg} 值={gpu['value']:.1f}% 阈值={gpu['threshold']:.1f}%  -> {'饱和' if gpu['saturated'] else '未饱和'}")
        else:
            print("GPU饱和判定: 无法获取相关指标（回退/字段缺失）")
    if fail or (args.use_gpu_threshold and gpu["saturated"]):
        for r in results:
            if not (r.ok and r.exit_code == 0 and (r.saw_eos or not args.require_eos)):
                print(f"- 管线#{r.idx} 失败，exit={r.exit_code}, EOS={r.saw_eos}, 示例错误: {r.err_lines[:3]}")

if __name__ == "__main__":
    main()
