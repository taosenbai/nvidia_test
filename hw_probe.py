import os
import re
import subprocess
from xml.etree import ElementTree as ET

def run(cmd):
    try:
        p = subprocess.run(cmd, shell=False, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", ""

def parse_cpu_info():
    # 先尝试 lscpu，失败则回退 /proc/cpuinfo
    rc, out, _ = run(["lscpu"])
    model, threads = None, None
    if rc == 0 and out:
        for line in out.splitlines():
            if line.startswith("Model name:"):
                model = line.split(":", 1)[1].strip()
            elif line.startswith("CPU(s):"):
                # 第一处 CPU(s) 即总逻辑核数
                try:
                    threads = int(line.split(":", 1)[1].strip())
                except:
                    pass
        if model and threads:
            return model, threads

    # 回退 /proc/cpuinfo
    model = None
    threads = 0
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if not model and line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                if line.lower().startswith("processor"):
                    threads += 1
    except:
        pass
    return model or "Unknown CPU", threads or os.cpu_count() or 0

def parse_mem_total_gib():
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 2)
    except:
        pass
    return None

def parse_gpus_from_xml(xml_text):
    gpus = []
    try:
        root = ET.fromstring(xml_text)
        # 适配 <nvidia_smi_log><gpu>...
        for idx, gpu in enumerate(root.findall(".//gpu")):
            name = gpu.findtext("product_name", default="Unknown")
            mem_total = gpu.findtext("fb_memory_usage/total", default="0 MiB")
            # 提取数字 MiB
            m = re.search(r"(\d+)\s*MiB", mem_total)
            mem_mib = int(m.group(1)) if m else None
            gpus.append({
                "index": idx,
                "name": name,
                "memory_total_mib": mem_mib
            })
    except ET.ParseError:
        pass
    return gpus

def parse_gpus_basic():
    # 优先用 nvidia-smi -q -x（可同时用于后续 limit 提取）
    rc, xml_out, _ = run(["nvidia-smi", "-q", "-x"])
    gpus = []
    if rc == 0 and xml_out.strip():
        gpus = parse_gpus_from_xml(xml_out)
        return gpus, xml_out

    # 回退：nvidia-smi --query
    rc, out, _ = run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
    if rc == 0 and out.strip():
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mib": int(parts[2])
                    })
                except:
                    pass
        return gpus, None

    # 回退：nvidia-smi -L
    rc, out, _ = run(["nvidia-smi", "-L"])
    if rc == 0 and out.strip():
        for idx, line in enumerate(out.strip().splitlines()):
            # 示例: "GPU 0: Tesla T4 (UUID: ...)"
            m = re.match(r"GPU\s+(\d+):\s+(.+?)\s+\(", line)
            if m:
                try:
                    gpus.append({
                        "index": int(m.group(1)),
                        "name": m.group(2).strip(),
                        "memory_total_mib": None
                    })
                except:
                    pass
    return gpus, None

def extract_session_limits(xml_text, text_q=None):
    # 返回每块 GPU 的 (enc_limit, dec_limit)，若未知则为 None
    # 1) 先尝试 XML（大多数版本并不提供最大并发）
    limits = []
    if xml_text:
        try:
            root = ET.fromstring(xml_text)
            for gpu in root.findall(".//gpu"):
                enc_lim = None
                dec_lim = None
                # 常见并无上限字段；保留占位
                limits.append((enc_lim, dec_lim))
        except ET.ParseError:
            pass

    # 2) 尝试从文本 nvidia-smi -q 中抓取可能出现的“Limit/Max Sessions”描述
    if text_q is None:
        rc, txt, _ = run(["nvidia-smi", "-q"])
        text_q = txt if rc == 0 else ""

    # 可能的正则（宽松尝试）
    enc_pat = re.compile(r"(?:Encoder|NVENC)[^\n\r]*?(?:Session|Sessions)[^\n\r]*?(?:Limit|Max)[^\d]*(\d+)", re.IGNORECASE)
    dec_pat = re.compile(r"(?:Decoder|NVDEC)[^\n\r]*?(?:Session|Sessions)[^\n\r]*?(?:Limit|Max)[^\d]*(\d+)", re.IGNORECASE)

    enc_hits = enc_pat.findall(text_q or "")
    dec_hits = dec_pat.findall(text_q or "")

    # 简化：如果解析到全局单值，应用到所有 GPU
    enc_global = int(enc_hits[0]) if enc_hits else None
    dec_global = int(dec_hits[0]) if dec_hits else None

    if not limits:
        # 不知道 GPU 数量时，空列表
        return []

    res = []
    for i in range(len(limits)):
        e, d = limits[i]
        res.append((e if e is not None else enc_global,
                    d if d is not None else dec_global))
    return res

def main():
    # CPU
    cpu_model, cpu_threads = parse_cpu_info()
    mem_gib = parse_mem_total_gib()

    # GPU
    gpus, xml_out = parse_gpus_basic()
    if not gpus:
        print("未检测到 NVIDIA GPU（或未安装 nvidia-smi）。")
    else:
        print(f"GPU 数量: {len(gpus)}")
        for g in gpus:
            mm = f"{g['memory_total_mib']} MiB" if g['memory_total_mib'] is not None else "未知"
            print(f"- GPU{g['index']}: {g['name']} / 显存: {mm}")

        # 并发上限（若驱动输出未提供，将为 N/A）
        limits = extract_session_limits(xml_out)
        if limits and len(limits) == len(gpus):
            for i, (enc_lim, dec_lim) in enumerate(limits):
                enc_s = str(enc_lim) if enc_lim is not None else "N/A"
                dec_s = str(dec_lim) if dec_lim is not None else "N/A"
                print(f"- GPU{gpus[i]['index']} 并发上限: NVENC={enc_s}, NVDEC={dec_s}")
        else:
            # 无法从驱动输出获得
            print("- 驱动未公开 NVENC/NVDEC 并发上限（nvidia-smi -q/-x 未提供），结果：N/A")

    # CPU / 内存
    print(f"CPU 型号: {cpu_model}")
    print(f"逻辑核数: {cpu_threads}")
    if mem_gib is not None:
        print(f"物理内存: {mem_gib} GiB")
    else:
        print("物理内存: 未知")

if __name__ == "__main__":
    main()
