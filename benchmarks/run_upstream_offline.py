"""Phase 0: run upstream `bench.py` unchanged N times and record throughput + environment.

Usage (from repo root, inside venv A):
    python benchmarks/run_upstream_offline.py --runs 3 --out benchmarks/results/00_upstream_offline.json

Each run is a separate process (one GPU process at a time). Nothing here alters bench.py.
"""
import argparse
import datetime
import json
import os
import platform
import re
import statistics
import subprocess
import sys

UPSTREAM_COMMIT = "bb823b3"
PATTERN = re.compile(r"Total: (\d+)tok, Time: ([\d.]+)s, Throughput: ([\d.]+)tok/s")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable ({e.__class__.__name__})"


def environment():
    import flash_attn
    import torch
    import transformers
    import triton
    gpu = sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader")
    return {
        "gpu": gpu,
        "nvidia_smi_cuda": sh("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
        "cpu": sh("lscpu | grep 'Model name' | sed 's/Model name:[ ]*//'"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "flash_attn": flash_attn.__version__,
        "transformers": transformers.__version__,
        "upstream_commit": UPSTREAM_COMMIT,
        "repo_commit": sh("git rev-parse --short HEAD"),
        "bench_py_unmodified_vs_upstream": sh(f"git diff --quiet {UPSTREAM_COMMIT} -- bench.py nanovllm && echo true || echo false"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="benchmarks/results/00_upstream_offline.json")
    args = ap.parse_args()

    runs = []
    for i in range(args.runs):
        proc = subprocess.run([sys.executable, "bench.py"], capture_output=True, text=True)
        m = PATTERN.search(proc.stdout)
        if proc.returncode != 0 or not m:
            sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-4000:])
            raise SystemExit(f"run {i} failed (rc={proc.returncode})")
        total, secs, tput = int(m.group(1)), float(m.group(2)), float(m.group(3))
        runs.append({"run": i, "total_tokens": total, "time_s": secs, "throughput_tok_s": tput})
        print(f"run {i}: {total} tok, {secs:.2f} s, {tput:.2f} tok/s", flush=True)

    tputs = [r["throughput_tok_s"] for r in runs]
    result = {
        "description": "upstream bench.py (256 seqs, input/output 100-1024 random tokens, Qwen3-0.6B), run unchanged",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "environment": environment(),
        "runs": runs,
        "summary": {
            "throughput_mean_tok_s": round(statistics.mean(tputs), 2),
            "throughput_stdev_tok_s": round(statistics.stdev(tputs), 2) if len(tputs) > 1 else None,
            "throughput_min_tok_s": min(tputs),
            "throughput_max_tok_s": max(tputs),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
