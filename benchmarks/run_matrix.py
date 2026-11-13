"""Phase 4 benchmark matrix: systems x workloads x request rates, driven by `vllm bench serve`.

    python benchmarks/run_matrix.py --plan                # print commands + runtime estimate, no GPU
    python benchmarks/run_matrix.py --find-saturation     # rate=inf per system/workload -> saturation.json
    python benchmarks/run_matrix.py --run                 # the matrix (resumable: existing points are skipped)
    python benchmarks/run_matrix.py --knob                # max_prefill_tokens_per_step sweep (system B)

Config: benchmarks/matrix.json. Raw output: benchmarks/results/raw/phase4/...
One GPU process at a time: each system's server is started, used for all of its points, and
stopped (GPU checked idle) before the next one starts. Every (system, workload) gets a
discarded warmup run first so torch.compile / CUDA-graph / mixed-path warmup is not measured.
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_utils import (VLLM_BIN, scrape_metrics, start_server, stop_server,  # noqa: E402
                          vllm_bench_serve)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "benchmarks", "results", "raw", "phase4")
NANO_PY = os.environ.get("NANO_PY", sys.executable)
NANO_BLOCK, VLLM_BLOCK = 256, 16


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def server_cmd(cfg, system: dict, extra: list[str] | None = None) -> list[str]:
    kv_tokens = cfg["kv_cache_tokens"]
    common = ["--port", str(cfg["port"]), "--served-model-name", cfg["served_model_name"],
              "--max-model-len", str(cfg["max_model_len"])]
    if system["kind"] == "nano":
        return [NANO_PY, "-m", "nanovllm.serve", "--model", cfg["model"], *common,
                "--max-kvcache-blocks", str(kv_tokens // NANO_BLOCK), *system["args"], *(extra or [])]
    return [VLLM_BIN, "serve", cfg["model"], *common, "--dtype", "bfloat16", "--seed", str(cfg["seed"]),
            "--num-gpu-blocks-override", str(kv_tokens // VLLM_BLOCK), *system["args"], *(extra or [])]


def bench_args(cfg, workload: dict, rate, num_prompts: int) -> list[str]:
    return [*workload["bench_args"], "--num-prompts", str(num_prompts), "--request-rate", str(rate),
            "--ignore-eos", "--seed", str(cfg["seed"]),
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", cfg["percentiles"]]


class MetricsPoller:
    """Samples the server's /metrics once a second during a run: KV-cache usage and queue gauges."""

    KEYS = ("nanovllm_kv_cache_blocks_used", "nanovllm_kv_cache_blocks_total", "nanovllm_requests_running",
            "nanovllm_requests_waiting", "vllm:kv_cache_usage_perc", "vllm:num_requests_running",
            "vllm:num_requests_waiting")

    def __init__(self, port):
        self.port, self.samples, self._stop = port, [], threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                m = scrape_metrics(self.port)
                self.samples.append({k: v for k, v in m.items() if k.split("{")[0] in self.KEYS})
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)

    def summary(self):
        out = {}
        for k in {k for s in self.samples for k in s}:
            vals = [s[k] for s in self.samples if k in s]
            out[k] = {"max": max(vals), "mean": sum(vals) / len(vals), "n": len(vals)}
        return out


def run_point(cfg, port, workload, rate, num_prompts, out_dir, name):
    path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(path):
        print(f"  skip (exists) {path}", flush=True)
        return json.load(open(path))
    os.makedirs(out_dir, exist_ok=True)
    before = scrape_metrics(port)
    with MetricsPoller(port) as poller:
        r = vllm_bench_serve(port, cfg["served_model_name"], cfg["model"], out_dir, f"{name}.json",
                             bench_args(cfg, workload, rate, num_prompts), os.path.join(out_dir, f"{name}.log"))
    after = scrape_metrics(port)
    counters = {k: after[k] - before.get(k, 0.0) for k in after
                if k.endswith(("_total", "_sum", "_count")) and after[k] != before.get(k, 0.0)}
    json.dump({"counters_delta": counters, "gauges": poller.summary()},
              open(os.path.join(out_dir, f"{name}.server_metrics.json"), "w"), indent=2)
    print(f"  {name}: {r['completed']} ok / {r['failed']} failed, {r['output_throughput']:.0f} out tok/s, "
          f"p99 TTFT {r.get('p99_ttft_ms', float('nan')):.0f} ms, p99 ITL {r.get('p99_itl_ms', float('nan')):.1f} ms",
          flush=True)
    return r


def with_server(cfg, system_name, extra, log_name, fn):
    system = cfg["systems"][system_name]
    cmd = server_cmd(cfg, system, extra)
    os.makedirs(RAW, exist_ok=True)
    proc = start_server(cmd, cfg["port"], os.path.join(RAW, log_name))
    try:
        return fn()
    finally:
        stop_server(proc)


def find_saturation(cfg, workloads):
    out = {}
    path = os.path.join(RAW, "saturation.json")
    if os.path.exists(path):
        out = json.load(open(path))
    for sname in cfg["systems"]:
        def body():
            for wname in workloads:
                key = f"{sname}/{wname}"
                if key in out:
                    continue
                w = cfg["workloads"][wname]
                d = os.path.join(RAW, "saturation", sname)
                run_point(cfg, cfg["port"], w, "inf", cfg["warmup_prompts"], d, f"{wname}_warmup")
                r = run_point(cfg, cfg["port"], w, "inf", cfg["saturation_prompts"], d, wname)
                out[key] = {"request_throughput": r["request_throughput"], "output_throughput": r["output_throughput"]}
                json.dump(out, open(path, "w"), indent=2)
        with_server(cfg, sname, None, f"server_saturation_{sname}.log", body)
    print(json.dumps(out, indent=2))


def run_matrix(cfg, workloads):
    for sname in cfg["systems"]:
        def body():
            for wname in workloads:
                rates = cfg["rates"].get(wname)
                if not rates:
                    print(f"  no rates for {wname}; skipping", flush=True)
                    continue
                w, d = cfg["workloads"][wname], os.path.join(RAW, sname, wname)
                run_point(cfg, cfg["port"], w, max(rates), cfg["warmup_prompts"], d, "warmup")
                for rate in rates:
                    run_point(cfg, cfg["port"], w, rate, cfg["num_prompts"], d, f"rate_{rate}")
        print(f"== {sname}", flush=True)
        with_server(cfg, sname, None, f"server_{sname}.log", body)


def run_knob(cfg):
    k = cfg["knob_sweep"]
    w, rates = cfg["workloads"][k["workload"]], k["rates"]
    assert rates, "set knob_sweep.rates in matrix.json"
    for v in k["max_prefill_tokens_per_step"]:
        def body():
            d = os.path.join(RAW, "knob", f"max_prefill_{v}", k["workload"])
            run_point(cfg, cfg["port"], w, max(rates), cfg["warmup_prompts"], d, "warmup")
            for rate in rates:
                run_point(cfg, cfg["port"], w, rate, cfg["num_prompts"], d, f"rate_{rate}")
        print(f"== knob max_prefill_tokens_per_step={v}", flush=True)
        with_server(cfg, k["system"], ["--max-prefill-tokens-per-step", str(v)], f"server_knob_{v}.log", body)


def plan(cfg, workloads, sat_estimate):
    """Commands and a runtime estimate (no GPU). Point time ~ num_prompts / min(rate, saturation)
    plus a drain tail; +90 s per server start; warmups counted."""
    total = 0.0
    lines = []
    for sname, system in cfg["systems"].items():
        lines.append(f"# {sname}: " + " ".join(server_cmd(cfg, system)))
        total += 90
        for wname in workloads:
            rates = cfg["rates"].get(wname) or []
            if not rates:
                lines.append(f"#   {wname}: rates not set yet (run --find-saturation first)")
                continue
            sat = sat_estimate.get(f"{sname}/{wname}", 4.0)
            total += cfg["warmup_prompts"] / min(max(rates), sat) + 30          # discarded warmup
            for rate in rates:
                total += cfg["num_prompts"] / min(float(rate), sat) + 30      # +30 s drain / startup
    w = cfg["workloads"][workloads[0]]
    lines.append("# bench (per point): " + " ".join([VLLM_BIN, "bench", "serve", "--backend", "openai",
                 "--base-url", f"http://127.0.0.1:{cfg['port']}", "--endpoint", "/v1/completions", "--model",
                 cfg["served_model_name"], "--tokenizer", cfg["model"], *bench_args(cfg, w, "<RATE>", cfg["num_prompts"])]))
    print("\n".join(lines))
    print(f"# estimated matrix runtime: {total / 3600:.1f} h (excluding saturation search and knob sweep)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "benchmarks", "matrix.json"))
    ap.add_argument("--workloads", default=None, help="comma-separated subset (default: all non-optional)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--find-saturation", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--knob", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    workloads = args.workloads.split(",") if args.workloads else \
        [w for w, spec in cfg["workloads"].items() if not spec.get("optional")]
    os.chdir(ROOT)
    if args.plan:
        sat_path = os.path.join(RAW, "saturation.json")
        sat = {k: v["request_throughput"] for k, v in json.load(open(sat_path)).items()} if os.path.exists(sat_path) else {}
        plan(cfg, workloads, sat)
    elif args.find_saturation:
        find_saturation(cfg, workloads)
    elif args.run:
        run_matrix(cfg, workloads)
    else:
        run_knob(cfg)


if __name__ == "__main__":
    main()
