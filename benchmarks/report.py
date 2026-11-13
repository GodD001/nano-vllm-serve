"""Phase 4 report: raw `vllm bench serve` JSON + server metrics -> summary.md and plots.

    python benchmarks/report.py   # -> benchmarks/results/summary.md, docs/images/*.png

Every number written here is read from benchmarks/results/raw/phase4; nothing is typed in.
Sanity checks: client-measured TTFT / ITL means vs the server's own histograms, and equal
output-token totals across systems at each (workload, rate).
"""
import glob
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "benchmarks", "results", "raw", "phase4")
OUT_MD = os.path.join(ROOT, "benchmarks", "results", "summary.md")
IMG = os.path.join(ROOT, "docs", "images")
SYSTEM_LABEL = {"A_prefill_first": "A: nano prefill_first (upstream policy)",
                "B_decode_first": "B: nano decode_first", "C_vllm": "C: vLLM"}


def load_points():
    """{(system, workload, rate): {"bench": ..., "server": ...}} for the main matrix."""
    pts = {}
    for path in glob.glob(os.path.join(RAW, "*", "*", "rate_*.json")):
        if path.endswith(".server_metrics.json"):
            continue
        system, workload = path.split(os.sep)[-3:-1]
        if system in ("saturation", "knob"):
            continue
        rate = float(re.search(r"rate_([0-9.]+)\.json$", path).group(1))
        sm = path[:-5] + ".server_metrics.json"
        pts[(system, workload, rate)] = {"bench": json.load(open(path)),
                                         "server": json.load(open(sm)) if os.path.exists(sm) else {}}
    return pts


def counter(server: dict, *names) -> float | None:
    """Sum a counter delta over label sets, trying several metric names (nano / vLLM)."""
    d = server.get("counters_delta", {})
    for name in names:
        vals = [v for k, v in d.items() if k.split("{")[0] == name]
        if vals:
            return sum(vals)
    return None


def hist_mean(server, *bases):
    for base in bases:
        s, c = counter(server, base + "_sum"), counter(server, base + "_count")
        if s is not None and c:
            return s / c
    return None


def kv_util(server):
    g = server.get("gauges", {})
    used = next((v for k, v in g.items() if k.startswith("nanovllm_kv_cache_blocks_used")), None)
    total = next((v for k, v in g.items() if k.startswith("nanovllm_kv_cache_blocks_total")), None)
    if used and total and total["max"]:
        return used["max"] / total["max"], used["mean"] / total["max"]
    v = next((v for k, v in g.items() if k.startswith("vllm:kv_cache_usage_perc")), None)
    return (v["max"], v["mean"]) if v else (None, None)


def fmt(x, nd=1):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def row(system, rate, b, s):
    preempt = counter(s, "nanovllm_preemptions_total", "vllm:num_preemptions_total")
    bs_sum = counter(s, "nanovllm_step_batch_size_sum")
    bs_cnt = counter(s, "nanovllm_step_batch_size_count")
    kv_max, _ = kv_util(s)
    return [SYSTEM_LABEL.get(system, system), fmt(rate, 2), fmt(b.get("request_throughput"), 2),
            fmt(b.get("output_throughput"), 0),
            fmt(b.get("p50_ttft_ms"), 0), fmt(b.get("p90_ttft_ms"), 0), fmt(b.get("p99_ttft_ms"), 0),
            fmt(b.get("p50_tpot_ms")), fmt(b.get("p99_tpot_ms")),
            fmt(b.get("p50_itl_ms")), fmt(b.get("p90_itl_ms")), fmt(b.get("p99_itl_ms")),
            fmt(b.get("p99_e2el_ms"), 0), fmt(b.get("failed")),
            fmt(preempt, 0), fmt(bs_sum / bs_cnt if bs_sum and bs_cnt else None),
            "—" if kv_max is None else f"{100 * kv_max:.0f}%"]


HEADER = ["system", "rate (req/s)", "req/s", "out tok/s", "TTFT p50", "TTFT p90", "TTFT p99", "TPOT p50",
          "TPOT p99", "ITL p50", "ITL p90", "ITL p99", "E2E p99", "errors", "preemptions", "mean batch",
          "KV max"]


def table(rows):
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def sanity(pts):
    out = ["| system | workload | rate | client TTFT mean / server | client ITL mean / server | output tokens |",
           "|---|---|---|---|---|---|"]
    tokens = defaultdict(dict)
    for (system, workload, rate), p in sorted(pts.items()):
        b, s = p["bench"], p["server"]
        st = hist_mean(s, "nanovllm_time_to_first_token_seconds", "vllm:time_to_first_token_seconds")
        si = hist_mean(s, "nanovllm_inter_token_latency_seconds", "vllm:inter_token_latency_seconds",
                       "vllm:time_per_output_token_seconds")
        r_ttft = b["mean_ttft_ms"] / (1e3 * st) if st else None
        r_itl = b["mean_itl_ms"] / (1e3 * si) if si else None
        tokens[(workload, rate)][system] = b.get("total_output_tokens")
        out.append(f"| {system} | {workload} | {rate:g} | {fmt(r_ttft, 2)} | {fmt(r_itl, 2)} | "
                   f"{b.get('total_output_tokens')} |")
    mismatched = [(k, v) for k, v in tokens.items() if len(set(v.values())) > 1]
    out.append("")
    out.append("Output-token totals match across systems at every point." if not mismatched else
               "**Output-token totals differ:** " + "; ".join(f"{k}: {v}" for k, v in mismatched))
    return "\n".join(out)


def plots(pts, knob):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["(matplotlib not installed: plots skipped)"]
    os.makedirs(IMG, exist_ok=True)
    made = []
    by_w = defaultdict(lambda: defaultdict(list))
    for (system, workload, rate), p in pts.items():
        by_w[workload][system].append((rate, p["bench"]))
    for workload, systems in by_w.items():
        for key, ylabel, fname in [("p99_itl_ms", "p99 inter-token latency (ms)", "p99_itl"),
                                   ("p99_ttft_ms", "p99 time to first token (ms)", "p99_ttft"),
                                   ("output_throughput", "output throughput (tok/s)", "throughput")]:
            fig, ax = plt.subplots(figsize=(6, 4))
            for system, series in sorted(systems.items()):
                series.sort()
                ax.plot([r for r, _ in series], [b.get(key) for _, b in series], marker="o",
                        label=SYSTEM_LABEL.get(system, system))
            ax.set_xlabel("request rate (req/s)")
            ax.set_ylabel(ylabel)
            ax.set_title(workload)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            path = os.path.join(IMG, f"{workload}_{fname}.png")
            fig.tight_layout()
            fig.savefig(path, dpi=130)
            plt.close(fig)
            made.append(os.path.relpath(path, ROOT))
    if knob:
        fig, ax = plt.subplots(figsize=(6, 4))
        for rate in sorted({r for _, r in knob}):
            series = sorted((v, b) for (v, r), b in knob.items() if r == rate)
            ax.plot([b.get("p99_ttft_ms") for _, b in series], [b.get("p99_itl_ms") for _, b in series],
                    marker="o", label=f"{rate:g} req/s")
            for v, b in series:
                ax.annotate(str(v), (b.get("p99_ttft_ms"), b.get("p99_itl_ms")), fontsize=7)
        ax.set_xlabel("p99 TTFT (ms)")
        ax.set_ylabel("p99 ITL (ms)")
        ax.set_title("decode_first: max_prefill_tokens_per_step trade-off")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        path = os.path.join(IMG, "knob_tradeoff.png")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.append(os.path.relpath(path, ROOT))
    return made


def load_knob():
    out = {}
    for path in glob.glob(os.path.join(RAW, "knob", "max_prefill_*", "*", "rate_*.json")):
        if path.endswith(".server_metrics.json"):
            continue
        v = int(re.search(r"max_prefill_(\d+)", path).group(1))
        rate = float(re.search(r"rate_([0-9.]+)\.json$", path).group(1))
        out[(v, rate)] = json.load(open(path))
    return out


def main():
    pts, knob = load_points(), load_knob()
    md = ["# Phase 4 benchmark summary", "",
          "Generated by `benchmarks/report.py` from `benchmarks/results/raw/phase4/`. Latencies in ms.", ""]
    hw = os.path.join(ROOT, "benchmarks", "results", "00_upstream_offline.json")
    if os.path.exists(hw):
        env = json.load(open(hw))["environment"]
        md += [f"Hardware: {env['gpu']}; CPU {env['cpu']}; torch {env['torch']}; flash-attn {env['flash_attn']}.", ""]
    for workload in sorted({w for _, w, _ in pts}):
        md += [f"## {workload}", "", table([row(s, r, p["bench"], p["server"])
                                           for (s, w, r), p in sorted(pts.items()) if w == workload]), ""]
    if knob:
        md += ["## Knob sweep: decode_first max_prefill_tokens_per_step", "",
               "| max_prefill_tokens_per_step | rate | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | out tok/s |",
               "|---|---|---|---|---|---|---|"]
        for (v, rate), b in sorted(knob.items()):
            md.append(f"| {v} | {rate:g} | {fmt(b.get('p50_ttft_ms'), 0)} | {fmt(b.get('p99_ttft_ms'), 0)} | "
                      f"{fmt(b.get('p50_itl_ms'))} | {fmt(b.get('p99_itl_ms'))} | {fmt(b.get('output_throughput'), 0)} |")
        md.append("")
    md += ["## Sanity checks", "", sanity(pts) if pts else "(no points)", ""]
    made = plots(pts, knob)
    md += ["## Plots", ""] + [f"![{os.path.basename(p)}](../../{p})" if p.endswith(".png") else p for p in made]
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w").write("\n".join(md) + "\n")
    print(f"wrote {os.path.relpath(OUT_MD, ROOT)} ({len(pts)} points, {len(knob)} knob points, {len(made)} plots)")


if __name__ == "__main__":
    main()
