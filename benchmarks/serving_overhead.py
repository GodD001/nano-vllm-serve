"""Phase 2d: cost of the online serving layer.

Same workload both ways (N requests, fixed input/output lengths, ignore_eos, temperature 1.0):
  offline: LLM.generate() in-process                      (benchmarks/offline_throughput.py)
  online : nanovllm.serve + `vllm bench serve` at request rate inf (all requests at t=0)
overhead = 1 - online_output_throughput / offline_output_throughput

Optionally profiles the server with py-spy during one online run (--profile).

    python benchmarks/serving_overhead.py --model /workspace/hf-models/Qwen3-0.6B
"""
import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_utils import assert_gpu_idle, scrape_metrics, start_server, stop_server, vllm_bench_serve  # noqa: E402


def run_offline(args) -> dict:
    assert_gpu_idle()
    cmd = [sys.executable, "benchmarks/offline_throughput.py", "--model", args.model,
           "--num-prompts", str(args.num_prompts), "--input-len", str(args.input_len),
           "--output-len", str(args.output_len), "--seed", "0"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stderr[-4000:])
        raise SystemExit("offline run failed")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", default="Qwen3-0.6B")
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--profile", action="store_true", help="py-spy record the server during the last online run")
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. 'after')")
    ap.add_argument("--server-arg", action="append", default=[],
                    help="extra argument for nanovllm.serve (repeatable), e.g. --server-arg=--gil-switch-interval-ms=1")
    ap.add_argument("--skip-offline", action="store_true", help="reuse no offline runs (online only)")
    ap.add_argument("--out-dir", default="benchmarks/results")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    raw_dir = os.path.join(args.out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    offline = []
    for i in range(0 if args.skip_offline else args.runs):
        r = run_offline(args)
        offline.append(r)
        print(f"offline run {i}: {r['output_throughput_tok_s']:.1f} tok/s", flush=True)

    server_cmd = [sys.executable, "-m", "nanovllm.serve", "--model", args.model, "--port", str(args.port),
                  "--served-model-name", args.served_model_name, "--scheduler-policy", "prefill_first",
                  "--max-waiting-requests", str(max(1024, args.num_prompts)), *args.server_arg]
    proc = start_server(server_cmd, args.port, os.path.join(raw_dir, f"02_overhead_server{tag}.log"))
    online, metrics_delta = [], []
    try:
        bench_args = ["--dataset-name", "random", "--random-input-len", str(args.input_len),
                      "--random-output-len", str(args.output_len), "--random-range-ratio", "0.0",
                      "--num-prompts", str(args.num_prompts), "--request-rate", "inf",
                      "--ignore-eos", "--seed", "0", "--num-warmups", "16"]
        for i in range(args.runs):
            before = scrape_metrics(args.port)
            spy = None
            if args.profile and i == args.runs - 1:
                spy = subprocess.Popen(["py-spy", "record", "--pid", str(proc.pid), "--rate", "200", "--threads",
                                        "--format", "speedscope", "-o",
                                        os.path.join(raw_dir, f"02_overhead_pyspy{tag}.speedscope.json")])
            r = vllm_bench_serve(args.port, args.served_model_name, args.model, raw_dir,
                                 f"02_overhead_bench{tag}_run{i}.json", bench_args,
                                 os.path.join(raw_dir, f"02_overhead_bench{tag}_run{i}.log"))
            if spy is not None:
                spy.send_signal(2)
                spy.wait(timeout=120)
            after = scrape_metrics(args.port)
            delta = {k: after[k] - before.get(k, 0.0) for k in after
                     if k.startswith(("nanovllm_engine_steps_total", "nanovllm_step_duration_seconds_sum",
                                      "nanovllm_generation_tokens_total"))}
            metrics_delta.append(delta)
            assert r["completed"] == args.num_prompts and r["failed"] == 0, (r["completed"], r["failed"])
            assert r["total_output_tokens"] == args.num_prompts * args.output_len, r["total_output_tokens"]
            online.append(r)
            print(f"online run {i}: {r['output_throughput']:.1f} tok/s", flush=True)
    finally:
        stop_server(proc)

    off = [r["output_throughput_tok_s"] for r in offline]
    on = [r["output_throughput"] for r in online]
    summary = {
        "offline_output_tok_s_mean": statistics.mean(off) if off else None,
        "online_output_tok_s_mean": statistics.mean(on),
        "overhead_pct": 100 * (1 - statistics.mean(on) / statistics.mean(off)) if off else None,
        "offline_runs_tok_s": off,
        "online_runs_tok_s": on,
    }
    result = {
        "description": "Phase 2d: offline LLM.generate() vs online server + vllm bench serve (rate inf), same workload",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "workload": {"num_prompts": args.num_prompts, "input_len": args.input_len, "output_len": args.output_len,
                     "ignore_eos": True, "temperature": 1.0},
        "server_cmd": " ".join(server_cmd),
        "bench_cmd": online[0]["_command"] if online else None,
        "repo_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "summary": summary,
        "offline": offline,
        "online": [{k: v for k, v in r.items() if not isinstance(v, list)} for r in online],
        "server_metrics_delta": metrics_delta,
    }
    path = os.path.join(args.out_dir, f"02_serving_overhead{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
