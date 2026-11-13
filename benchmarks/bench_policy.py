"""Phase 3 offline regression: upstream bench.py's exact workload (seed 0, 256 seqs,
input/output 100-1024 random tokens) under a chosen scheduler policy, N runs, each in its
own process. Compared with Phase 0's upstream baseline (benchmarks/results/00_upstream_offline.json).

    python benchmarks/bench_policy.py --runs 3 --out benchmarks/results/03_offline_regression.json
"""
import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys

BENCH = r'''
import os, time, json
from random import randint, seed
from nanovllm import LLM, SamplingParams
seed(0)
num_seqs, max_input_len, max_ouput_len = 256, 1024, 1024
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
llm = LLM(path, enforce_eager=False, max_model_len=4096, scheduler_policy=POLICY)
prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
llm.generate(["Benchmark: "], SamplingParams())
t = time.time()
llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
t = time.time() - t
total_tokens = sum(sp.max_tokens for sp in sampling_params)
print(json.dumps({"total_tokens": total_tokens, "time_s": t, "throughput_tok_s": total_tokens / t}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--policies", default="prefill_first,decode_first")
    ap.add_argument("--baseline", default="benchmarks/results/00_upstream_offline.json")
    ap.add_argument("--out", default="benchmarks/results/03_offline_regression.json")
    args = ap.parse_args()

    result = {"description": "upstream bench.py workload under each scheduler policy (same seed and lengths)",
              "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "repo_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                            text=True).stdout.strip(),
              "policies": {}}
    for policy in args.policies.split(","):
        runs = []
        for i in range(args.runs):
            code = BENCH.replace("POLICY", repr(policy))
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            if out.returncode != 0:
                sys.stderr.write(out.stderr[-3000:])
                raise SystemExit(f"{policy} run {i} failed")
            runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
            print(f"{policy} run {i}: {runs[-1]['throughput_tok_s']:.1f} tok/s", flush=True)
        t = [r["throughput_tok_s"] for r in runs]
        result["policies"][policy] = {"runs": runs, "mean_tok_s": statistics.mean(t),
                                      "stdev_tok_s": statistics.stdev(t) if len(t) > 1 else None}
    if os.path.exists(args.baseline):
        base = json.load(open(args.baseline))["summary"]
        result["baseline"] = {"file": args.baseline, "mean_tok_s": base["throughput_mean_tok_s"],
                              "stdev_tok_s": base["throughput_stdev_tok_s"]}
        for p, r in result["policies"].items():
            r["vs_baseline_pct"] = 100 * (r["mean_tok_s"] / base["throughput_mean_tok_s"] - 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps({p: {k: v for k, v in r.items() if k != "runs"} for p, r in result["policies"].items()}, indent=2))


if __name__ == "__main__":
    main()
