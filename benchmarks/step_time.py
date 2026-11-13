"""Phase 3 (GPU): per-step wall time (schedule + prepare + forward + sample + postprocess)
at a fixed decode batch size B:
  decode_graph  : decode-only step, CUDA graph (upstream decode path)
  decode_eager  : decode-only step, eager (isolates what CUDA graphs buy)
  mixed[P]      : B decodes + one P-token prefill in one varlen forward (decode_first)
  separate[P]   : P-token prefill-only step + decode-only step (what prefill_first does)
Medians over --repeats.

    python benchmarks/step_time.py --model /workspace/hf-models/Qwen3-0.6B
"""
import argparse
import datetime
import json
import os
import random
import statistics

import torch

from nanovllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-sizes", default="1,8,32,64,128,256")
    ap.add_argument("--prefill-lens", default="64,512,2048")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", default="benchmarks/results/03_step_time.json")
    args = ap.parse_args()

    llm = LLM(os.path.expanduser(args.model), max_model_len=4096, scheduler_policy="decode_first")
    sched, mr = llm.scheduler, llm.model_runner
    rng = random.Random(0)
    rand_ids = lambda n: [rng.randrange(0, 10000) for _ in range(n)]  # noqa: E731
    long_sp = SamplingParams(temperature=1.0, max_tokens=3000, ignore_eos=True)
    one_sp = SamplingParams(temperature=1.0, max_tokens=1, ignore_eos=True)

    def timed_step(expect_kind):
        llm.step()
        st = llm.last_step
        assert st.kind == expect_kind, (st.kind, expect_kind)
        return st.end_time - st.start_time

    rows = []
    for B in [int(x) for x in args.batch_sizes.split(",")]:
        seqs = [llm.add_request(rand_ids(args.prompt_len), long_sp) for _ in range(B)]
        while sched.waiting:
            llm.step()
        for _ in range(3):                                     # warm graphs for this bucket
            timed_step("decode")
        row = {"batch_size": B}
        row["decode_graph_ms"] = 1e3 * statistics.median(timed_step("decode") for _ in range(args.repeats))
        mr.enforce_eager = True
        row["decode_eager_ms"] = 1e3 * statistics.median(timed_step("decode") for _ in range(args.repeats))
        mr.enforce_eager = False
        for P in [int(x) for x in args.prefill_lens.split(",")]:
            mixed, separate = [], []
            for r in range(args.repeats + 1):                  # first repeat warms compile caches
                sched.policy = "decode_first"
                llm.add_request(rand_ids(P), one_sp)
                t_mixed = timed_step("mixed")
                sched.policy = "prefill_first"
                llm.add_request(rand_ids(P), one_sp)
                t_prefill = timed_step("prefill")
                t_decode = timed_step("decode")
                if r:
                    mixed.append(t_mixed)
                    separate.append(t_prefill + t_decode)
            sched.policy = "decode_first"
            row[f"mixed_P{P}_ms"] = 1e3 * statistics.median(mixed)
            row[f"separate_P{P}_ms"] = 1e3 * statistics.median(separate)
        rows.append(row)
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        for s in seqs:
            llm.abort(s.seq_id)

    result = {
        "description": "median step wall time (ms) at decode batch size B; prompt_len "
                       f"{args.prompt_len}; mixed = B decodes + one P-token prefill in one step; "
                       "separate = P-token prefill step + B-decode step (prefill_first)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
