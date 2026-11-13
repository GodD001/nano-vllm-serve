"""Offline throughput of LLM.generate() on a fixed-length random workload.
Prints one JSON line. Used by serving_overhead.py as the in-process baseline.

    python benchmarks/offline_throughput.py --model ~/huggingface/Qwen3-0.6B --num-prompts 256 \
        --input-len 512 --output-len 256
"""
import argparse
import json
import os
import random
import time

from nanovllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--scheduler-policy", default="prefill_first")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    llm = LLM(os.path.expanduser(args.model), max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, scheduler_policy=args.scheduler_policy)
    rng = random.Random(args.seed)
    prompts = [[rng.randrange(0, 10000) for _ in range(args.input_len)] for _ in range(args.num_prompts)]
    sp = SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=args.output_len)

    llm.generate(prompts[:8], SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=16), use_tqdm=False)  # warmup
    t = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    elapsed = time.perf_counter() - t
    total_out = sum(len(o["token_ids"]) for o in outs)
    assert total_out == args.num_prompts * args.output_len
    print(json.dumps({
        "num_prompts": args.num_prompts, "input_len": args.input_len, "output_len": args.output_len,
        "duration_s": elapsed, "total_output_tokens": total_out,
        "output_throughput_tok_s": total_out / elapsed,
        "request_throughput_req_s": args.num_prompts / elapsed,
        "num_kvcache_blocks": llm.config.num_kvcache_blocks,
    }))


if __name__ == "__main__":
    main()
