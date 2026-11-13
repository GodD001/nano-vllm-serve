"""Phase 3 (GPU): greedy outputs of the same request set under prefill_first vs decode_first.

Requests arrive on a fixed step schedule (one every --arrival-every steps) so decode_first
actually runs mixed steps. Each policy runs in its own process (one GPU process at a time).
For each request whose outputs diverge, a teacher-forced forward pass on
prompt + common prefix gives the logit margin between the two candidate tokens at the
divergence point: a small margin means a near-tie argmax flip from a different batch
composition (acceptable); a large one would be a bug.

    python benchmarks/policy_agreement.py --model /workspace/hf-models/Qwen3-0.6B
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile

PROMPTS = [
    "Explain how a hash map works.", "Write a haiku about autumn.", "What is 17 * 23? Show your work.",
    "List three uses of graph algorithms.", "用三句话介绍一下长城。", "Translate 'good morning' into French and German.",
    "Summarize Romeo and Juliet in two sentences.", "Why is the sky blue?", "Write a Python function that reverses a string.",
    "What causes inflation?", "Describe the water cycle.", "写一首关于大海的短诗。", "What are the SOLID principles?",
    "How does TCP congestion control work?", "Name five prime numbers greater than 100.",
    "What is the difference between a process and a thread?", "Tell me a fun fact about octopuses. 🐙",
    "Explain gradient descent to a beginner.", "What is a Merkle tree?", "How do vaccines work?",
    "Compare SQL and NoSQL databases.", "What is the capital of Australia and why?", "Explain the Monty Hall problem.",
    "Write a limerick about a cat.", "解释一下什么是机器学习。", "What does a load balancer do?",
    "Describe photosynthesis briefly.", "What is Big-O notation?", "How does a bloom filter work?",
    "Give three tips for writing clean code.", "What is the Pythagorean theorem?", "Explain what an LLM KV cache is.",
    "Describe the plot of a heist movie you invent, in detail. " * 8,     # a few long prompts force chunking
    "Explain the history of the printing press and its effects on Europe. " * 10,
]


def run_policy(model: str, policy: str, max_tokens: int, arrival_every: int, out_path: str):
    from nanovllm import LLM, SamplingParams
    llm = LLM(model, max_model_len=4096, scheduler_policy=policy, max_num_batched_tokens=2048)
    tok = llm.tokenizer
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
               for p in PROMPTS]
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    seqs, kinds, step = {}, {}, 0
    pending = list(enumerate(prompts))
    while pending or not llm.is_finished():
        while pending and step >= pending[0][0] * arrival_every:
            i, p = pending.pop(0)
            seqs[i] = llm.add_request(p, sp)
        if not llm.is_finished():
            llm.step()
            kinds[llm.last_step.kind] = kinds.get(llm.last_step.kind, 0) + 1
        step += 1
    json.dump({"policy": policy, "step_kinds": kinds,
               "outputs": {i: s.completion_token_ids for i, s in seqs.items()},
               "prompt_ids": {i: s.prompt_token_ids for i, s in seqs.items()}}, open(out_path, "w"))


def margins(model: str, cases: list, out_path: str):
    """cases: [(prompt_ids + common_prefix, token_a, token_b)] -> logit(a) - logit(b), top-2 gap."""
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.sequence import Sequence
    from nanovllm.utils.context import reset_context
    llm = LLM(model, max_model_len=4096)
    mr, bm = llm.model_runner, llm.scheduler.block_manager
    out = []
    with torch.inference_mode():
        for ids, a, b in cases:
            s = Sequence(ids, SamplingParams(temperature=0))
            bm.allocate(s, bm.can_allocate(s))
            s.num_scheduled_tokens = s.num_tokens - s.num_cached_tokens
            x, pos = mr.prepare_prefill([s])
            logits = mr.run_model(x, pos, True)[0].float()
            reset_context()
            bm.deallocate(s)
            top2 = logits.topk(2).values
            out.append({"logit_a_minus_b": (logits[a] - logits[b]).item(), "top2_gap": (top2[0] - top2[1]).item(),
                        "argmax": int(logits.argmax())})
    json.dump(out, open(out_path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--arrival-every", type=int, default=3)
    ap.add_argument("--out", default="benchmarks/results/03_policy_agreement.json")
    ap.add_argument("--_worker", nargs="*", help=argparse.SUPPRESS)
    args = ap.parse_args()
    model = os.path.expanduser(args.model)

    if args._worker:
        kind, *rest = args._worker
        if kind == "policy":
            run_policy(model, rest[0], args.max_tokens, args.arrival_every, rest[1])
        else:
            margins(model, json.load(open(rest[0])), rest[1])
        return

    tmp = tempfile.mkdtemp()
    runs = {}
    for policy in ("prefill_first", "decode_first"):
        path = os.path.join(tmp, f"{policy}.json")
        subprocess.run([sys.executable, __file__, "--model", model, "--max-tokens", str(args.max_tokens),
                        "--arrival-every", str(args.arrival_every), "--_worker", "policy", policy, path], check=True)
        runs[policy] = json.load(open(path))

    a, b = runs["prefill_first"]["outputs"], runs["decode_first"]["outputs"]
    per, cases = [], []
    agree = total = 0
    for i in sorted(a, key=int):
        x, y = a[i], b[i]
        d = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), None)
        agree += len(x) if d is None else d
        total += len(x)
        per.append({"request": int(i), "first_divergence": d})
        if d is not None:
            cases.append((runs["prefill_first"]["prompt_ids"][i] + x[:d], x[d], y[d]))
    if cases:
        cpath, mpath = os.path.join(tmp, "cases.json"), os.path.join(tmp, "margins.json")
        json.dump(cases, open(cpath, "w"))
        subprocess.run([sys.executable, __file__, "--model", model, "--_worker", "margins", cpath, mpath], check=True)
        for p, m in zip([p for p in per if p["first_divergence"] is not None], json.load(open(mpath))):
            p.update(m)

    result = {
        "description": "greedy outputs, prefill_first vs decode_first, staggered arrivals "
                       f"(one request every {args.arrival_every} steps), max_num_batched_tokens=2048",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "num_requests": len(per), "max_tokens": args.max_tokens,
        "step_kinds": {p: runs[p]["step_kinds"] for p in runs},
        "identical_sequences": sum(p["first_divergence"] is None for p in per),
        "token_agreement_before_first_divergence": agree / total,
        "divergences": [p for p in per if p["first_divergence"] is not None],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
