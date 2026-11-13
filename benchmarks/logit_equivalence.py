"""Phase 3 (GPU): the same sequences in the same state, run as
  (a) upstream's separate steps: decode-only (CUDA graph) + prefill-only (varlen), and
  (b) one mixed step (decodes as 1-token chunks + prefill chunks, one varlen forward),
must give the same logits up to bf16 noise.

Tolerance is anchored to noise upstream already has: the same decodes run as a batch of 4
vs one at a time (batch composition changes kernel shapes / reduction order). Pass if the
mixed-vs-separate max |diff| <= 2x that noise floor (min 0.05) and top-1 tokens agree except
where the top-2 margin is below the observed diff (a near tie).

    python benchmarks/logit_equivalence.py --model /workspace/hf-models/Qwen3-0.6B
"""
import argparse
import datetime
import json
import os

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context

DECODE_PROMPTS = [
    "Write a short story about a lighthouse keeper.",
    "解释一下快速排序的时间复杂度。",
    "List the planets of the solar system in order, with one fact each. " * 6,   # > 256 tokens
    "What is the derivative of x^3 * sin(x)?",
]
PREFILL_PROMPTS = [
    "Summarize the causes of the French Revolution in three bullet points.",
    "Describe how a transformer language model generates text, step by step. " * 12,  # > 512 tokens
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--decode-steps", type=int, default=7)
    ap.add_argument("--out", default="benchmarks/results/03_logit_equivalence.json")
    args = ap.parse_args()

    llm = LLM(os.path.expanduser(args.model), max_model_len=4096, scheduler_policy="decode_first")
    mr, sched, tok = llm.model_runner, llm.scheduler, llm.tokenizer
    bm = sched.block_manager
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,  # noqa: E731
                                             add_generation_prompt=True)

    # bring the decode sequences into decode state (several tokens generated, all running)
    sp = SamplingParams(temperature=0, max_tokens=1000, ignore_eos=True)
    dec = [llm.add_request(chat(p), sp) for p in DECODE_PROMPTS]
    for _ in range(args.decode_steps):
        llm.step()
    assert all(s in sched.running and s.num_completion_tokens >= 2 for s in dec)
    assert not sched.waiting

    # set up one step by hand: every decode gets its next KV slot, prefills get their blocks
    for s in dec:
        assert bm.can_append(s)
        bm.may_append(s)
        s.num_scheduled_tokens, s.is_prefill = 1, False
    pf = [Sequence(tok.encode(chat(p)), SamplingParams(temperature=0)) for p in PREFILL_PROMPTS]
    for s in pf:
        bm.allocate(s, bm.can_allocate(s))
        s.num_scheduled_tokens = s.num_tokens - s.num_cached_tokens

    def run(seqs, prefill, eager=False):
        ids, pos = mr.prepare_prefill(seqs) if prefill else mr.prepare_decode(seqs)
        if eager:
            logits = mr.model.compute_logits(mr.model(ids, pos))
        else:
            logits = mr.run_model(ids, pos, prefill)
        reset_context()
        return logits.float()

    with torch.inference_mode():
        sep_dec = run(dec, False)                                     # upstream decode (graph, bs 4)
        sep_pf = run(pf, True)                                        # upstream prefill (varlen)
        mixed = run(dec + pf, True)                                   # one mixed step
        alone = torch.cat([run([s], False) for s in dec])             # noise floor: bs 1 vs bs 4
        eager_dec = run(dec, False, eager=True)                       # graph vs eager, same batch

    def stats(a, b):
        diff = (a - b).abs()
        top2 = a.topk(2, dim=-1)
        margin = (top2.values[:, 0] - top2.values[:, 1])
        agree = (a.argmax(-1) == b.argmax(-1))
        return {"max_abs_diff": diff.max().item(), "mean_abs_diff": diff.mean().item(),
                "top1_agree": agree.tolist(), "top1_margin": margin.tolist()}

    noise = stats(sep_dec, alone)
    res = {
        "mixed_vs_separate_decode_rows": stats(sep_dec, mixed[:len(dec)]),
        "mixed_vs_separate_prefill_rows": stats(sep_pf, mixed[len(dec):]),
        "noise_floor_decode_batch4_vs_batch1": noise,
        "decode_graph_vs_eager": stats(sep_dec, eager_dec),
        "logit_abs_max": sep_dec.abs().max().item(),
    }
    tol = max(2 * noise["max_abs_diff"], 0.05)
    ok = True
    for key in ("mixed_vs_separate_decode_rows", "mixed_vs_separate_prefill_rows"):
        r = res[key]
        within = r["max_abs_diff"] <= tol
        ties_ok = all(a or m < r["max_abs_diff"] for a, m in zip(r["top1_agree"], r["top1_margin"]))
        r["pass"] = within and ties_ok
        ok &= r["pass"]
    result = {
        "description": "logits of one mixed step vs upstream's separate decode + prefill steps (bf16)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(),
        "decode_context_lens": [len(s) for s in dec],
        "prefill_lens": [s.num_tokens for s in pf],
        "tolerance": {"max_abs_diff": tol, "rule": "<= 2x noise floor (bs4 vs bs1 decode), min 0.05; "
                                                   "top-1 must agree unless top-2 margin < observed diff"},
        "results": res,
        "pass": ok,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    summary = {k: {kk: v[kk] for kk in ("max_abs_diff", "top1_agree") if kk in v} if isinstance(v, dict) else v
               for k, v in res.items()}
    print(json.dumps({"tolerance": tol, "pass": ok, **summary}, indent=2))
    for s in pf:
        bm.deallocate(s)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
