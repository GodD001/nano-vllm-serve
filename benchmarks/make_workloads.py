"""Phase 4: generate the fixed-seed custom workloads for `vllm bench serve --dataset-name custom`.

W2 (mixed): a share of long prompts (default 25%, 2048-3072 tokens) among short ones
(192-320 tokens). This is where prefill stalls should show. Output length is set at
benchmark time with --custom-output-len (fixed, with --ignore-eos).

Prompts are random sequences of single-token English words from the model's own vocabulary
(stable under re-tokenization), cut to the exact target length; each line records the
token count measured with the same tokenizer the benchmark client uses.

    python benchmarks/make_workloads.py --model /workspace/hf-models/Qwen3-0.6B
"""
import argparse
import hashlib
import json
import os
import random

from transformers import AutoTokenizer


def word_pool(tok, rng: random.Random, size: int = 4000) -> list[str]:
    pool = []
    for tid in rng.sample(range(1000, min(tok.vocab_size, 100000)), 40000):
        s = tok.decode([tid])
        if len(s) >= 4 and s[0] == " " and s[1:].isascii() and s[1:].isalpha() and tok.encode(s) == [tid]:
            pool.append(s)
            if len(pool) >= size:
                break
    return pool


def make_prompt(tok, rng, pool, n_tokens: int) -> tuple[str, int]:
    words = rng.choices(pool, k=int(n_tokens * 1.1) + 8)
    ids = tok.encode("".join(words).lstrip())[:n_tokens]
    text = tok.decode(ids)
    return text, len(tok.encode(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--long-fraction", type=float, default=0.25)
    ap.add_argument("--short-range", default="192,320")
    ap.add_argument("--long-range", default="2048,3072")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/workloads/w2_mixed.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(os.path.expanduser(args.model))
    rng = random.Random(args.seed)
    pool = word_pool(tok, rng)
    s_lo, s_hi = map(int, args.short_range.split(","))
    l_lo, l_hi = map(int, args.long_range.split(","))
    n_long = round(args.num_prompts * args.long_fraction)
    kinds = ["long"] * n_long + ["short"] * (args.num_prompts - n_long)
    rng.shuffle(kinds)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lens = {"short": [], "long": []}
    with open(args.out, "w") as f:
        for kind in kinds:
            target = rng.randint(l_lo, l_hi) if kind == "long" else rng.randint(s_lo, s_hi)
            text, n = make_prompt(tok, rng, pool, target)
            lens[kind].append(n)
            f.write(json.dumps({"prompt": text, "prompt_tokens": n, "kind": kind}) + "\n")
    meta = {
        "file": args.out, "seed": args.seed, "num_prompts": args.num_prompts, "long_fraction": args.long_fraction,
        "short_tokens": {"min": min(lens["short"]), "max": max(lens["short"]), "mean": sum(lens["short"]) / len(lens["short"])},
        "long_tokens": {"min": min(lens["long"]), "max": max(lens["long"]), "mean": sum(lens["long"]) / len(lens["long"])},
        "tokenizer": args.model,
        # The .jsonl (~5 MB) is not committed; regenerate it with this script and check the hash.
        "sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest(),
    }
    meta_path = args.out.replace(".jsonl", ".meta.json")
    if os.path.exists(meta_path):
        # The committed meta is the reference: a regenerated file must be byte-identical.
        committed = json.load(open(meta_path))
        if committed["sha256"] != meta["sha256"]:
            raise SystemExit(f"regenerated {args.out} differs from {meta_path} "
                             f"({meta['sha256']} != {committed['sha256']}); not overwriting the meta")
        print(f"{args.out}: sha256 matches committed meta")
    else:
        json.dump(meta, open(meta_path, "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
