"""Phase 2c (GPU): greedy decoding of the same prompt alone vs inside a batch.

Batch composition changes kernel shapes / reduction order, so near-tie argmax flips are
possible; this reports how often and where outputs diverge rather than asserting equality.

    python benchmarks/greedy_determinism.py --model /workspace/hf-models/Qwen3-0.6B
"""
import argparse
import datetime
import json
import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

PROMPTS = [
    "Explain how a hash map works.", "Write a haiku about autumn.", "What is 17 * 23?",
    "List three uses of graph algorithms.", "用三句话介绍一下长城。", "Translate 'good morning' into French and German.",
    "Summarize the plot of Romeo and Juliet in two sentences.", "Why is the sky blue?",
    "Give me a Python function that reverses a string.", "What causes inflation?",
    "Describe the water cycle.", "写一首关于大海的短诗。", "What are the SOLID principles?",
    "How does TCP congestion control work?", "Name five prime numbers greater than 100.",
    "What is the difference between a process and a thread?", "Tell me a fun fact about octopuses. 🐙",
    "Explain gradient descent to a beginner.", "What is a Merkle tree?", "How do vaccines work?",
    "Compare SQL and NoSQL databases.", "What is the capital of Australia and why?",
    "Explain the Monty Hall problem.", "Write a limerick about a cat.", "解释一下什么是机器学习。",
    "What does a load balancer do?", "Describe photosynthesis briefly.", "What is Big-O notation?",
    "How does a bloom filter work?", "Give three tips for writing clean code.",
    "What is the Pythagorean theorem?", "Explain what an LLM KV cache is.",
]


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="benchmarks/results/02_greedy_determinism.json")
    args = ap.parse_args()

    model = os.path.expanduser(args.model)
    tok = AutoTokenizer.from_pretrained(model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
               for p in PROMPTS]
    llm = LLM(model, max_model_len=4096)
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)

    alone = [llm.generate([p], sp, use_tqdm=False)[0]["token_ids"] for p in prompts]
    batched = [o["token_ids"] for o in llm.generate(prompts, sp, use_tqdm=False)]
    alone_again = [llm.generate([p], sp, use_tqdm=False)[0]["token_ids"] for p in prompts[:8]]

    per_prompt = []
    agree_tokens = total_tokens = 0
    for i, (a, b) in enumerate(zip(alone, batched)):
        d = first_divergence(a, b)
        same_prefix = len(a) if d is None else d
        agree_tokens += same_prefix
        total_tokens += len(a)
        per_prompt.append({"prompt": PROMPTS[i], "first_divergence": d, "identical": d is None})
    rerun_identical = sum(a == b for a, b in zip(alone[:8], alone_again))

    result = {
        "description": "greedy (temperature=0, ignore_eos) same prompt alone (batch 1) vs in a batch of 32",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "max_tokens": args.max_tokens,
        "num_prompts": len(prompts),
        "identical_sequences": sum(p["identical"] for p in per_prompt),
        "token_agreement_before_first_divergence": agree_tokens / total_tokens,
        "alone_rerun_identical_of_8": rerun_identical,
        "per_prompt": per_prompt,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "per_prompt"}, indent=2))


if __name__ == "__main__":
    main()
