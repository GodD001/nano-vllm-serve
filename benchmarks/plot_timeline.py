"""Scheduling timeline: the same arrivals under prefill_first vs decode_first.

Drives the real Scheduler (fake model: every sampled token is 1) and draws, per step, what
each request does: decode one token, run a prefill chunk, or wait while it is running.
Scenario: three requests are streaming when a 1200-token prompt arrives (step budget 512).

    python benchmarks/plot_timeline.py   # -> docs/images/scheduling_timeline.png
"""
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nanovllm.engine.scheduler import Scheduler  # noqa: E402
from nanovllm.engine.sequence import Sequence, SequenceStatus  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402

NAMES = ["A", "B", "C", "D (1200-token prompt)"]


def simulate(policy: str, steps: int = 9):
    Sequence.block_size = 256
    cfg = SimpleNamespace(num_kvcache_blocks=64, kvcache_block_size=256, max_num_seqs=16,
                          max_num_batched_tokens=512, eos=-1, scheduler_policy=policy,
                          max_prefill_tokens_per_step=-1)
    sched = Scheduler(cfg)
    sp = SamplingParams(max_tokens=50, ignore_eos=True)
    seqs = [Sequence([7] * 40, sp) for _ in range(3)]
    for s in seqs:
        sched.add(s)
    # bring A, B, C to steady decoding
    while sched.waiting:
        batch, prefill = sched.schedule()
        sched.postprocess(batch, [1] * len(batch), prefill)
    long_seq = Sequence(list(range(1000, 2200)), sp)
    sched.add(long_seq)
    seqs.append(long_seq)
    grid = []   # per step: {seq index: "decode" | "prefill" | "stalled"}
    for _ in range(steps):
        batch, prefill = sched.schedule()
        row = {}
        for i, s in enumerate(seqs):
            if s in batch:
                row[i] = ("prefill", s.num_scheduled_tokens) if s.is_prefill else ("decode", 1)
            elif s.status == SequenceStatus.RUNNING:
                row[i] = ("stalled", 0)
        sched.postprocess(batch, [1] * len(batch), prefill)
        grid.append(row)
    return grid


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"decode": "#2ca02c", "prefill": "#ff7f0e", "stalled": "#d62728"}
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
    for ax, policy in zip(axes, ["prefill_first", "decode_first"]):
        grid = simulate(policy)
        for t, row in enumerate(grid):
            for i, (kind, n) in row.items():
                y = len(NAMES) - 1 - i
                ax.add_patch(plt.Rectangle((t + 0.05, y + 0.1), 0.9, 0.8, color=colors[kind]))
                label = {"decode": "1", "prefill": str(n), "stalled": "–"}[kind]
                ax.text(t + 0.5, y + 0.5, label, ha="center", va="center", fontsize=8, color="white")
        ax.set_xlim(0, len(grid))
        ax.set_ylim(0, len(NAMES))
        ax.set_yticks([i + 0.5 for i in range(len(NAMES))], NAMES[::-1], fontsize=8)
        ax.set_title(f"{policy}" + ("  (upstream)" if policy == "prefill_first" else "  (this fork)"), fontsize=10)
    axes[-1].set_xticks([t + 0.5 for t in range(len(grid))], [str(t + 1) for t in range(len(grid))])
    axes[-1].set_xlabel("engine step after the long prompt arrives (budget 512 tokens/step; cell = tokens run)")
    fig.legend(handles=[Patch(color=c, label=k if k != "stalled" else "running but no token this step")
                        for k, c in colors.items()], loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(ROOT, "docs", "images", "scheduling_timeline.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
