"""A CPU-only stand-in for LLMEngine: real Scheduler + BlockManager, fake model runner.

`FakeEngine.step()` mirrors `LLMEngine.step()` exactly (schedule -> run -> postprocess),
except the model runner returns random token ids. Before "running", the fake runner
asserts the same preconditions the real `ModelRunner.prepare_prefill/prepare_decode`
rely on (block tables cover every slot that will be written / read).
"""
import random
from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_config(num_blocks=64, block_size=4, max_num_seqs=8, max_num_batched_tokens=32, eos=0,
                scheduler_policy="prefill_first", max_prefill_tokens_per_step=-1):
    return SimpleNamespace(
        num_kvcache_blocks=num_blocks,
        kvcache_block_size=block_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=eos,
        scheduler_policy=scheduler_policy,
        max_prefill_tokens_per_step=max_prefill_tokens_per_step,
    )


def ceil_div(a, b):
    return (a + b - 1) // b


def check_block_manager(bm: BlockManager, live_seqs=()):
    """Structural invariants of the paged KV cache."""
    n = len(bm.blocks)
    free = list(bm.free_block_ids)
    used = bm.used_block_ids
    assert len(free) == len(set(free)), "duplicate id in free list"
    assert not (set(free) & used), "block both free and used"
    assert set(free) | used == set(range(n)), "block neither free nor used"
    for b in bm.blocks:
        assert b.ref_count >= 0, f"negative ref_count on block {b.block_id}"
    for bid in free:
        assert bm.blocks[bid].ref_count == 0
    # ref_count == number of live block tables that hold the block
    holders = {}
    for seq in live_seqs:
        for bid in seq.block_table:
            holders[bid] = holders.get(bid, 0) + 1
    for bid in used:
        assert bm.blocks[bid].ref_count == holders.get(bid, 0) > 0, (
            f"block {bid}: ref_count={bm.blocks[bid].ref_count}, holders={holders.get(bid, 0)}"
        )
    assert set(holders) == used
    # prefix-cache index only points at blocks carrying that hash
    for h, bid in bm.hash_to_block_id.items():
        assert bm.blocks[bid].hash == h


class FakeEngine:

    def __init__(self, config, seed=0, vocab_size=50):
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.scheduler = Scheduler(config)
        self.rng = random.Random(seed)
        self.vocab_size = vocab_size
        self.seqs: list[Sequence] = []
        self.index: dict[int, int] = {}  # global seq_id -> local index (stable across test order)
        # per step: [is_prefill, [[local_idx, num_scheduled_tokens], ...]]
        self.trace = []
        # callables (engine, seqs, is_prefill) run after schedule(), before the model runs
        self.step_checks = []

    @property
    def bm(self):
        return self.scheduler.block_manager

    def add(self, token_ids, max_tokens=8, ignore_eos=True):
        seq = Sequence(token_ids, SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=ignore_eos))
        self.scheduler.add(seq)
        self.index[seq.seq_id] = len(self.seqs)
        self.seqs.append(seq)
        return seq

    def live_seqs(self):
        return [s for s in self.seqs if s.block_table]

    def _fake_run(self, seqs, is_prefill):
        bs = self.config.kvcache_block_size
        for seq in seqs:
            assert seq.num_scheduled_tokens >= 1
            if is_prefill and seq.is_prefill:
                end = seq.num_cached_tokens + seq.num_scheduled_tokens
                assert end <= seq.num_tokens
                assert len(seq.block_table) >= ceil_div(end, bs)
            else:   # decode, or a decode sequence inside a mixed (varlen) step
                assert seq.num_scheduled_tokens == 1
                assert seq.num_cached_tokens == seq.num_tokens - 1
                assert len(seq.block_table) == ceil_div(seq.num_tokens, bs)
        return [self.rng.randrange(self.vocab_size) for _ in seqs]

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        self.trace.append([is_prefill, [[self.index[s.seq_id], s.num_scheduled_tokens] for s in seqs]])
        for check in self.step_checks:
            check(self, seqs, is_prefill)
        plan = list(seqs)
        token_ids = self._fake_run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return plan, is_prefill

    def run_to_completion(self, max_steps=100_000, check_every_step=True, arrivals=None):
        """arrivals: {step_index: [(token_ids, max_tokens, ignore_eos), ...]} added before that step."""
        arrivals = dict(arrivals or {})
        steps = 0
        while not self.scheduler.is_finished() or arrivals:
            for args in arrivals.pop(steps, []):
                self.add(*args)
            if self.scheduler.is_finished():  # idle step: nothing to schedule yet
                steps += 1
                continue
            self.step()
            if check_every_step:
                check_block_manager(self.bm, self.live_seqs())
            steps += 1
            assert steps < max_steps, "scheduler did not converge (livelock?)"
        return steps

    def all_finished(self):
        return all(s.status == SequenceStatus.FINISHED for s in self.seqs)
