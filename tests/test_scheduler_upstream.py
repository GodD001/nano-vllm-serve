"""Safety net around the upstream `prefill_first` scheduling policy (fake model runner, CPU only)."""
import json
import os
import random

import pytest

from nanovllm.engine.sequence import SequenceStatus
from tests.fake_engine import FakeEngine, check_block_manager, make_config

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "prefill_first_trace.json")


def upstream_policy_checks(engine, seqs, is_prefill):
    """Per-step properties of the upstream policy, checked after schedule() and before the model runs."""
    cfg = engine.config
    assert 1 <= len(seqs) <= cfg.max_num_seqs
    assert len({s.seq_id for s in seqs}) == len(seqs)
    if is_prefill:
        assert sum(s.num_scheduled_tokens for s in seqs) <= cfg.max_num_batched_tokens
        for i, s in enumerate(seqs):
            assert s.is_prefill
            if i > 0:  # only the head-of-line sequence may be chunked
                assert s.num_cached_tokens + s.num_scheduled_tokens == s.num_tokens
        head = seqs[0]
        if head.num_cached_tokens + head.num_scheduled_tokens < head.num_tokens:
            assert engine.scheduler.waiting[0] is head  # partial chunk stays at the front
            assert len(seqs) == 1 or sum(s.num_scheduled_tokens for s in seqs) == cfg.max_num_batched_tokens
    else:  # decode step: never mixed with prefill, one token per sequence
        for s in seqs:
            assert not s.is_prefill
            assert s.num_scheduled_tokens == 1
            assert s.status == SequenceStatus.RUNNING


def check_finished(engine):
    assert engine.all_finished()
    for s in engine.seqs:
        assert not s.block_table and s.num_cached_tokens == 0
        n = s.num_completion_tokens
        assert 1 <= n <= s.max_tokens
        if s.ignore_eos:
            assert n == s.max_tokens
        else:
            body = s.completion_token_ids[:-1]
            assert engine.config.eos not in body
            assert n == s.max_tokens or s.completion_token_ids[-1] == engine.config.eos
    bm = engine.bm
    check_block_manager(bm, [])
    assert len(bm.free_block_ids) == len(bm.blocks)


# ---------------------------------------------------------------------------
# Randomized workloads (online-style arrivals, prefix sharing, chunking, preemption)
# ---------------------------------------------------------------------------

def random_workload(seed):
    rng = random.Random(seed)
    bs = rng.choice([4, 8])
    cfg = make_config(
        num_blocks=rng.randint(6, 40),
        block_size=bs,
        max_num_seqs=rng.randint(1, 8),
        max_num_batched_tokens=rng.randint(bs, 64),
        eos=0,
    )
    prefixes = [[rng.randrange(1, 20) for _ in range(rng.randint(bs, 3 * bs))] for _ in range(2)]
    arrivals = {}
    for _ in range(rng.randint(1, 25)):
        while True:
            prompt = (rng.choice(prefixes) if rng.random() < 0.5 else []) + \
                     [rng.randrange(1, 20) for _ in range(rng.randint(1, 4 * bs))]
            max_tokens = rng.randint(1, 12)
            if -(-(len(prompt) + max_tokens) // bs) <= cfg.num_kvcache_blocks:  # a lone seq must fit
                break
        step = rng.choice([0, 0, rng.randint(0, 40)])
        arrivals.setdefault(step, []).append((prompt, max_tokens, rng.random() < 0.6))
    return cfg, arrivals


@pytest.mark.parametrize("seed", range(200))
def test_random_workloads_respect_policy_and_finish(seed):
    cfg, arrivals = random_workload(seed)
    engine = FakeEngine(cfg, seed=seed, vocab_size=20)
    engine.step_checks.append(upstream_policy_checks)
    engine.run_to_completion(arrivals=arrivals)
    check_finished(engine)


# ---------------------------------------------------------------------------
# Deterministic scenarios
# ---------------------------------------------------------------------------

def test_chunked_prefill_only_head_of_line():
    engine = FakeEngine(make_config(num_blocks=64, block_size=4, max_num_batched_tokens=10))
    engine.add(list(range(1, 26)), max_tokens=2)  # 25 tokens -> chunked 10 / 10 / 5
    engine.add([1, 2, 3], max_tokens=2)
    engine.add([4, 5, 6], max_tokens=2)
    engine.step_checks.append(upstream_policy_checks)
    engine.run_to_completion()
    assert engine.trace[:5] == [
        [True, [[0, 10]]],
        [True, [[0, 10]]],
        [True, [[0, 5], [1, 3]]],  # seq 2 (3 tokens) doesn't fit in the 2 left and is NOT chunked
        [True, [[2, 3]]],
        [False, [[0, 1], [1, 1], [2, 1]]],
    ]
    check_finished(engine)


def test_prefill_first_stalls_running_decodes():
    """The behavior Phase 3 targets: a long arrival blocks decode for every step of its prefill."""
    engine = FakeEngine(make_config(num_blocks=64, block_size=4, max_num_batched_tokens=10))
    engine.add([1, 2, 3, 4], max_tokens=4)
    engine.add(list(range(100, 120)), max_tokens=1)  # 20 tokens (no shared prefix), doesn't fit behind seq 0
    engine.run_to_completion()
    assert engine.trace[:4] == [
        [True, [[0, 4]]],
        [True, [[1, 10]]],  # seq 0 is running but gets no token
        [True, [[1, 10]]],
        [False, [[0, 1]]],  # seq 1 finished at max_tokens=1 on its last prefill chunk
    ]
    check_finished(engine)


def test_preemption_victim_is_newest_and_goes_to_front_of_waiting():
    bs = 4
    engine = FakeEngine(make_config(num_blocks=3, block_size=bs, max_num_batched_tokens=64))
    a = engine.add([1, 2, 3, 4], max_tokens=3)
    b = engine.add([5, 6, 7, 8], max_tokens=3)
    c = engine.add([9, 10, 11, 12], max_tokens=3)
    d = engine.add(list(range(20, 29)), max_tokens=3)  # needs all 3 blocks -> stays waiting

    _, is_prefill = engine.step()
    assert is_prefill and list(engine.scheduler.running) == [a, b, c]
    assert list(engine.scheduler.waiting) == [d]
    assert len(engine.bm.free_block_ids) == 0

    # Every running seq now has 5 tokens and needs a new block for its next KV slot.
    plan, is_prefill = engine.step()
    assert not is_prefill and plan == [a]
    # a needed a block -> newest running (c) preempted; b then found none -> preempted itself.
    assert list(engine.scheduler.waiting) == [b, c, d]
    assert list(engine.scheduler.running) == [a]
    for victim in (b, c):
        assert victim.status == SequenceStatus.WAITING
        assert victim.is_prefill
        assert victim.block_table == [] and victim.num_cached_tokens == 0
        assert victim.num_completion_tokens == 1  # generated tokens are kept and recomputed later
    check_block_manager(engine.bm, engine.live_seqs())

    engine.step_checks.append(upstream_policy_checks)
    engine.run_to_completion()
    check_finished(engine)


def test_prefix_cache_hit_shortens_prefill():
    bs = 4
    engine = FakeEngine(make_config(num_blocks=32, block_size=bs, max_num_batched_tokens=64))
    prefix = list(range(1, 9))  # 2 full blocks
    engine.add(prefix + [50, 51], max_tokens=2)
    engine.step()  # prefill seq 0 and hash its full blocks
    b = engine.add(prefix + [60], max_tokens=2)
    plan, is_prefill = engine.step()
    assert is_prefill and plan == [b]
    assert engine.trace[-1] == [True, [[1, 1]]]  # only the 1 uncached token is computed
    engine.run_to_completion()
    check_finished(engine)


def test_eos_stops_unless_ignore_eos():
    engine = FakeEngine(make_config(num_blocks=32, block_size=4, eos=0), vocab_size=2)  # eos is likely
    stop = engine.add([1, 1, 1], max_tokens=50, ignore_eos=False)
    keep = engine.add([1, 1, 1], max_tokens=50, ignore_eos=True)
    engine.run_to_completion()
    assert stop.completion_token_ids[-1] == 0 and stop.num_completion_tokens < 50
    assert keep.num_completion_tokens == 50
    check_finished(engine)


# ---------------------------------------------------------------------------
# Golden trace: the exact decisions of upstream prefill_first (commit bb823b3).
# Phase 3 must keep this byte-for-byte. Regenerate only on purpose: UPDATE_GOLDEN=1.
# ---------------------------------------------------------------------------

def golden_trace():
    traces = []
    for seed in range(20):
        cfg, arrivals = random_workload(1000 + seed)
        engine = FakeEngine(cfg, seed=seed, vocab_size=20)
        engine.run_to_completion(arrivals=arrivals)
        traces.append(engine.trace)
    return traces


def test_golden_trace_prefill_first():
    traces = golden_trace()
    if os.environ.get("UPDATE_GOLDEN"):
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump(traces, f, separators=(",", ":"))
        pytest.skip("golden trace regenerated")
    with open(GOLDEN) as f:
        expected = json.load(f)
    assert traces == expected
