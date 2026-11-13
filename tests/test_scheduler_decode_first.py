"""decode_first policy (fake model runner, CPU only).

Per step: every running sequence decodes one token (while running <= max_num_seqs), then the
leftover token budget min(max_num_batched_tokens - decodes, max_prefill_tokens_per_step) is
filled with FCFS prefill where only the first admitted sequence may be chunked."""
import random

import pytest

from nanovllm.engine.sequence import SequenceStatus
from nanovllm.sampling_params import SamplingParams
from tests.fake_engine import FakeEngine, check_block_manager, make_config
from tests.fakes import FakeLLMEngine
from tests.test_scheduler_upstream import check_finished, random_workload


def decode_first_checks(engine, seqs, is_prefill):
    cfg = engine.config
    sched = engine.scheduler
    assert 1 <= len(seqs) <= cfg.max_num_seqs
    assert len({s.seq_id for s in seqs}) == len(seqs)
    decodes = [s for s in seqs if not s.is_prefill]
    prefills = [s for s in seqs if s.is_prefill]
    assert seqs == decodes + prefills                       # decodes first
    assert all(s.num_scheduled_tokens == 1 for s in decodes)
    assert is_prefill == bool(prefills)
    budget = min(cfg.max_num_batched_tokens - len(decodes), sched.max_prefill_tokens_per_step)
    assert sum(s.num_scheduled_tokens for s in prefills) <= budget
    for i, s in enumerate(prefills):
        if i > 0:                                            # only the first prefill may be chunked
            assert s.num_cached_tokens + s.num_scheduled_tokens == s.num_tokens
    # the decode-first guarantee: nobody already running is skipped this step
    ids = {s.seq_id for s in seqs}
    if len(sched.running) <= cfg.max_num_seqs:
        assert all(s.seq_id in ids for s in sched.running)


def df_config(**kw):
    kw.setdefault("scheduler_policy", "decode_first")
    return make_config(**kw)


# ---------------------------------------------------------------------------- deterministic

def test_running_request_decodes_every_step_while_long_prompt_prefills():
    """Same arrivals as tests/test_scheduler_upstream.py::test_prefill_first_stalls_running_decodes."""
    engine = FakeEngine(df_config(num_blocks=64, block_size=4, max_num_batched_tokens=10))
    engine.add([1, 2, 3, 4], max_tokens=6)
    engine.add(list(range(100, 120)), max_tokens=1)   # 20 tokens
    engine.step_checks.append(decode_first_checks)
    engine.run_to_completion()
    assert engine.trace[:5] == [
        [True, [[0, 4]]],                 # nothing running: plain prefill (B does not fit behind A)
        [True, [[0, 1], [1, 9]]],         # A decodes AND B prefills 9 = 10 - 1 decode
        [True, [[0, 1], [1, 9]]],
        [True, [[0, 1], [1, 2]]],         # B completes (and finishes at max_tokens=1)
        [False, [[0, 1]]],
    ]
    check_finished(engine)


def test_prefill_cap_limits_tokens_per_step():
    engine = FakeEngine(df_config(num_blocks=64, block_size=4, max_num_batched_tokens=64,
                                  max_prefill_tokens_per_step=3))
    engine.add([1, 2], max_tokens=10)
    engine.add(list(range(100, 110)), max_tokens=1)   # 10 tokens -> chunks 3, 3, 3, 1
    engine.step_checks.append(decode_first_checks)
    engine.run_to_completion()
    assert engine.trace[:6] == [
        [True, [[0, 2]]],                 # cap 3: A (2); B is not head-of-line, so not chunked
        [True, [[0, 1], [1, 3]]],         # 3 prefill tokens ride along with A's decode
        [True, [[0, 1], [1, 3]]],
        [True, [[0, 1], [1, 3]]],
        [True, [[0, 1], [1, 1]]],
        [False, [[0, 1]]],
    ]
    check_finished(engine)


def test_only_head_of_line_is_chunked():
    engine = FakeEngine(df_config(num_blocks=64, block_size=4, max_num_batched_tokens=10))
    engine.add([1], max_tokens=20)
    engine.step()                                      # A running
    engine.add(list(range(100, 120)), max_tokens=2)    # B: 20 tokens
    engine.add([7, 8, 9], max_tokens=2)                # C: 3 tokens, must not be chunked
    engine.step_checks.append(decode_first_checks)
    engine.run_to_completion()
    c_steps = [entry for entry in engine.trace if any(i == 2 for i, _ in entry[1])]
    first_c = [n for i, n in c_steps[0][1] if i == 2][0]
    assert first_c == 3
    check_finished(engine)


def test_preemption_in_decode_first_skips_admission():
    bs = 4
    engine = FakeEngine(df_config(num_blocks=3, block_size=bs, max_num_batched_tokens=64))
    a = engine.add([1, 2, 3, 4], max_tokens=3)
    b = engine.add([5, 6, 7, 8], max_tokens=3)
    c = engine.add([9, 10, 11, 12], max_tokens=3)
    d = engine.add(list(range(20, 29)), max_tokens=3)
    engine.step()
    assert list(engine.scheduler.running) == [a, b, c] and list(engine.scheduler.waiting) == [d]
    plan, is_prefill = engine.step()
    assert plan == [a] and not is_prefill             # a preempted c, b preempted itself; no admission
    assert list(engine.scheduler.waiting) == [b, c, d]
    for victim in (b, c):
        assert victim.status == SequenceStatus.WAITING and victim.block_table == []
    engine.step_checks.append(decode_first_checks)
    engine.run_to_completion()
    check_finished(engine)


def test_waiting_request_is_not_starved_by_full_batch():
    engine = FakeEngine(df_config(num_blocks=64, block_size=4, max_num_seqs=2, max_num_batched_tokens=64))
    engine.add([1], max_tokens=30)
    engine.add([2], max_tokens=5)
    engine.step()
    late = engine.add([3, 3], max_tokens=4)            # no seat until the short one finishes
    engine.step_checks.append(decode_first_checks)
    engine.run_to_completion()
    assert late.status == SequenceStatus.FINISHED
    check_finished(engine)


# ---------------------------------------------------------------------------- randomized

@pytest.mark.parametrize("seed", range(200))
def test_random_workloads_respect_policy_and_finish(seed):
    cfg, arrivals = random_workload(seed)
    rng = random.Random(seed)
    cfg.scheduler_policy = "decode_first"
    cfg.max_prefill_tokens_per_step = rng.choice([-1, 1, 3, cfg.max_num_batched_tokens // 2])
    engine = FakeEngine(cfg, seed=seed, vocab_size=20)
    engine.step_checks.append(decode_first_checks)
    engine.run_to_completion(arrivals=arrivals)
    check_finished(engine)


@pytest.mark.parametrize("seed", range(50))
def test_first_step_matches_upstream_when_idle(seed):
    """With nothing running, decode_first degenerates to a prefill step identical to upstream's."""
    traces = []
    for policy in ("prefill_first", "decode_first"):
        cfg, arrivals = random_workload(seed)
        cfg.scheduler_policy = policy
        engine = FakeEngine(cfg, seed=seed, vocab_size=20)
        for args in arrivals.get(0, []):
            engine.add(*args)
        if engine.scheduler.is_finished():
            return
        engine.step()
        traces.append(engine.trace[0])
    assert traces[0] == traces[1]


@pytest.mark.parametrize("seed", range(30))
def test_random_aborts_during_mixed_steps(seed):
    rng = random.Random(seed)
    eng = FakeLLMEngine(num_blocks=12, block_size=4, max_num_seqs=4, max_num_batched_tokens=12,
                        scheduler_policy="decode_first", max_prefill_tokens_per_step=rng.choice([-1, 2, 5]))
    kinds = set()

    def live():
        return list(eng.scheduler.waiting) + list(eng.scheduler.running)

    for _ in range(300):
        r = rng.random()
        if r < 0.3:
            eng.add_request([rng.randrange(1, 50) for _ in range(rng.randint(1, 16))],
                            SamplingParams(max_tokens=rng.randint(1, 12)))
        elif r < 0.45 and live():
            eng.abort(rng.choice(live()).seq_id)
        elif not eng.is_finished():
            eng.step()
            kinds.add(eng.last_step.kind)
        check_block_manager(eng.scheduler.block_manager, live())
    for s in live():
        eng.abort(s.seq_id)
    bm = eng.scheduler.block_manager
    check_block_manager(bm, [])
    assert len(bm.free_block_ids) == len(bm.blocks)
    assert "mixed" in kinds


def test_engine_reports_mixed_steps():
    eng = FakeLLMEngine(scheduler_policy="decode_first", max_num_batched_tokens=16)
    a = eng.add_request([1, 2, 3], SamplingParams(max_tokens=10))
    eng.step()
    assert eng.last_step.kind == "prefill"
    eng.step()
    assert eng.last_step.kind == "decode"
    b = eng.add_request(list(range(10, 40)), SamplingParams(max_tokens=2))   # 30 tokens
    outs = eng.step()
    assert eng.last_step.kind == "mixed"
    assert eng.last_step.num_seqs == 2 and eng.last_step.num_tokens == 16    # 1 decode + 15 prefill
    assert [o.seq_id for o in outs] == [a.seq_id]                             # b's chunk is partial
    while not eng.is_finished():
        eng.step()
    assert a.num_completion_tokens == 10 and b.num_completion_tokens == 2
