"""LLMEngine.step() outputs, generate() on top of it, finish reasons, and Scheduler.abort()
in every sequence state. Real Scheduler / LLMEngine code, fake model runner."""
import pytest

from nanovllm.engine.sequence import SequenceStatus
from nanovllm.sampling_params import SamplingParams
from tests.fake_engine import check_block_manager
from tests.fakes import EOS, REPLY, FakeLLMEngine, FakeRunner


def live(engine):
    return list(engine.scheduler.waiting) + list(engine.scheduler.running)


def assert_all_blocks_free(engine):
    bm = engine.scheduler.block_manager
    check_block_manager(bm, live(engine))
    if not live(engine):
        assert len(bm.free_block_ids) == len(bm.blocks)


def test_step_reports_every_new_token():
    eng = FakeLLMEngine()
    a = eng.add_request([1, 2, 3], SamplingParams(max_tokens=3))
    b = eng.add_request([4, 5], SamplingParams(max_tokens=2))
    seen = {a.seq_id: [], b.seq_id: []}
    reasons = {}
    while not eng.is_finished():
        for out in eng.step():
            seen[out.seq_id].append(out.token_id)
            if out.finished:
                reasons[out.seq_id] = out.finish_reason
    assert seen[a.seq_id] == a.completion_token_ids and len(seen[a.seq_id]) == 3
    assert seen[b.seq_id] == b.completion_token_ids and len(seen[b.seq_id]) == 2
    assert reasons == {a.seq_id: "length", b.seq_id: "length"}
    assert_all_blocks_free(eng)


def test_partial_prefill_chunk_yields_no_token():
    eng = FakeLLMEngine(max_num_batched_tokens=8)
    eng.add_request(list(range(1, 21)), SamplingParams(max_tokens=1))  # 20 tokens -> 8 / 8 / 4
    assert eng.step() == [] and eng.last_step.kind == "prefill" and eng.last_step.num_tokens == 8
    assert eng.step() == []
    outs = eng.step()
    assert len(outs) == 1 and outs[0].finished and outs[0].finish_reason == "length"


def test_step_stats_and_first_schedule():
    eng = FakeLLMEngine()
    a = eng.add_request([1, 2, 3], SamplingParams(max_tokens=2))
    eng.step()
    assert eng.last_step.newly_scheduled == [a.seq_id]
    assert eng.last_step.kind == "prefill" and eng.last_step.num_seqs == 1 and eng.last_step.num_tokens == 3
    eng.step()
    assert eng.last_step.newly_scheduled == [] and eng.last_step.kind == "decode"


def test_generate_offline_api_still_works():
    eng = FakeLLMEngine()
    outs = eng.generate(["hi", "there"], SamplingParams(max_tokens=len(REPLY.encode())), use_tqdm=False)
    assert [o["text"] for o in outs] == [REPLY, REPLY]
    assert_all_blocks_free(eng)


def test_eos_and_stop_token_finish_with_stop():
    eng = FakeLLMEngine(FakeRunner(eos_after=2))   # every sequence emits EOS as its 3rd token
    a = eng.add_request([1], SamplingParams(max_tokens=10))
    c = eng.add_request([1], SamplingParams(max_tokens=10, ignore_eos=True))
    while not eng.is_finished():
        eng.step()
    assert a.finish_reason == "stop" and a.completion_token_ids[-1] == EOS and a.num_completion_tokens == 3
    assert c.finish_reason == "length" and c.num_completion_tokens == 10   # EOS ignored
    assert_all_blocks_free(eng)

    eng = FakeLLMEngine()
    comma = REPLY.encode().index(b",")   # first occurrence -> stops exactly there
    b = eng.add_request([1], SamplingParams(max_tokens=10, stop_token_ids=[ord(",")]))
    while not eng.is_finished():
        eng.step()
    assert b.finish_reason == "stop" and b.num_completion_tokens == comma + 1   # stop token included, like EOS
    assert_all_blocks_free(eng)


# ---------------------------------------------------------------------------- abort

def test_abort_waiting_never_scheduled():
    eng = FakeLLMEngine()
    a = eng.add_request([1, 2, 3], SamplingParams())
    assert eng.abort(a.seq_id) is a
    assert a.status == SequenceStatus.FINISHED and a.finish_reason == "abort"
    assert eng.is_finished()
    assert_all_blocks_free(eng)


def test_abort_mid_chunked_prefill_frees_blocks():
    eng = FakeLLMEngine(max_num_batched_tokens=16, block_size=16)
    a = eng.add_request(list(range(1, 60)), SamplingParams())  # 59 tokens, 4 blocks, 4 chunks
    eng.step()
    assert a.block_table and a.num_cached_tokens == 16 and a in eng.scheduler.waiting
    eng.abort(a.seq_id)
    assert not a.block_table and eng.is_finished()
    assert_all_blocks_free(eng)


def test_abort_running_and_shared_prefix_refcounts():
    eng = FakeLLMEngine(block_size=16)
    prefix = list(range(1, 33))  # 2 full blocks
    a = eng.add_request(prefix + [100], SamplingParams(max_tokens=50))
    eng.step()
    b = eng.add_request(prefix + [200], SamplingParams(max_tokens=50))
    eng.step()
    bm = eng.scheduler.block_manager
    shared = a.block_table[:2]
    assert b.block_table[:2] == shared and all(bm.blocks[i].ref_count == 2 for i in shared)
    eng.abort(a.seq_id)
    assert all(bm.blocks[i].ref_count == 1 for i in shared)
    assert_all_blocks_free(eng)
    eng.step()  # b keeps decoding on its own
    eng.abort(b.seq_id)
    assert eng.is_finished()
    assert_all_blocks_free(eng)


def test_abort_preempted_sequence():
    eng = FakeLLMEngine(num_blocks=3, block_size=4, max_num_batched_tokens=64)
    # 4 prompt + 8 generated = 12 tokens = 3 blocks: each fits alone, not all together
    seqs = [eng.add_request([i, i, i, i], SamplingParams(max_tokens=8)) for i in (1, 2, 3)]
    eng.step()   # prefill all three (3 blocks)
    eng.step()   # decode: needs new blocks -> preemptions
    preempted = [s for s in seqs if s.status == SequenceStatus.WAITING]
    assert preempted
    for s in preempted:
        eng.abort(s.seq_id)
    assert_all_blocks_free(eng)
    while not eng.is_finished():
        eng.step()
    assert_all_blocks_free(eng)


def test_abort_unknown_or_finished_is_noop():
    eng = FakeLLMEngine()
    a = eng.add_request([1], SamplingParams(max_tokens=1))
    eng.step()
    assert a.is_finished
    assert eng.abort(a.seq_id) is None
    assert eng.abort(12345678) is None


@pytest.mark.parametrize("seed", range(30))
def test_random_aborts_keep_block_accounting(seed):
    import random
    rng = random.Random(seed)
    eng = FakeLLMEngine(num_blocks=12, block_size=4, max_num_seqs=4, max_num_batched_tokens=12)
    for _ in range(200):
        r = rng.random()
        if r < 0.3:
            eng.add_request([rng.randrange(1, 50) for _ in range(rng.randint(1, 16))],
                            SamplingParams(max_tokens=rng.randint(1, 12)))
        elif r < 0.45 and live(eng):
            eng.abort(rng.choice(live(eng)).seq_id)
        elif not eng.is_finished():
            eng.step()
        assert_all_blocks_free(eng)
    for s in live(eng):
        eng.abort(s.seq_id)
    assert_all_blocks_free(eng)
