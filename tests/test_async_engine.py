"""AsyncLLMEngine: engine thread + asyncio streams, cancellation, backpressure, shutdown."""
import asyncio
import contextlib
import time

import pytest

from nanovllm.engine.async_llm_engine import AsyncLLMEngine, EngineDeadError, QueueFullError
from nanovllm.sampling_params import SamplingParams
from tests.fakes import REPLY, FakeLLMEngine, FakeRunner

REPLY_BYTES = REPLY.encode()


def expected_text(n_tokens: int) -> str:
    return REPLY_BYTES[:n_tokens].decode("utf-8", errors="ignore")


async def wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.005)


def all_blocks_free(engine: AsyncLLMEngine) -> bool:
    s = engine.snapshot
    return s.kv_blocks_used == 0 and s.num_waiting == 0 and s.num_running == 0


def make_engine(runner=None, max_waiting_requests=1024, **kw):
    runner = runner or FakeRunner()
    eng = AsyncLLMEngine(lambda: FakeLLMEngine(runner, **kw), max_waiting_requests=max_waiting_requests)
    return eng, runner


async def collect(engine, prompt, sp, stop=None):
    outs = [o async for o in engine.generate(prompt, sp, stop)]
    return outs


def run(coro):
    return asyncio.run(coro)


def test_concurrent_streams_complete_well_formed():
    async def main():
        eng, runner = make_engine(max_num_seqs=8, num_blocks=128)
        await eng.start()
        dispatches = 0
        orig = eng._on_step

        def counting(result):
            nonlocal dispatches
            dispatches += 1
            orig(result)
        eng._on_step = counting

        n_tokens = [5, 17, len(REPLY_BYTES), 1, 30] * 8   # 40 concurrent requests, 8 run at a time
        results = await asyncio.gather(*[
            collect(eng, list(b"prompt %d" % i), SamplingParams(max_tokens=n)) for i, n in enumerate(n_tokens)])
        for n, outs in zip(n_tokens, results):
            assert len(outs) == n                      # exactly one update per generated token
            assert [o.num_completion_tokens for o in outs] == list(range(1, n + 1))
            assert all(not o.finished for o in outs[:-1])
            assert outs[-1].finished and outs[-1].finish_reason == "length"
            assert "".join(o.text for o in outs) == expected_text(n)
            assert all("�" not in o.text for o in outs)
        await wait_until(lambda: all_blocks_free(eng))
        assert dispatches == runner.num_steps          # one call_soon_threadsafe per step
        await eng.shutdown()
        assert runner.exited
    run(main())


def test_close_stream_aborts_and_frees_blocks():
    async def main():
        eng, runner = make_engine(FakeRunner(step_delay=0.002), num_blocks=32)
        await eng.start()
        agen = eng.generate(list(b"long request"), SamplingParams(max_tokens=400, ignore_eos=True))
        got = [await agen.__anext__() for _ in range(3)]
        assert eng.snapshot.kv_blocks_used > 0
        await agen.aclose()                           # client went away
        await wait_until(lambda: all_blocks_free(eng))
        assert len(got) == 3
        await eng.shutdown()
    run(main())


def test_cancelled_task_aborts_and_frees_blocks():
    async def main():
        eng, runner = make_engine(FakeRunner(step_delay=0.002), num_blocks=32)
        await eng.start()
        tasks = [asyncio.ensure_future(collect(eng, list(b"x%d" % i), SamplingParams(max_tokens=400)))
                 for i in range(4)]
        await wait_until(lambda: eng.snapshot.num_running == 4)
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        await wait_until(lambda: all_blocks_free(eng))
        await eng.shutdown()
    run(main())


def test_abort_before_first_schedule():
    async def main():
        runner = FakeRunner()
        runner.gate.clear()                            # engine blocks inside its first step
        eng, _ = make_engine(runner, num_blocks=32)
        await eng.start()
        first = eng.add_request(list(b"a"), SamplingParams(max_tokens=50))
        second = eng.add_request(list(b"b"), SamplingParams(max_tokens=50))
        eng.abort(second)                              # still in the inbox / waiting queue
        runner.gate.set()
        outs = [o async for o in eng.stream(first)]
        assert outs[-1].finished and outs[-1].num_completion_tokens == 50
        await wait_until(lambda: all_blocks_free(eng))
        await eng.shutdown()
    run(main())


def test_queue_full_rejects_immediately():
    async def main():
        runner = FakeRunner()
        runner.gate.clear()
        eng, _ = make_engine(runner, max_waiting_requests=2)
        await eng.start()
        a = eng.add_request(list(b"a"), SamplingParams(max_tokens=3))
        b = eng.add_request(list(b"b"), SamplingParams(max_tokens=3))
        with pytest.raises(QueueFullError):
            eng.add_request(list(b"c"), SamplingParams(max_tokens=3))
        assert eng.metrics.requests_rejected._value.get() == 1
        runner.gate.set()
        for rid in (a, b):
            outs = [o async for o in eng.stream(rid)]
            assert outs[-1].finished
        await wait_until(lambda: eng.num_unscheduled == 0)
        eng.add_request(list(b"d"), SamplingParams(max_tokens=1))   # room again
        await eng.shutdown()
    run(main())


def test_request_validation():
    async def main():
        eng, _ = make_engine(max_model_len=64, num_blocks=8, block_size=16)
        await eng.start()
        with pytest.raises(ValueError):
            eng.add_request(list(range(60)), SamplingParams(max_tokens=10))   # > max_model_len
        with pytest.raises(ValueError):
            eng.add_request([], SamplingParams())
        await eng.shutdown()
    run(main())


def test_stop_string_finishes_and_frees_blocks():
    async def main():
        eng, _ = make_engine()
        await eng.start()
        outs = await collect(eng, list(b"p"), SamplingParams(max_tokens=200, ignore_eos=True), stop=["世界"])
        assert "".join(o.text for o in outs) == "Hello, "
        assert outs[-1].finished and outs[-1].finish_reason == "stop"
        await wait_until(lambda: all_blocks_free(eng))
        await eng.shutdown()
    run(main())


def test_shutdown_aborts_in_flight_requests():
    async def main():
        eng, runner = make_engine(FakeRunner(step_delay=0.002))
        await eng.start()
        agen = eng.generate(list(b"p"), SamplingParams(max_tokens=500))
        await agen.__anext__()
        await eng.shutdown()
        rest = [o async for o in agen]
        assert rest[-1].finished and rest[-1].finish_reason == "abort"
        assert runner.exited and not eng.is_healthy
        with pytest.raises(EngineDeadError):
            eng.add_request(list(b"q"), SamplingParams())
    run(main())


def test_engine_failure_surfaces_to_streams():
    class Boom(FakeRunner):
        def call(self, method, *args):
            if method == "run" and self.num_steps >= 2:
                raise RuntimeError("CUDA error (simulated)")
            return super().call(method, *args)

    async def main():
        eng, runner = make_engine(Boom())
        await eng.start()
        with pytest.raises(EngineDeadError):
            await collect(eng, list(b"p"), SamplingParams(max_tokens=50))
        assert not eng.is_healthy
        assert runner.exited                            # GPU released even on failure
    run(main())


def test_metrics_observed():
    async def main():
        eng, _ = make_engine()
        await eng.start()
        await asyncio.gather(*[collect(eng, list(b"p%d" % i), SamplingParams(max_tokens=6)) for i in range(5)])
        text = eng.metrics.render().decode()
        assert "nanovllm_time_to_first_token_seconds_count 5.0" in text
        assert "nanovllm_inter_token_latency_seconds_count 25.0" in text   # 5 requests x 5 gaps
        assert "nanovllm_e2e_request_latency_seconds_count 5.0" in text
        assert 'nanovllm_requests_finished_total{reason="length"} 5.0' in text
        assert 'nanovllm_engine_steps_total{type="prefill"}' in text
        assert "nanovllm_generation_tokens_total 30.0" in text
        await eng.shutdown()
    run(main())


def test_on_step_failure_fails_only_that_steps_requests(monkeypatch):
    """An exception while processing a step (e.g. detokenizer) must not leave streams hanging."""
    import nanovllm.engine.async_llm_engine as mod
    from nanovllm.engine.async_llm_engine import RequestFailedError
    real = mod.detokenize_step
    calls = {"n": 0}

    def flaky(tokenizer, items, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated detokenizer bug")
        return real(tokenizer, items, **kw)

    monkeypatch.setattr(mod, "detokenize_step", flaky)

    async def main():
        eng, runner = make_engine(FakeRunner(step_delay=0.002), num_blocks=64)
        await eng.start()
        with pytest.raises(RequestFailedError):
            await asyncio.wait_for(collect(eng, list(b"p"), SamplingParams(max_tokens=50)), timeout=5)
        assert eng.is_healthy                             # engine keeps serving
        await wait_until(lambda: all_blocks_free(eng) and not eng._requests)
        outs = await collect(eng, list(b"q"), SamplingParams(max_tokens=5))
        assert outs[-1].finished and outs[-1].finish_reason == "length"
        text = eng.metrics.render().decode()
        assert 'nanovllm_requests_finished_total{reason="error"} 1.0' in text
        await eng.shutdown()
    run(main())


def test_stop_token_text_not_emitted():
    async def main():
        eng, _ = make_engine()
        await eng.start()
        outs = await collect(eng, list(b"p"), SamplingParams(max_tokens=50, stop_token_ids=[ord(",")]))
        assert "".join(o.text for o in outs) == "Hello"          # "," is in token_ids, not in text
        assert outs[-1].token_ids == [ord(",")] and outs[-1].finish_reason == "stop"
        await eng.shutdown()
    run(main())


def test_stop_string_on_last_allowed_token_reports_stop():
    async def main():
        eng, _ = make_engine()
        await eng.start()
        # "Hello" completes exactly at token 5 == max_tokens: both stop and length apply
        outs = await collect(eng, list(b"p"), SamplingParams(max_tokens=5), stop=["llo"])
        assert outs[-1].finished and outs[-1].finish_reason == "stop"
        assert "".join(o.text for o in outs) == "He"
        await wait_until(lambda: all_blocks_free(eng))
        await eng.shutdown()
    run(main())


def test_decode_first_streams_complete_with_mixed_steps():
    async def main():
        eng = AsyncLLMEngine(lambda: FakeLLMEngine(FakeRunner(step_delay=0.001), num_blocks=128, max_num_seqs=8,
                                                   max_num_batched_tokens=32, scheduler_policy="decode_first"))
        await eng.start()

        async def staggered(i, n):
            await asyncio.sleep(0.003 * i)            # arrivals while others are decoding
            return n, await collect(eng, list(b"prompt number %d" % i), SamplingParams(max_tokens=n))

        results = await asyncio.gather(*[staggered(i, 5 + (i * 11) % 30) for i in range(24)])
        for n, outs in results:
            assert len(outs) == n and outs[-1].finish_reason == "length"
            assert "".join(o.text for o in outs) == expected_text(n)
        await wait_until(lambda: all_blocks_free(eng))
        assert 'nanovllm_engine_steps_total{type="mixed"}' in eng.metrics.render().decode()
        await eng.shutdown()
    run(main())
