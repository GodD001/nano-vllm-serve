"""AsyncLLMEngine: an asyncio front end over the synchronous LLMEngine.

Threading model
---------------
* One dedicated **engine thread** owns the LLMEngine (scheduler, block manager, model
  runner). It constructs the engine itself, so the CUDA context, CUDA graphs and
  torch.compile state all live on that thread. Nothing else touches them.
* The **asyncio thread** (the HTTP server's event loop) owns per-request state:
  token queues, detokenizers, timestamps and Prometheus metrics.
* asyncio -> engine: a thread-safe inbox (`queue.SimpleQueue`) of add / abort / shutdown
  commands. When the engine has no work it blocks on the inbox (no busy-wait); otherwise
  it drains the inbox without blocking before every step.
* engine -> asyncio: exactly one `loop.call_soon_threadsafe` per step, carrying every
  token produced by that step. Detokenization for the whole step is one batch_decode.

Request lifecycle: add_request (sync; may raise QueueFullError -> HTTP 429) -> inbox ->
engine adds a Sequence -> steps produce tokens -> per-request asyncio.Queue -> stream().
Closing the stream before it finishes (client disconnect, cancelled task) aborts the
sequence in the engine, which frees its KV blocks.
"""
import asyncio
import itertools
import logging
import queue
import threading
from dataclasses import dataclass, field
from time import perf_counter
from typing import AsyncIterator, Callable

from nanovllm.engine.detokenizer import IncrementalDetokenizer, detokenize_step
from nanovllm.metrics import EngineSnapshot, ServerMetrics, observe_many
from nanovllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    """The bounded waiting queue is full; the request was rejected (HTTP 429)."""


class RequestFailedError(RuntimeError):
    """A request could not be completed (raised from its stream)."""


class EngineDeadError(RequestFailedError):
    """The engine thread failed; no further requests can be served."""


@dataclass(slots=True)
class RequestOutput:
    """One streamed update for one request: at most one new token."""
    request_id: str
    text: str                      # newly released text (may be "", e.g. half a UTF-8 char)
    token_ids: list[int]           # the new token, [] for a final abort notice
    finished: bool
    finish_reason: str | None      # "stop" | "length" | "abort" | None
    num_prompt_tokens: int
    num_completion_tokens: int


@dataclass
class _Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    detok: IncrementalDetokenizer
    queue: asyncio.Queue
    arrival_time: float
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    finish_time: float | None = None
    num_completion_tokens: int = 0
    finished: bool = False
    finish_reason: str | None = None
    token_times: list[float] = field(default_factory=list)


@dataclass(slots=True)
class _StepResult:
    kind: str
    num_seqs: int
    num_tokens: int
    start_time: float
    end_time: float
    newly_scheduled: list[str]
    outputs: list[tuple[str, int, bool, str | None]]   # (request_id, token_id, finished, reason)
    snapshot: EngineSnapshot


class AsyncLLMEngine:

    def __init__(self, engine_factory: Callable[[], "object"], *, max_waiting_requests: int = 1024,
                 metrics: ServerMetrics | None = None, record_token_times: bool = False):
        """engine_factory builds an LLMEngine (or a test double with the same interface)
        and is called *on the engine thread*."""
        self._engine_factory = engine_factory
        self.max_waiting_requests = max_waiting_requests
        self.metrics = metrics or ServerMetrics()
        self.record_token_times = record_token_times
        self.engine = None
        self.tokenizer = None
        self.max_model_len = None
        self.block_size = None
        self.num_kv_blocks = None
        self.snapshot = EngineSnapshot()

        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._dead: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._requests: dict[str, _Request] = {}
        self._unscheduled: set[str] = set()
        self._ids = itertools.count()
        self._stopped = False

    @classmethod
    def from_model(cls, model: str, *, max_waiting_requests: int = 1024,
                   metrics: ServerMetrics | None = None, **engine_kwargs) -> "AsyncLLMEngine":
        def factory():
            from nanovllm.engine.llm_engine import LLMEngine
            return LLMEngine(model, **engine_kwargs)
        return cls(factory, max_waiting_requests=max_waiting_requests, metrics=metrics)

    # ------------------------------------------------------------------ lifecycle
    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._engine_thread_main, name="nanovllm-engine", daemon=True)
        self._thread.start()
        await self._loop.run_in_executor(None, self._ready.wait)
        if self._startup_error is not None:
            raise RuntimeError("engine failed to start") from self._startup_error
        cfg = self.engine.config
        self.tokenizer = self.engine.tokenizer
        self.max_model_len = cfg.max_model_len
        self.block_size = cfg.kvcache_block_size
        self.num_kv_blocks = cfg.num_kvcache_blocks

    async def shutdown(self):
        """Abort everything still running, stop the engine thread, release the GPU."""
        if self._thread is None or self._stopped:
            return
        self._stopped = True
        self._inbox.put(("shutdown",))
        await self._loop.run_in_executor(None, self._thread.join)
        for req in list(self._requests.values()):
            self._finish_locally(req, "abort")

    @property
    def is_healthy(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._dead is None and not self._stopped

    @property
    def num_unscheduled(self) -> int:
        return len(self._unscheduled)

    # ------------------------------------------------------------------ requests
    def add_request(self, prompt_token_ids: list[int], sampling_params: SamplingParams,
                    stop: list[str] | None = None, request_id: str | None = None) -> str:
        """Validate and enqueue a request. Raises QueueFullError (-> 429), ValueError (-> 400)
        or EngineDeadError. Returns the request id to pass to `stream()`."""
        if self._dead is not None or self._stopped:
            raise EngineDeadError("engine is not running")
        if not prompt_token_ids:
            raise ValueError("prompt is empty")
        total = len(prompt_token_ids) + sampling_params.max_tokens
        if total > self.max_model_len:
            raise ValueError(f"prompt ({len(prompt_token_ids)} tokens) + max_tokens ({sampling_params.max_tokens}) "
                             f"exceeds max_model_len ({self.max_model_len})")
        if -(-total // self.block_size) > self.num_kv_blocks:
            raise ValueError(f"request needs more KV cache than exists ({self.num_kv_blocks} blocks)")
        if len(self._unscheduled) >= self.max_waiting_requests:
            self.metrics.requests_rejected.inc()
            raise QueueFullError(f"waiting queue is full ({self.max_waiting_requests} requests)")
        request_id = request_id or f"req-{next(self._ids)}"
        if request_id in self._requests:
            raise ValueError(f"duplicate request id {request_id}")
        req = _Request(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            detok=IncrementalDetokenizer(prompt_token_ids, stop),
            queue=asyncio.Queue(),
            arrival_time=perf_counter(),
        )
        self._requests[request_id] = req
        self._unscheduled.add(request_id)
        self.metrics.requests_received.inc()
        self.metrics.prompt_tokens.inc(len(prompt_token_ids))
        self._inbox.put(("add", request_id, req.prompt_token_ids, sampling_params))
        return request_id

    async def stream(self, request_id: str) -> AsyncIterator[RequestOutput]:
        """Yield the request's outputs until it finishes. If the consumer stops early
        (closes the generator / is cancelled), the request is aborted in the engine.

        A request may finish before its stream is first read (short requests): its
        outputs wait in its queue, and the entry is only dropped here, once consumed."""
        req = self._requests.get(request_id)
        if req is None:
            raise KeyError(request_id)
        try:
            while True:
                item = await req.queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finished:
                    return
        finally:
            if not req.finished:
                self.abort(request_id)
            self._requests.pop(request_id, None)

    async def generate(self, prompt_token_ids: list[int], sampling_params: SamplingParams,
                       stop: list[str] | None = None, request_id: str | None = None) -> AsyncIterator[RequestOutput]:
        request_id = self.add_request(prompt_token_ids, sampling_params, stop, request_id)
        async for out in self.stream(request_id):
            yield out

    def abort(self, request_id: str):
        """Abort from the asyncio side (client went away). Frees the KV blocks in the engine."""
        req = self._requests.pop(request_id, None)   # the caller is abandoning this request
        if req is None or req.finished:
            return
        self._finish_locally(req, "abort", notify=False)
        self._inbox.put(("abort", request_id))

    def request_timeline(self, request_id: str) -> _Request | None:
        return self._requests.get(request_id)

    # ------------------------------------------------------------------ asyncio side
    def _finish_locally(self, req: _Request, reason: str, notify: bool = True):
        if req.finished:
            return
        req.finished = True
        req.finish_reason = reason
        req.finish_time = perf_counter()
        self._unscheduled.discard(req.request_id)
        self.metrics.requests_finished.labels(reason).inc()
        if notify:
            req.queue.put_nowait(RequestOutput(req.request_id, req.detok.finish(), [], True, reason,
                                               len(req.prompt_token_ids), req.num_completion_tokens))

    def _on_step(self, result: _StepResult):
        """Runs on the event loop, once per engine step (via call_soon_threadsafe).

        asyncio would only log an exception raised here, leaving this step's requests
        waiting forever; instead, fail exactly those requests and abort them in the engine."""
        try:
            self._process_step(result)
        except Exception as e:  # noqa: BLE001
            logger.exception("failed to process an engine step; failing its requests")
            rids = {rid for rid, _, _, _ in result.outputs} | set(result.newly_scheduled)
            self._fail_requests(rids, e)

    def _fail_requests(self, rids, exc: BaseException):
        for rid in rids:
            req = self._requests.get(rid)
            if req is None or req.finished:
                continue
            req.finished = True
            req.finish_reason = "error"
            req.finish_time = perf_counter()
            self._unscheduled.discard(rid)
            self.metrics.requests_finished.labels("error").inc()
            req.queue.put_nowait(RequestFailedError(f"request {rid} failed: {exc!r}"))
            self._inbox.put(("abort", rid))

    @staticmethod
    def _is_stop_token(req: _Request, tok: int, engine_finished: bool, reason: str | None) -> bool:
        return engine_finished and reason == "stop" and tok in req.sampling_params.stop_token_ids

    def _process_step(self, result: _StepResult):
        m = self.metrics
        m.observe_step(result.kind, result.num_seqs, result.num_tokens, result.end_time - result.start_time)
        m.update_snapshot(result.snapshot)
        for rid in result.newly_scheduled:
            self._unscheduled.discard(rid)
            req = self._requests.get(rid)
            if req is not None:
                req.first_scheduled_time = result.start_time
                m.queue_time.observe(result.start_time - req.arrival_time)

        live = []
        for rid, tok, fin, reason in result.outputs:
            req = self._requests.get(rid)
            if req is not None and not req.finished:   # skip requests aborted on this side
                live.append((req, tok, fin, reason))
        if not live:
            return
        # A matched stop_token_id is kept in token_ids (like EOS) but its text is not emitted,
        # matching vLLM's default (include_stop_str_in_output=False).
        texts = iter(detokenize_step(self.tokenizer, [(req.detok, tok) for req, tok, fin, reason in live
                                                      if not self._is_stop_token(req, tok, fin, reason)]))
        t = result.end_time
        m.generation_tokens.inc(len(live))
        itl_groups: dict[float, int] = {}   # requests decoded together share their previous token time
        for req, tok, engine_finished, reason in live:
            text = "" if self._is_stop_token(req, tok, engine_finished, reason) else next(texts)
            req.num_completion_tokens += 1
            if req.first_token_time is None:
                req.first_token_time = t
                m.ttft.observe(t - req.arrival_time)
            else:
                itl_groups[req.last_token_time] = itl_groups.get(req.last_token_time, 0) + 1
            req.last_token_time = t
            if self.record_token_times:
                req.token_times.append(t)

            finished = engine_finished
            if req.detok.stopped:            # stop string matched: wins over "length"
                finished, reason = True, "stop"
                if not engine_finished:
                    self._inbox.put(("abort", req.request_id))
            elif engine_finished:
                text += req.detok.finish()
            if finished:
                req.finished = True
                req.finish_reason = reason
                req.finish_time = t
                self._unscheduled.discard(req.request_id)
                m.requests_finished.labels(reason).inc()
                m.e2e.observe(t - req.arrival_time)
            req.queue.put_nowait(RequestOutput(req.request_id, text, [tok], finished, reason if finished else None,
                                               len(req.prompt_token_ids), req.num_completion_tokens))
        for prev_t, count in itl_groups.items():
            observe_many(m.itl, t - prev_t, count)

    def _on_engine_dead(self, exc: BaseException):
        self._dead = exc
        err = EngineDeadError(f"engine thread failed: {exc!r}")
        for req in list(self._requests.values()):
            if not req.finished:
                req.finished = True
                req.queue.put_nowait(err)
        self._unscheduled.clear()

    # ------------------------------------------------------------------ engine thread
    def _snapshot(self) -> EngineSnapshot:
        sched = self.engine.scheduler
        bm = sched.block_manager
        return EngineSnapshot(
            num_waiting=len(sched.waiting),
            num_running=len(sched.running),
            kv_blocks_used=len(bm.used_block_ids),
            kv_blocks_total=len(bm.blocks),
            prefix_cache_hit_blocks=bm.num_prefix_cache_hit_blocks,
            preemptions=sched.num_preemptions,
        )

    def _engine_thread_main(self):
        try:
            self.engine = self._engine_factory()
        except BaseException as e:  # noqa: BLE001 - report any startup failure to start()
            self._startup_error = e
            self._ready.set()
            return
        self.snapshot = self._snapshot()
        self._ready.set()
        try:
            self._engine_loop()
        except BaseException as e:  # noqa: BLE001
            logger.exception("engine thread failed")
            self._loop.call_soon_threadsafe(self._on_engine_dead, e)
        finally:
            try:
                self.engine.exit()
            except Exception:  # noqa: BLE001
                logger.exception("engine exit failed")

    def _engine_loop(self):
        engine = self.engine
        rid_to_seq: dict[str, int] = {}
        seq_to_rid: dict[int, str] = {}
        shutting_down = False
        while not shutting_down:
            commands = [] if not engine.is_finished() else [self._inbox.get()]  # block only when idle
            while True:
                try:
                    commands.append(self._inbox.get_nowait())
                except queue.Empty:
                    break
            for cmd in commands:
                op = cmd[0]
                if op == "add":
                    _, rid, prompt_ids, sp = cmd
                    seq = engine.add_request(prompt_ids, sp)
                    rid_to_seq[rid] = seq.seq_id
                    seq_to_rid[seq.seq_id] = rid
                elif op == "abort":
                    seq_id = rid_to_seq.pop(cmd[1], None)
                    if seq_id is not None:
                        seq_to_rid.pop(seq_id, None)
                        engine.abort(seq_id)
                elif op == "shutdown":
                    shutting_down = True
            if shutting_down:
                for seq_id in list(seq_to_rid):
                    engine.abort(seq_id)
                self.snapshot = self._snapshot()
                return
            if engine.is_finished():
                self.snapshot = self._snapshot()
                continue

            outputs = engine.step()
            stats = engine.last_step
            newly_scheduled = [seq_to_rid[s] for s in stats.newly_scheduled if s in seq_to_rid]
            result_outputs = []
            for out in outputs:
                rid = seq_to_rid.get(out.seq_id)
                if rid is None:
                    continue
                result_outputs.append((rid, out.token_id, out.finished, out.finish_reason))
                if out.finished:
                    del seq_to_rid[out.seq_id]
                    rid_to_seq.pop(rid, None)
            snapshot = self._snapshot()
            self.snapshot = snapshot
            result = _StepResult(
                kind=stats.kind, num_seqs=stats.num_seqs, num_tokens=stats.num_tokens,
                start_time=stats.start_time, end_time=stats.end_time,
                newly_scheduled=newly_scheduled,
                outputs=result_outputs, snapshot=snapshot,
            )
            self._loop.call_soon_threadsafe(self._on_step, result)
