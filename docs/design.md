# nano-vllm-serve: design notes

How the server turns nano-vLLM's offline engine into an online one, and how the decode-first
scheduler works. Numbers are from `benchmarks/results/` (RTX A6000, Qwen3-0.6B).

## 1. Threading model

Two threads in one process:

- **Engine thread** — owns the `LLMEngine`: scheduler, block manager, model runner, KV cache,
  CUDA graphs. It *constructs* the engine itself, so the CUDA context, captured graphs and
  `torch.compile` state live on the thread that uses them. Nothing else touches GPU state, so the
  scheduler needs no locks.
- **Event-loop thread** (uvicorn/asyncio) — owns sockets, per-request queues, detokenizers and
  Prometheus metrics.

They communicate through two channels only:

1. **Inbox** (`queue.SimpleQueue`) of `add` / `abort` / `shutdown` commands. When the engine has
   work it drains the inbox without blocking before every step; when idle it blocks on it (no
   busy-wait).
2. **One `loop.call_soon_threadsafe` per engine step**, carrying every token that step produced.
   Per-token callbacks would cost one cross-thread wake-up per token per request.

Why not async all the way down: a model forward is a blocking GPU call; on the event loop it would
stall every socket. Why not a separate process (vLLM V1's choice): it removes GIL contention but
adds serialization and IPC; in-process was simpler and, after tuning, cost 7%.

**GIL contention was the main serving cost.** Both threads run Python. py-spy (launched as the
server's parent, since the pod forbids ptrace attach) showed the event loop holding the GIL ~3× as
often as the engine thread, so every engine step waited for it. Fixes, each measured: decode
through the Rust `tokenizers` backend in one batch per step instead of transformers' per-item
wrapper; pre-serialized SSE templates (JSON-escape only the new text); one histogram update per
distinct inter-token latency per step; and `sys.setswitchinterval(0.1 ms)` so the GIL returns to
the engine thread quickly. Overhead vs offline `generate()`: 42% → 7% (sweep in
`benchmarks/results/raw/sweep`). What remains is uvicorn/Starlette's per-chunk write path.

## 2. Request lifecycle

1. **HTTP** — FastAPI parses an OpenAI request (unknown fields ignored), applies the chat template,
   tokenizes, maps sampling fields, and calls `add_request`, which validates length and KV
   feasibility (400) and the waiting-queue bound (429) *before* any response starts.
2. **Inbox → engine** — the engine thread adds a `Sequence` to `waiting`.
3. **Steps** — `schedule()` picks sequences; `ModelRunner.run()` builds inputs, runs the model,
   samples; `postprocess()` appends tokens, detects EOS / stop tokens / max_tokens, frees KV of
   finished sequences, and returns `(seq, token)` pairs.
4. **Dispatch** — one callback per step: a single `batch_decode` for all new tokens, timestamps
   (TTFT, ITL, E2E), then one `RequestOutput` into each request's `asyncio.Queue`.
5. **Stream** — the SSE generator turns each output into one chunk. One chunk per token even when
   the token releases no text yet (half of a multi-byte character), because clients such as
   `vllm bench serve` time every chunk; `finish_reason` rides on the last token's chunk.
6. **Finish / cancel** — see §6.

**Incremental detokenization** keeps a sliding window of ids per request and emits
`decode(window)[len(decode(prefix)):]` only when it does not end in U+FFFD, so no partial UTF-8 is
sent and each token costs O(window), not O(length). Stop strings hold back `len(stop) − 1`
characters so no prefix of a stop string leaks before it is matched.

## 3. Decode-first scheduling and the mixed batch

**Problem.** Upstream's `schedule()` returns early once any waiting sequence is scheduled, so a step
is all-prefill or all-decode. Offline that maximizes throughput; online, every arrival — and every
chunk of a long prompt — is a step in which no streaming request gets a token (see the timeline in
the README). Those gaps are inter-token-latency spikes.

**Policy (`decode_first`).** Per step:
1. every running sequence gets one decode token (same KV append / preemption logic as upstream);
2. budget = `min(max_num_batched_tokens − decodes, max_prefill_tokens_per_step)`;
3. waiting sequences are admitted FCFS into that budget; only the first admitted one may be
   chunked; admission stops when the budget is spent or KV blocks cannot be allocated.

With nothing running this is exactly upstream's prefill step (tested).

**Mapping to attention.** A step with any prefill tokens is one varlen forward. A decode sequence
is just a 1-token chunk: `seqlen_q = 1`, `seqlen_k = len(seq)`, its keys read from the paged cache
through its block table. Upstream already used this path for prefix-cached prefills ("few new query
tokens + cached keys via block table"), and its LM head already takes the last query position of
each sequence, so `prepare_prefill` needed no change — the scheduler just sets per-sequence
`num_scheduled_tokens`. `postprocess` became per-sequence ("a partial chunk yields no token"),
which is equivalent for upstream's steps.

**Verification.** Kernel gate: `flash_attn_varlen_func` with `seqlen_q = 1` has the same error vs
fp32 as upstream's `flash_attn_with_kvcache` (0.0029); prefill rows inside a mixed call are
bit-identical to a prefill-only call. Model level: mixed-step logits differ from the separate
decode + prefill steps by at most 0.375 — within a 0.625 tolerance set at twice the noise upstream
already has between running the same decodes at batch size 4 and 1 (0.3125). Greedy outputs under both policies diverge only at exact ties or one
bf16 ulp.

## 4. Why CUDA graphs only for decode-only steps

A CUDA graph replays a fixed sequence of kernels with fixed shapes. Upstream captures decode at
batch-size buckets: shapes depend only on batch size. A mixed step's shapes depend on how many
prefill tokens it carries and how they split across sequences, so it runs eagerly. On this machine
an eager forward costs 24–26 ms almost independent of batch size (CPU-bound kernel launches), vs
3.5–20 ms for a graph decode step. Decode-only steps under `decode_first` still use graphs.

This is the measured cost of the policy: when arrivals are frequent, most steps are mixed. It
explains the higher p90 ITL at moderate load and the throughput loss on the shared-prefix workload
near saturation. The fix is piecewise graphs (capture everything except attention at token-count
buckets), which is what vLLM does — see §7.

## 5. TTFT ↔ ITL trade-off

Every prefill token in a step delays that step's decodes. `max_prefill_tokens_per_step` caps it:

- small cap → short mixed steps → low ITL, but prefill drains slowly → higher TTFT;
- large cap → admits long prompts in one step → low TTFT, but that step stalls all decodes.

Measured on W2 at 9 req/s: 16,384 (default) gives p99 ITL 60.6 ms / p99 TTFT 2,175 ms; 1,024 gives
43.6 / 2,662 ms; 256 starves prefill (p99 TTFT 26 s, throughput −31%).

## 6. Preemption and abort

**Preemption** (KV exhausted while decoding): as upstream, the most recently admitted running
sequence is freed and re-queued for recomputation. One decode_first-specific rule: a victim never
queues ahead of a sequence that is mid-chunked-prefill (that sequence holds KV blocks; putting a
block-less victim before it can deadlock admission once nothing is running). Randomized tests found
this. A step that had to preempt admits no prefill unless nothing else is decoding.

**Abort** (client disconnect, closed stream, stop string, shutdown) removes a sequence in any state
— waiting, mid-chunk, preempted, running — and frees its blocks with correct ref counts, including
shared prefix blocks. Aborts go through the same FIFO inbox as adds, so an abort can never overtake
its add. The HTTP layer aborts in a `finally` around the *whole response*: Starlette does not close
the body generator when a socket write fails, so a generator-level `finally` alone leaked requests
(tests drive raw ASGI sends that fail at the first, second and third message).

## 7. What I would build next

1. **Piecewise CUDA graphs for mixed steps** — removes decode_first's main cost (§4) and should
   close much of the gap to vLLM.
2. **Smaller KV blocks** (16–32 tokens) — the 256-token blocks fill the 40,960-token budget earlier
   than vLLM's 16-token blocks and trigger preemptions sooner.
3. **KV-headroom admission watermark** — hold back prefills when free blocks cannot cover running
   sequences' growth, to cut preempt-and-recompute thrash past saturation.
4. **Speculative decoding** (n-gram / draft model), and a **multi-process front end** to remove
   the last 7% of serving overhead.
