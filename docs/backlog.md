# Backlog

Open items from the Phase 2 code review, to handle in Phase 3 or document in Phase 5.

## Measure in Phase 3 / 4
- **Full-vocab sort for top-k / top-p.** If any row in a batch uses top-k or top-p, the whole
  batch is sorted over the vocabulary (~152K for Qwen3) every step. Measure ITL with and without
  `top_p` under `vllm bench serve`. Options: `torch.topk` when only top-k is set; FlashInfer's
  `top_k_top_p_sampling_from_probs`.
- **GIL contention.** `batch_decode`, JSON serialization and SSE run on the event-loop thread and
  compete with the engine thread (scheduling and `prepare_*` are pure Python). Compare the server's
  `step_duration` with client-measured ITL as concurrency grows. vLLM V1 moves detokenization to a
  separate process for this reason.

## Known limitations (README)
- `prefix_cache_hit_blocks` also counts a preempted sequence re-hitting its own blocks, so it
  overstates cross-request prefix sharing (stated in the metric's help text).
- If the process exits without the lifespan shutdown, `atexit` calls `LLMEngine.exit()` on the
  main thread while the daemon engine thread may be mid-step; `_exited` is not locked. Low impact.
- `seed` is accepted and ignored.

## Behavior changes vs upstream (README)
- `LLMEngine.step()` now returns `list[TokenOutput]` (was `(outputs, num_tokens)`); per-step
  stats are on `engine.last_step`. `LLM.generate()` is unchanged.
- `temperature < 1e-5` is greedy (argmax). Upstream rejected `temperature <= 1e-10` and sampled
  above that.
- `SamplingParams` validation raises `ValueError` (upstream used `assert`).
