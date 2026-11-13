# nano-vllm-serve

**Turning a 1.4K-line offline inference engine into an online LLM server, and fixing the scheduling stall that shows up once requests arrive while others are streaming.**

by **Chendi Li** ([@GodD001](https://github.com/GodD001)) · built on [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) by Xingkai Yu (MIT)

> 🚧 **Work in progress.** Status is in the table below. No performance numbers yet. Every number that appears here later will come from a script in this repo, with the raw JSON committed.

---

## The problem

nano-vLLM is a clean, fast **offline** engine: you hand `generate()` all prompts up front and it runs until they finish. Its scheduler is strictly **prefill-first**. A step is either all-prefill or all-decode, and as long as any request is waiting to be prefilled, no running request gets a token.

Offline that is fine. Online it is not: every new arrival, and every chunk of a long prompt, takes a whole step during which every user who is already streaming sees nothing. The trace below comes from this repo's test suite ([`test_prefill_first_stalls_running_decodes`](tests/test_scheduler_upstream.py)), with a 10-token step budget:

```
step 1  prefill  A[4 tokens]          A starts streaming
step 2  prefill  B[10 of 20 tokens]   A gets nothing
step 3  prefill  B[10 of 20 tokens]   A gets nothing
step 4  decode   A                    A resumes
```

With long prompts arriving under load, those gaps become inter-token-latency spikes for everyone.

## What I'm building

| | upstream nano-vLLM | nano-vllm-serve |
|---|---|---|
| API | offline `LLM.generate()` | + OpenAI-compatible `/v1/completions` and `/v1/chat/completions`, SSE streaming |
| Engine | synchronous loop | + `AsyncLLMEngine`: dedicated engine thread, per-request async token streams, cancellation, backpressure (429) |
| Scheduling | prefill-first, prefill and decode never share a step | + **decode-first** mixed batches: every running request gets its token, and leftover budget goes to (chunked) prefill |
| Sampling | temperature only | + greedy, top-k, top-p, stop tokens |
| Observability | none | + Prometheus `/metrics`: TTFT / ITL / E2E histograms, KV usage, preemptions |
| Evaluation | offline throughput | + online benchmark with `vllm bench serve`: both policies vs vLLM, latency percentiles vs request rate |

The upstream policy stays the default (`--scheduler-policy prefill_first`) and must make **exactly** the same decisions as upstream. A golden trace test enforces that.

## Status

| Phase | | |
|---|---|---|
| 1 | Safety-net tests for upstream scheduler / block manager (CPU only) | ✅ done |
| 0 | Upstream baseline on GPU | ⬜ |
| 2 | Async engine, OpenAI-compatible server, metrics | ⬜ |
| 3 | Decode-first mixed-batch scheduler | ⬜ |
| 4 | Benchmarks vs upstream policy and vLLM | ⬜ |
| 5 | Design doc | ⬜ |

## Results

TBD

## Tests

The scheduler and block-manager tests run on any machine, no GPU needed. They use the real `Scheduler` / `BlockManager` with a fake model runner:

```bash
pip install pytest xxhash numpy
python -m pytest -q
```

- randomized block-manager operations with ref-count / free-list invariants checked after every step
- randomized online workloads (arrivals, shared prefixes, chunked prefill, preemption), all requests must finish with every KV block returned
- a golden trace of upstream's scheduling decisions, so any change that alters `prefill_first` fails loudly

## Attribution

Everything up to commit [`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3) is upstream nano-vLLM by Xingkai Yu, with its git history kept intact. My work is the commits on top of it ([see the diff](https://github.com/GodD001/nano-vllm-serve/compare/bb823b3...main)). The upstream [`LICENSE`](LICENSE) (MIT) is unchanged, and the original README is preserved at [`docs/upstream-README.md`](docs/upstream-README.md).
