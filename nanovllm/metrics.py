"""Prometheus metrics for the online server (prometheus_client).

All observations happen on the asyncio thread, once per engine step (plus per-request
events), so the engine thread never touches the registry.
"""
from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

# Latency buckets (seconds): fine resolution at the low end for inter-token latency.
_LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1,
                    0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0)
_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512)
_TOKEN_BUCKETS = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


@dataclass(slots=True)
class EngineSnapshot:
    """Engine state published by the engine thread after every inbox drain / step."""
    num_waiting: int = 0
    num_running: int = 0
    kv_blocks_used: int = 0
    kv_blocks_total: int = 0
    prefix_cache_hit_blocks: int = 0   # cumulative
    preemptions: int = 0               # cumulative


class ServerMetrics:

    def __init__(self, model_name: str = "", registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        p = "nanovllm_"
        self.requests_waiting = Gauge(p + "requests_waiting", "Sequences waiting to be (re)scheduled", registry=r)
        self.requests_running = Gauge(p + "requests_running", "Sequences in the running batch", registry=r)
        self.kv_blocks_used = Gauge(p + "kv_cache_blocks_used", "KV cache blocks in use", registry=r)
        self.kv_blocks_total = Gauge(p + "kv_cache_blocks_total", "KV cache blocks allocated", registry=r)
        self.steps = Counter(p + "engine_steps", "Engine steps by type", ["type"], registry=r)
        self.step_batch_size = Histogram(p + "step_batch_size", "Sequences per step", ["type"],
                                         buckets=_BATCH_BUCKETS, registry=r)
        self.step_tokens = Histogram(p + "step_num_tokens", "Tokens run through the model per step", ["type"],
                                     buckets=_TOKEN_BUCKETS, registry=r)
        self.step_duration = Histogram(p + "step_duration_seconds", "Wall time of one engine step", ["type"],
                                       buckets=_LATENCY_BUCKETS, registry=r)
        self.prefix_cache_hit_blocks = Counter(
            p + "prefix_cache_hit_blocks",
            "KV blocks reused from the prefix cache at allocation time. Includes a preempted sequence "
            "re-hitting its own earlier blocks, so it overstates cross-request prefix sharing.", registry=r)
        self.preemptions = Counter(p + "preemptions", "Sequences preempted (KV freed, recomputed later)", registry=r)
        self.requests_received = Counter(p + "requests_received", "Requests accepted", registry=r)
        self.requests_rejected = Counter(p + "requests_rejected", "Requests rejected because the waiting queue was full", registry=r)
        self.requests_finished = Counter(p + "requests_finished", "Finished requests by reason", ["reason"], registry=r)
        self.prompt_tokens = Counter(p + "prompt_tokens", "Prompt tokens of accepted requests", registry=r)
        self.generation_tokens = Counter(p + "generation_tokens", "Generated tokens", registry=r)
        self.ttft = Histogram(p + "time_to_first_token_seconds", "Arrival to first generated token",
                              buckets=_LATENCY_BUCKETS, registry=r)
        self.itl = Histogram(p + "inter_token_latency_seconds", "Time between consecutive generated tokens",
                             buckets=_LATENCY_BUCKETS, registry=r)
        self.e2e = Histogram(p + "e2e_request_latency_seconds", "Arrival to last token",
                             buckets=_LATENCY_BUCKETS, registry=r)
        self.queue_time = Histogram(p + "request_queue_time_seconds", "Arrival to first scheduled",
                                    buckets=_LATENCY_BUCKETS, registry=r)
        self._last_prefix_hits = 0
        self._last_preemptions = 0

    def update_snapshot(self, s: EngineSnapshot):
        self.requests_waiting.set(s.num_waiting)
        self.requests_running.set(s.num_running)
        self.kv_blocks_used.set(s.kv_blocks_used)
        self.kv_blocks_total.set(s.kv_blocks_total)
        if s.prefix_cache_hit_blocks > self._last_prefix_hits:
            self.prefix_cache_hit_blocks.inc(s.prefix_cache_hit_blocks - self._last_prefix_hits)
            self._last_prefix_hits = s.prefix_cache_hit_blocks
        if s.preemptions > self._last_preemptions:
            self.preemptions.inc(s.preemptions - self._last_preemptions)
            self._last_preemptions = s.preemptions

    def observe_step(self, kind: str, num_seqs: int, num_tokens: int, duration: float):
        self.steps.labels(kind).inc()
        self.step_batch_size.labels(kind).observe(num_seqs)
        self.step_tokens.labels(kind).observe(num_tokens)
        self.step_duration.labels(kind).observe(duration)

    def render(self) -> bytes:
        return generate_latest(self.registry)
