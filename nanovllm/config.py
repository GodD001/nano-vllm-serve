import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # --- nano-vllm-serve additions ---
    # "prefill_first": upstream's policy (default). "decode_first": every running sequence
    # decodes each step; leftover token budget goes to (chunked) prefill in the same step.
    scheduler_policy: str = "prefill_first"
    # decode_first only: cap on prefill tokens riding along with a step (TTFT <-> ITL knob).
    # -1 = max_num_batched_tokens.
    max_prefill_tokens_per_step: int = -1
    # Upper bound on KV-cache blocks (-1 = as many as gpu_memory_utilization allows). Lets
    # benchmarks give every system the same KV capacity.
    max_kvcache_blocks: int = -1

    def __post_init__(self):
        if self.scheduler_policy not in ("prefill_first", "decode_first"):
            raise ValueError(f"unknown scheduler_policy {self.scheduler_policy!r}")
        if self.scheduler_policy == "decode_first" and self.tensor_parallel_size != 1:
            # Mixed steps run decode sequences through the varlen path, which needs their
            # token ids; Sequence.__getstate__ only ships the last token to TP workers.
            raise ValueError("scheduler_policy='decode_first' requires tensor_parallel_size == 1")
        if self.max_prefill_tokens_per_step <= 0:
            self.max_prefill_tokens_per_step = self.max_num_batched_tokens
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
