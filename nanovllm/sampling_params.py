from dataclasses import dataclass, field


@dataclass(slots=True)
class SamplingParams:
    """Per-request sampling options.

    Defaults reproduce upstream nano-vLLM exactly (temperature sampling, no filtering).
    - temperature < 1e-5 (e.g. 0) -> greedy (argmax)
    - top_k == -1       -> disabled
    - top_p == 1.0      -> disabled
    - stop_token_ids    -> generation stops after emitting any of these (in addition to EOS
                           unless ignore_eos); the stop token is included in the output ids,
                           like EOS is upstream.
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = -1
    top_p: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        # ValueError (not assert) so validation survives `python -O`.
        if not self.temperature >= 0:
            raise ValueError("temperature must be >= 0 (0 = greedy)")
        if not self.max_tokens >= 1:
            raise ValueError("max_tokens must be >= 1")
        if not (self.top_k == -1 or self.top_k >= 1):
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def greedy(self) -> bool:
        # Tiny temperatures would overflow logits / T; treat them as greedy.
        return self.temperature < GREEDY_EPS


GREEDY_EPS = 1e-5
