from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SamplingTensors:
    """Batched sampling parameters (one row per sequence). Built once per step.

    Optional tensors are None when no row in the batch uses that feature, so the common
    case (plain temperature sampling, upstream defaults) runs upstream's exact kernel.
    """
    temperatures: torch.Tensor            # float32 [B]; greedy rows hold 1.0 (ignored)
    top_ks: torch.Tensor | None = None    # int64 [B]; disabled rows hold vocab size
    top_ps: torch.Tensor | None = None    # float32 [B]; disabled rows hold 1.0
    greedy: torch.Tensor | None = None    # bool [B]
    all_greedy: bool = False


@torch.compile
def _upstream_sample(logits: torch.Tensor, temperatures: torch.Tensor):
    # Unchanged from upstream nano-vLLM: Gumbel-max style sampling via exponential noise.
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return sample_tokens


def apply_top_k_top_p(logits: torch.Tensor, top_ks: torch.Tensor | None, top_ps: torch.Tensor | None):
    """Mask logits outside each row's top-k, then outside its top-p nucleus. Batched, no Python loop.

    top-p keeps the smallest prefix (by descending probability, after top-k) whose mass reaches p;
    the most likely token is always kept.
    """
    if top_ks is None and top_ps is None:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    if top_ks is not None:
        ranks = torch.arange(logits.size(-1), device=logits.device)
        sorted_logits.masked_fill_(ranks.unsqueeze(0) >= top_ks.unsqueeze(1), float("-inf"))
    if top_ps is not None:
        probs = sorted_logits.softmax(dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs
        sorted_logits.masked_fill_(mass_before >= top_ps.unsqueeze(1), float("-inf"))
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)


def _filtered_sample(logits: torch.Tensor, temperatures: torch.Tensor, top_ks, top_ps):
    logits = logits.div(temperatures.unsqueeze(dim=1))
    logits = apply_top_k_top_p(logits, top_ks, top_ps)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, sampling: SamplingTensors | torch.Tensor):
        if isinstance(sampling, torch.Tensor):  # upstream call signature: (logits, temperatures)
            sampling = SamplingTensors(temperatures=sampling)
        if sampling.all_greedy:
            return logits.argmax(dim=-1)
        if sampling.greedy is None and sampling.top_ks is None and sampling.top_ps is None:
            return _upstream_sample(logits, sampling.temperatures)
        logits = logits.float()
        sampled = _filtered_sample(logits, sampling.temperatures, sampling.top_ks, sampling.top_ps)
        if sampling.greedy is not None:
            sampled = torch.where(sampling.greedy, logits.argmax(dim=-1), sampled)
        return sampled
